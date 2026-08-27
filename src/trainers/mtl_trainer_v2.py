"""
Enhanced Multi-Task Learning Trainer
=====================================
Complete training pipeline for MTL experiments with:
- Configurable gradient manipulation methods
- Loss function composition
- Cross-validation support
- Data augmentation integration
- Comprehensive logging and checkpointing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path
import json
import time
import copy
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from ..models.gradient_methods import (
    GradientMethod, NoneGradient, PCGrad, GradNorm, 
    CAGrad, MGDA, UncertaintyWeighting, DWA, 
    create_gradient_method
)
from ..models.losses import MTLLoss, FocalLoss, MixupLoss, TASK_TYPE_REGISTRY
from ..data.augmentation import TimeSeriesAugmentor, mixup_data, cutmix_data
from ..data.dataset import GaitDataset
from ..data.normalization import FoldStandardizer
from ..evaluators.metrics import Evaluator


class MTLTrainer:
    """
    Enhanced Multi-Task Learning Trainer.
    
    Supports:
    - Any MTL model architecture (hard sharing, MMoE, PLE, etc.)
    - All gradient methods (PCGrad, GradNorm, CAGrad, MGDA, etc.)
    - Data augmentation (jitter, scaling, rotation, mixup, cutmix)
    - Focal loss / class-balanced loss for class imbalance
    - Hierarchy consistency loss
    - Cross-validation
    - Comprehensive logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[MTLLoss] = None,
        gradient_method: Optional[GradientMethod] = None,
        augmentor: Optional[TimeSeriesAugmentor] = None,
        device: str = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
        use_cutmix: bool = False,
        cutmix_alpha: float = 1.0,
        gradient_clip: float = 1.0,
        log_interval: int = 10,
        checkpoint_dir: Optional[str] = None,
        experiment_name: str = 'mtl_experiment',
        use_amp: bool = False,
        preprocessing: Optional[Dict[str, Any]] = None
    ):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn or MTLLoss()
        self.loss_fn = self.loss_fn.to(self.device)
        self.gradient_method = gradient_method or NoneGradient()
        self.augmentor = augmentor
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.use_cutmix = use_cutmix
        self.cutmix_alpha = cutmix_alpha
        self.gradient_clip = gradient_clip
        self.log_interval = log_interval
        self.experiment_name = experiment_name
        self.preprocessing = preprocessing
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.amp_dtype = torch.bfloat16 if self.use_amp and torch.cuda.is_bf16_supported() else torch.float16
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Move gradient method parameters to device if needed
        if hasattr(self.gradient_method, 'to'):
            self.gradient_method.to(self.device)
        if hasattr(self.gradient_method, 'parameters'):
            # Add uncertainty weighting params to optimizer
            extra_params = list(self.gradient_method.parameters())
            if extra_params:
                self.optimizer.add_param_group({'params': extra_params, 'lr': 0.025})
        
        # Evaluator
        self.evaluator = Evaluator()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'task_weights': [],
            'lr': []
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch with all configured strategies."""
        self.model.train()
        
        epoch_losses = defaultdict(float)
        n_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
        
        for batch_idx, (data, targets) in enumerate(pbar):
            data = data.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Data augmentation
            if self.augmentor is not None:
                data = self.augmentor(data)
            
            # Mixup / CutMix
            mixed_targets_a = None
            mixed_targets_b = None
            lam = None
            
            if self.use_mixup and np.random.random() < 0.5:
                data, mixed_targets_a, mixed_targets_b, lam = mixup_data(
                    data, targets, self.mixup_alpha
                )
            elif self.use_cutmix and np.random.random() < 0.5:
                data, mixed_targets_a, mixed_targets_b, lam = cutmix_data(
                    data, targets, self.cutmix_alpha
                )
            
            # Forward pass (with optional AMP autocast)
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(data)
                
                # Compute losses
                regression_mask = targets.get('regression', torch.zeros(1).to(self.device)) >= 0
                
                if lam is not None and mixed_targets_a is not None:
                    # Mixup/CutMix: compute loss with both label sets
                    losses_a, _ = self.loss_fn(outputs, mixed_targets_a, regression_mask)
                    losses_b, _ = self.loss_fn(outputs, mixed_targets_b, regression_mask)
                    losses = {
                        k: lam * losses_a.get(k, 0) + (1 - lam) * losses_b.get(k, 0)
                        for k in set(list(losses_a.keys()) + list(losses_b.keys()))
                    }
                    log_dict = {f'{k}_loss': v.item() for k, v in losses.items()}
                    log_dict['total_loss'] = sum(v.item() for v in losses.values())
                else:
                    losses, log_dict = self.loss_fn(outputs, targets, regression_mask)
            
            # Add soft sharing regularization loss to the losses dict
            if hasattr(self.model, 'get_regularization_loss'):
                reg_loss = self.model.get_regularization_loss()
                if reg_loss.item() > 0:
                    losses['regularization'] = reg_loss
                    log_dict['reg_loss'] = reg_loss.item()
            
            # Backward pass with gradient method
            total_loss = self.gradient_method.backward(losses, self.model, self.optimizer)
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.gradient_clip
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Accumulate losses
            for k, v in log_dict.items():
                epoch_losses[k] += v
            
            # Update progress bar
            pbar.set_postfix({
                'loss': log_dict.get('total_loss', 0),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Average losses
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        
        # Record task weights if available
        if hasattr(self.gradient_method, 'get_weights'):
            weights = self.gradient_method.get_weights()
            avg_losses['task_weights'] = weights.tolist() if isinstance(weights, np.ndarray) else weights
        
        return avg_losses
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        prefix: str = 'val'
    ) -> Dict[str, float]:
        """Evaluate model on a dataset."""
        self.model.eval()
        
        all_outputs = defaultdict(list)
        all_targets = defaultdict(list)
        all_probs = defaultdict(list)
        total_loss = 0
        n_batches = 0
        
        for data, targets in data_loader:
            data = data.to(self.device)
            targets_gpu = {k: v.to(self.device) for k, v in targets.items()}
            
            with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                outputs = self.model(data)
                
                # Compute loss
                regression_mask = targets_gpu.get('regression', torch.zeros(1).to(self.device)) >= 0
                losses, log_dict = self.loss_fn(outputs, targets_gpu, regression_mask)
            total_loss += log_dict.get('total_loss', 0)
            n_batches += 1
            
            # Collect predictions (cast to float32 for numpy compatibility)
            for task_name, output in outputs.items():
                output_cpu = output.float().cpu()
                task_type = TASK_TYPE_REGISTRY.get(task_name, 'classification')
                if task_type == 'classification':
                    # Classification: get predictions and probabilities
                    probs = F.softmax(output_cpu, dim=-1)
                    preds = probs.argmax(dim=-1)
                    all_outputs[task_name].append(preds)
                    all_probs[task_name].append(probs)
                else:
                    # Regression
                    all_outputs[task_name].append(output_cpu.squeeze(-1))
                
                if task_name in targets:
                    all_targets[task_name].append(targets[task_name])
        
        # Concatenate
        metrics = {f'{prefix}_loss': total_loss / max(n_batches, 1)}
        
        outputs_cat = {k: torch.cat(v).numpy() for k, v in all_outputs.items()}
        targets_cat = {k: torch.cat(v).numpy() for k, v in all_targets.items()}
        probs_cat = {k: torch.cat(v).numpy() for k, v in all_probs.items()}
        
        # Compute metrics using evaluator
        eval_results = self.evaluator.evaluate_all_tasks(
            outputs_cat, targets_cat, probs_cat
        )
        
        # Flatten metrics
        for task_name, task_metrics in eval_results.items():
            for metric_name, value in task_metrics.items():
                if metric_name == 'confusion_matrix':
                    continue
                if isinstance(value, (int, float)):
                    metrics[f'{prefix}_{task_name}_{metric_name}'] = value
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int = 100,
        early_stopping_patience: int = 15,
        save_best: bool = True,
        monitor_metric: str = 'val_loss',
        monitor_mode: str = 'min',
        verbose: bool = True
    ) -> Dict:
        """
        Full training loop with early stopping and checkpointing.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            n_epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            monitor_metric: Metric to monitor for early stopping
            monitor_mode: 'min' or 'max'
            verbose: Print training progress
            
        Returns:
            Training history dictionary
        """
        best_metric = float('inf') if monitor_mode == 'min' else float('-inf')
        patience_counter = 0
        best_epoch = 0
        best_model_state = None
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Training: {self.experiment_name}")
            print(f"Model: {self.model.__class__.__name__}")
            print(f"Gradient method: {self.gradient_method.name}")
            print(f"Device: {self.device}")
            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Parameters: {params:,}")
            print(f"{'='*60}")
        
        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader, prefix='val')
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get(monitor_metric, val_metrics['val_loss']))
                else:
                    self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Store history
            self.history['train_loss'].append(train_metrics.get('total_loss', 0))
            self.history['val_loss'].append(val_metrics['val_loss'])
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            if 'task_weights' in train_metrics:
                self.history['task_weights'].append(train_metrics['task_weights'])
            
            # Logging
            if verbose:
                log_str = (
                    f"Epoch {epoch:3d}/{n_epochs} "
                    f"| Train Loss: {train_metrics.get('total_loss', 0):.4f} "
                    f"| Val Loss: {val_metrics['val_loss']:.4f} "
                    f"| Time: {epoch_time:.1f}s"
                )
                
                # Add classification task accuracies (dynamic)
                task_accs = []
                for task in TASK_TYPE_REGISTRY:
                    if TASK_TYPE_REGISTRY[task] == 'classification':
                        key = f'val_{task}_accuracy'
                        if key in val_metrics:
                            task_accs.append(f"{task}={val_metrics[key]:.3f}")
                if task_accs:
                    log_str += f" | Acc: {', '.join(task_accs[:4])}"  # show max 4
                
                # Add regression MAEs (dynamic)
                reg_maes = []
                for task in TASK_TYPE_REGISTRY:
                    if TASK_TYPE_REGISTRY[task] == 'regression':
                        key = f'val_{task}_mae'
                        if key in val_metrics:
                            reg_maes.append(f"{task}={val_metrics[key]:.4f}")
                if reg_maes:
                    log_str += f" | MAE: {', '.join(reg_maes[:3])}"  # show max 3
                
                print(log_str)
            
            # Early stopping check
            current_metric = val_metrics.get(monitor_metric, val_metrics['val_loss'])
            
            is_better = (
                (monitor_mode == 'min' and current_metric < best_metric - 1e-6) or
                (monitor_mode == 'max' and current_metric > best_metric + 1e-6)
            )
            
            if is_better:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                best_model_state = copy.deepcopy(self.model.state_dict())
                
                if save_best and self.checkpoint_dir:
                    self._save_checkpoint('best_model.pt', epoch, val_metrics)
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch} "
                          f"(best epoch: {best_epoch}, {monitor_metric}: {best_metric:.4f})")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            if verbose:
                print(f"Restored best model from epoch {best_epoch}")
        
        self.history['best_epoch'] = best_epoch
        self.history['best_metric'] = best_metric
        
        return self.history
    
    @torch.no_grad()
    def predict(self, data_loader: DataLoader) -> Dict[str, np.ndarray]:
        """Generate predictions for all tasks."""
        self.model.eval()
        
        all_preds = defaultdict(list)
        all_probs = defaultdict(list)
        
        for data, _ in data_loader:
            data = data.to(self.device)
            outputs = self.model(data)
            
            for task_name, output in outputs.items():
                output_cpu = output.cpu()
                task_type = TASK_TYPE_REGISTRY.get(task_name, 'classification')
                if task_type == 'classification':
                    probs = F.softmax(output_cpu, dim=-1)
                    all_preds[task_name].append(probs.argmax(dim=-1))
                    all_probs[f'{task_name}_proba'].append(probs)
                else:
                    all_preds[task_name].append(output_cpu.squeeze(-1))
        
        results = {}
        for k, v in all_preds.items():
            results[k] = torch.cat(v).numpy()
        for k, v in all_probs.items():
            results[k] = torch.cat(v).numpy()
        
        return results
    
    def _save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            'experiment_name': self.experiment_name,
            'preprocessing': self.preprocessing,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.preprocessing = checkpoint.get('preprocessing', self.preprocessing)
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint.get('epoch', 0), checkpoint.get('metrics', {})


# ============================================================================
# Cross-Validation Runner
# ============================================================================

class MTLCrossValidator:
    """
    Cross-validation runner for MTL experiments.
    
    Performs subject-wise stratified K-fold cross-validation,
    training a new model for each fold and aggregating results.
    """
    
    def __init__(
        self,
        model_factory,
        optimizer_factory,
        scheduler_factory=None,
        loss_fn: Optional[MTLLoss] = None,
        gradient_method_name: str = 'none',
        gradient_method_kwargs: Optional[Dict] = None,
        n_folds: int = 5,
        device: str = None,
        augmentor: Optional[TimeSeriesAugmentor] = None,
        use_mixup: bool = False,
        checkpoint_dir: Optional[str] = None,
        experiment_name: str = 'mtl_cv',
        split_mode: str = 'subject_wise',
        use_amp: bool = False,
        num_workers: int = 0,
        normalize_inputs: bool = True
    ):
        """
        Args:
            model_factory: Callable that returns a new model instance
            optimizer_factory: Callable(model) that returns an optimizer
            scheduler_factory: Optional callable(optimizer) that returns a scheduler
            loss_fn: Loss function
            split_mode: 'subject_wise' or 'trial_wise'
            gradient_method_name: Name of gradient method
            gradient_method_kwargs: Kwargs for gradient method
            n_folds: Number of CV folds
            use_amp: Enable mixed precision training (bf16/fp16)
            num_workers: DataLoader num_workers for parallel data loading
            normalize_inputs: Fit one channel standardizer on each training fold
        """
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.loss_fn = loss_fn
        self.gradient_method_name = gradient_method_name
        self.gradient_method_kwargs = gradient_method_kwargs or {}
        self.n_folds = n_folds
        self.device = device
        self.augmentor = augmentor
        self.use_mixup = use_mixup
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.experiment_name = experiment_name
        self.split_mode = split_mode  # 'subject_wise' or 'trial_wise'
        self.use_amp = use_amp
        self.num_workers = num_workers
        self.normalize_inputs = normalize_inputs
        
        self.fold_results = []

    def _make_fold_datasets(self, dataset, metadata, train_indices, val_indices, test_indices):
        """Create independent split datasets with training-only preprocessing."""
        if not isinstance(dataset, GaitDataset):
            raise TypeError("MTLCrossValidator expects a GaitDataset with raw windows")

        raw_data = dataset.data.detach().cpu().numpy()
        label_arrays = {
            name: values.detach().cpu().numpy()
            for name, values in dataset.labels.items()
        }
        train_indices = np.asarray(train_indices, dtype=int)
        val_indices = np.asarray(val_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)

        train_data = raw_data[train_indices]
        val_data = raw_data[val_indices]
        test_data = raw_data[test_indices]
        if self.normalize_inputs:
            standardizer = FoldStandardizer()
            train_data = standardizer.fit_transform(train_data)
            val_data = standardizer.transform(val_data)
            test_data = standardizer.transform(test_data)
            preprocessing = standardizer.state_dict()
        else:
            preprocessing = {
                "scope": "disabled",
                "channels": int(raw_data.shape[2]),
                "fit_sample_count": 0,
            }

        def make_split(indices, split_data):
            return GaitDataset(
                split_data,
                {name: values[indices] for name, values in label_arrays.items()},
                metadata.iloc[indices].reset_index(drop=True),
                transform=dataset.transform,
            )

        return (
            make_split(train_indices, train_data),
            make_split(val_indices, val_data),
            make_split(test_indices, test_data),
            preprocessing,
        )
    
    def run(
        self,
        dataset,
        metadata,
        batch_size: int = 32,
        n_epochs: int = 100,
        early_stopping_patience: int = 15,
        monitor_metric: str = 'val_binary_accuracy',
        monitor_mode: str = 'max',
        verbose: bool = True
    ) -> Dict:
        """
        Run cross-validation.
        
        Args:
            dataset: Full GaitDataset
            metadata: DataFrame with subject and label columns
            batch_size: Batch size
            n_epochs: Max epochs per fold
            early_stopping_patience: Patience for early stopping
            monitor_metric: Metric to monitor for early stopping
            monitor_mode: 'max' or 'min'
            
        Returns:
            Summary results dictionary
        """
        if self.split_mode == 'trial_wise':
            return self._run_trial_wise(dataset, metadata, batch_size, n_epochs, 
                                        early_stopping_patience, verbose)
        
        # Default: subject_wise splitting
        # Get subjects and their stratification labels
        subjects = metadata['subject'].unique()
        subject_labels = []
        for subj in subjects:
            mask = metadata['subject'] == subj
            # Use fine label for stratification
            label = metadata.loc[mask, 'pathology'].iloc[0]
            subject_labels.append(label)
        
        subject_labels = np.array(subject_labels)
        
        # Handle single fold (no cross-validation, simple train/test split)
        if self.n_folds == 1:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            fold_splits = list(sss.split(subjects, subject_labels))
        else:
            # Stratified K-fold on subjects
            skf = StratifiedKFold(
                n_splits=self.n_folds, shuffle=True, random_state=42
            )
            fold_splits = list(skf.split(subjects, subject_labels))
        
        all_fold_metrics = []
        
        for fold, (train_subj_idx, test_subj_idx) in enumerate(fold_splits):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  FOLD {fold + 1}/{self.n_folds}")
                print(f"{'='*60}")
            
            outer_train_subjects = subjects[train_subj_idx]
            outer_train_labels = subject_labels[train_subj_idx]
            inner_split = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=42 + fold
            )
            inner_train_idx, val_subj_idx = next(
                inner_split.split(outer_train_subjects, outer_train_labels)
            )
            train_subjects = set(outer_train_subjects[inner_train_idx])
            val_subjects = set(outer_train_subjects[val_subj_idx])
            test_subjects = set(subjects[test_subj_idx])
            
            # Create index masks
            train_mask = metadata['subject'].isin(train_subjects).values
            val_mask = metadata['subject'].isin(val_subjects).values
            test_mask = metadata['subject'].isin(test_subjects).values
            
            train_indices = np.where(train_mask)[0].tolist()
            val_indices = np.where(val_mask)[0].tolist()
            test_indices = np.where(test_mask)[0].tolist()
            
            train_dataset, val_dataset, test_dataset, preprocessing = (
                self._make_fold_datasets(
                    dataset, metadata, train_indices, val_indices, test_indices
                )
            )
            
            # Create dataloaders
            _nw = self.num_workers
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0)
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0)
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0)
            )
            
            if verbose:
                print(f"  Train: {len(train_indices)} samples "
                      f"({len(train_subjects)} subjects)")
                print(f"  Val:   {len(val_indices)} samples "
                      f"({len(val_subjects)} subjects)")
                print(f"  Test:  {len(test_indices)} samples "
                      f"({len(test_subjects)} subjects)")
            
            # Create fresh model and optimizer for this fold
            model = self.model_factory()
            optimizer = self.optimizer_factory(model)
            scheduler = self.scheduler_factory(optimizer) if self.scheduler_factory else None
            
            # Create gradient method
            gradient_method = create_gradient_method(
                self.gradient_method_name, **self.gradient_method_kwargs
            )
            
            # Checkpoint dir for this fold
            fold_ckpt_dir = None
            if self.checkpoint_dir:
                fold_ckpt_dir = str(self.checkpoint_dir / f'fold_{fold + 1}')
            
            # Create trainer
            trainer = MTLTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=self.loss_fn or MTLLoss(),
                gradient_method=gradient_method,
                augmentor=self.augmentor,
                device=self.device,
                use_mixup=self.use_mixup,
                checkpoint_dir=fold_ckpt_dir,
                experiment_name=f'{self.experiment_name}_fold{fold + 1}',
                use_amp=self.use_amp,
                preprocessing=preprocessing
            )
            
            # Train - use configurable metric for early stopping
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
                save_best=True,
                monitor_metric=monitor_metric,
                monitor_mode=monitor_mode,
                verbose=verbose
            )
            
            # Test
            test_metrics = trainer.evaluate(test_loader, prefix='test')
            
            if verbose:
                print(f"\n  Fold {fold + 1} Test Results:")
                for k, v in sorted(test_metrics.items()):
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
            
            test_metrics['fold'] = fold + 1
            test_metrics['best_epoch'] = history.get('best_epoch', 0)
            all_fold_metrics.append(test_metrics)
        
        # Aggregate results
        summary = self._aggregate_results(all_fold_metrics)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  CROSS-VALIDATION SUMMARY ({self.n_folds} folds)")
            print(f"{'='*60}")
            for key, (mean, std) in sorted(summary.items()):
                if key.startswith('test_'):
                    print(f"  {key}: {mean:.4f} +/- {std:.4f}")
        
        return {
            'fold_results': all_fold_metrics,
            'summary': summary,
            'experiment_name': self.experiment_name,
            'n_folds': self.n_folds,
            'gradient_method': self.gradient_method_name
        }
    
    def _aggregate_results(
        self, fold_results: List[Dict]
    ) -> Dict[str, Tuple[float, float]]:
        """Compute mean and std across folds."""
        summary = {}
        
        # Collect all numeric metrics
        all_keys = set()
        for result in fold_results:
            for k, v in result.items():
                if isinstance(v, (int, float)) and k not in ['fold', 'best_epoch']:
                    all_keys.add(k)
        
        for key in all_keys:
            values = [r[key] for r in fold_results if key in r]
            if values:
                summary[key] = (np.mean(values), np.std(values))
        
        return summary
    
    def _run_trial_wise(
        self,
        dataset,
        metadata,
        batch_size: int = 32,
        n_epochs: int = 100,
        early_stopping_patience: int = 15,
        verbose: bool = True
    ) -> Dict:
        """Run with trial-wise splitting and training-only preprocessing."""
        
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        # Use fine labels for stratification
        strat_labels = metadata['pathology'].values
        
        if self.n_folds == 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
            fold_splits = list(sss.split(indices, strat_labels))
        else:
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            fold_splits = list(skf.split(indices, strat_labels))
        
        all_fold_metrics = []
        
        for fold, (train_val_idx, test_idx) in enumerate(fold_splits):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  FOLD {fold + 1}/{self.n_folds} (trial-wise)")
                print(f"{'='*60}")
            
            # Deterministic pathology-stratified train/validation split
            inner_split = StratifiedShuffleSplit(
                n_splits=1, test_size=0.15, random_state=42 + fold
            )
            train_rel_idx, val_rel_idx = next(
                inner_split.split(train_val_idx, strat_labels[train_val_idx])
            )
            train_idx = train_val_idx[train_rel_idx]
            val_idx = train_val_idx[val_rel_idx]
            
            train_indices = train_idx.tolist()
            val_indices = val_idx.tolist()
            test_indices = test_idx.tolist()
            
            train_dataset, val_dataset, test_dataset, preprocessing = (
                self._make_fold_datasets(
                    dataset, metadata, train_indices, val_indices, test_indices
                )
            )
            _nw = self.num_workers
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True,
                num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0)
            )
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False,
                num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0)
            )
            test_loader = DataLoader(
                test_dataset, batch_size=batch_size, shuffle=False,
                num_workers=_nw, pin_memory=True, persistent_workers=(_nw > 0)
            )
            
            if verbose:
                print(f"  Train: {len(train_indices)} samples")
                print(f"  Val:   {len(val_indices)} samples")
                print(f"  Test:  {len(test_indices)} samples")
            
            # Create fresh model and optimizer
            model = self.model_factory()
            optimizer = self.optimizer_factory(model)
            scheduler = self.scheduler_factory(optimizer) if self.scheduler_factory else None
            gradient_method = create_gradient_method(
                self.gradient_method_name, **self.gradient_method_kwargs
            )
            
            fold_ckpt_dir = None
            if self.checkpoint_dir:
                fold_ckpt_dir = str(self.checkpoint_dir / f'fold_{fold + 1}')
            
            trainer = MTLTrainer(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_fn=self.loss_fn or MTLLoss(),
                gradient_method=gradient_method,
                augmentor=self.augmentor,
                device=self.device,
                use_mixup=self.use_mixup,
                checkpoint_dir=fold_ckpt_dir,
                experiment_name=f'{self.experiment_name}_fold{fold + 1}',
                use_amp=self.use_amp,
                preprocessing=preprocessing
            )
            
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                n_epochs=n_epochs,
                early_stopping_patience=early_stopping_patience,
                save_best=True,
                verbose=verbose
            )
            
            test_metrics = trainer.evaluate(test_loader, prefix='test')
            
            if verbose:
                print(f"\n  Fold {fold + 1} Test Results:")
                for k, v in sorted(test_metrics.items()):
                    if isinstance(v, float):
                        print(f"    {k}: {v:.4f}")
            
            test_metrics['fold'] = fold + 1
            test_metrics['best_epoch'] = history.get('best_epoch', 0)
            all_fold_metrics.append(test_metrics)
        
        summary = self._aggregate_results(all_fold_metrics)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"  CROSS-VALIDATION SUMMARY ({self.n_folds} folds, trial-wise)")
            print(f"{'='*60}")
            for key, (mean, std) in sorted(summary.items()):
                if key.startswith('test_'):
                    print(f"  {key}: {mean:.4f} +/- {std:.4f}")
        
        return {
            'fold_results': all_fold_metrics,
            'summary': summary,
            'experiment_name': self.experiment_name,
            'n_folds': self.n_folds,
            'gradient_method': self.gradient_method_name
        }


# ============================================================================
# Helper Functions
# ============================================================================

def create_optimizer(
    model: nn.Module,
    optimizer_type: str = 'AdamW',
    lr: float = 0.001,
    weight_decay: float = 0.01,
    **kwargs
) -> torch.optim.Optimizer:
    """Create optimizer with optional per-group learning rates."""
    
    # Separate encoder and head parameters for different learning rates
    if hasattr(model, 'encoder'):
        encoder_params = list(model.encoder.parameters())
        head_params = [p for p in model.parameters() if not any(
            p is ep for ep in encoder_params
        )]
        param_groups = [
            {'params': encoder_params, 'lr': lr},
            {'params': head_params, 'lr': lr * 2}  # Higher LR for heads
        ]
    else:
        param_groups = [{'params': model.parameters(), 'lr': lr}]
    
    if optimizer_type == 'AdamW':
        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        return torch.optim.Adam(param_groups, weight_decay=weight_decay)
    elif optimizer_type == 'SGD':
        return torch.optim.SGD(
            param_groups, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'CosineAnnealingWarmRestarts',
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler with optional warmup."""
    warmup_epochs = kwargs.pop('warmup_epochs', 0)
    
    # Build the main scheduler
    if scheduler_type == 'CosineAnnealingWarmRestarts':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 2),
            eta_min=kwargs.get('eta_min', 1e-5)
        )
    elif scheduler_type == 'CosineAnnealing':
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100),
            eta_min=kwargs.get('eta_min', 1e-5)
        )
    elif scheduler_type == 'StepLR':
        main_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_type == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'min'),
            patience=kwargs.get('patience', 10),
            factor=kwargs.get('factor', 0.5)
        )
    elif scheduler_type == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.01),
            total_steps=kwargs.get('total_steps', 1000)
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    # Wrap with warmup if requested
    if warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    
    return main_scheduler
