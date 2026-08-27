"""
Multi-Task Learning Experiments Runner
=======================================
Main script for configurable MTL and STL experiments.

Usage:
    # Run all experiments
    python scripts/run_mtl_experiments.py
    
    # Run specific experiment
    python scripts/run_mtl_experiments.py --experiment hard_sharing
    
    # Run with specific encoder
    python scripts/run_mtl_experiments.py --encoder transformer
    
    # Run ablation study
    python scripts/run_mtl_experiments.py --ablation
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import GaitDataLoader, GaitDataset
from src.data.augmentation import TimeSeriesAugmentor
from src.data.normalization import FoldStandardizer
from src.models.mtl_models import (
    HardSharingMTL, SoftSharingMTL, CrossStitchMTL,
    MMoEMTL, PLEMTL, MTANMTL, create_mtl_model, DEFAULT_TASKS
)
from src.models.hierarchical_mtl import HierarchicalMTLModel
from src.models.losses import MTLLoss, FocalLoss, ClassBalancedLoss
from src.models.gradient_methods import create_gradient_method
from src.trainers.mtl_trainer_v2 import (
    MTLTrainer, MTLCrossValidator, create_optimizer, create_scheduler
)
from src.evaluators.metrics import Evaluator
from src.utils.helpers import set_seed, load_config, save_results, get_device


# ============================================================================
# Experiment Configurations
# ============================================================================

# Maps experiment names from YAML to (model_type, gradient_method) pairs
EXPERIMENT_CONFIGS = {
    # Baseline: Single Task Learning (placeholder - uses separate training)
    'single_task': {
        'model_type': 'hard_sharing',
        'gradient_method': 'none',
        'description': 'Single Task Learning Baseline'
    },
    
    # MTL Methods
    'hard_sharing': {
        'model_type': 'hard_sharing',
        'gradient_method': 'none',
        'description': 'Hard Parameter Sharing MTL'
    },
    'soft_sharing': {
        'model_type': 'soft_sharing',
        'gradient_method': 'none',
        'description': 'Soft Parameter Sharing MTL'
    },
    'cross_stitch': {
        'model_type': 'cross_stitch',
        'gradient_method': 'none',
        'description': 'Cross-Stitch Networks'
    },
    'mmoe': {
        'model_type': 'mmoe',
        'gradient_method': 'none',
        'description': 'Multi-gate Mixture of Experts'
    },
    'ple': {
        'model_type': 'ple',
        'gradient_method': 'none',
        'description': 'Progressive Layered Extraction'
    },
    'mtan': {
        'model_type': 'mtan',
        'gradient_method': 'none',
        'description': 'Multi-Task Attention Network'
    },
    
    # Gradient-based MTL Methods (use hard sharing model)
    'uncertainty_weighting': {
        'model_type': 'hard_sharing',
        'gradient_method': 'uncertainty',
        'gradient_kwargs': {'num_tasks': 10},
        'description': 'Uncertainty Weighting (Kendall 2018)'
    },
    'gradnorm': {
        'model_type': 'hard_sharing',
        'gradient_method': 'gradnorm',
        'gradient_kwargs': {'num_tasks': 10, 'alpha': 1.5},
        'description': 'GradNorm (Chen 2018)'
    },
    'pcgrad': {
        'model_type': 'hard_sharing',
        'gradient_method': 'pcgrad',
        'description': 'PCGrad (Yu 2020)'
    },
    'cagrad': {
        'model_type': 'hard_sharing',
        'gradient_method': 'cagrad',
        'gradient_kwargs': {'c': 0.4},
        'description': 'CAGrad (Liu 2021)'
    },
    'mgda': {
        'model_type': 'hard_sharing',
        'gradient_method': 'mgda',
        'description': 'MGDA (Sener 2018)'
    },
    'dwa': {
        'model_type': 'hard_sharing',
        'gradient_method': 'dwa',
        'gradient_kwargs': {'num_tasks': 10, 'temperature': 2.0},
        'description': 'Dynamic Weight Average (Liu 2019)'
    },
    
    # MMoE + Gradient Method Combos
    'mmoe_dwa': {
        'model_type': 'mmoe',
        'gradient_method': 'dwa',
        'gradient_kwargs': {'num_tasks': 10, 'temperature': 2.0},
        'description': 'MMoE + Dynamic Weight Average'
    },
    'mmoe_cagrad': {
        'model_type': 'mmoe',
        'gradient_method': 'cagrad',
        'gradient_kwargs': {'c': 0.4},
        'description': 'MMoE + CAGrad'
    },
    'mmoe_pcgrad': {
        'model_type': 'mmoe',
        'gradient_method': 'pcgrad',
        'description': 'MMoE + PCGrad'
    },
    'mtan_dwa': {
        'model_type': 'mtan',
        'gradient_method': 'dwa',
        'gradient_kwargs': {'num_tasks': 10, 'temperature': 2.0},
        'description': 'MTAN + Dynamic Weight Average'
    },
    'mmoe_dwa_hierarchy': {
        'model_type': 'mmoe',
        'gradient_method': 'dwa',
        'gradient_kwargs': {'num_tasks': 10, 'temperature': 2.0},
        'use_hierarchy_loss': True,
        'use_focal_loss': True,
        'description': 'MMoE + DWA + Hierarchy Consistency Loss'
    },
    'mmoe_dwa_t1': {
        'model_type': 'mmoe',
        'gradient_method': 'dwa',
        'gradient_kwargs': {'num_tasks': 10, 'temperature': 1.0},
        'description': 'MMoE + DWA (temperature=1.0, sharper weighting)'
    },
    
    # Our Method: Hierarchical MTL
    'hierarchical_mtl': {
        'model_type': 'hierarchical',
        'gradient_method': 'pcgrad',
        'use_hierarchy_loss': True,
        'use_focal_loss': True,
        'description': 'Hierarchical MTL (Ours)'
    },
}


# ============================================================================
# Data Loading
# ============================================================================

def load_data(
    config: dict,
    data_path: str = None,
    verbose: bool = True
) -> tuple:
    """Load raw windows; scaling is deferred until fold indices are fixed."""
    data_config = config.get('data', {})
    base_path = data_path or os.environ.get('HMTL_DATA_PATH') or data_config.get(
        'base_path', 'data'
    )
    base_path = Path(base_path).expanduser()
    if not base_path.is_absolute():
        base_path = project_root / base_path
    base_path = str(base_path.resolve())
    
    if verbose:
        print(f"Loading data from: {base_path}")
    
    loader = GaitDataLoader(base_path)
    
    # Get sensor/signal configuration
    sensors = data_config.get('sensors', ['LB', 'LF', 'RF'])
    signals = data_config.get('signals', ['acc', 'freeacc', 'gyr'])
    
    # Handle sensor config from YAML
    if isinstance(sensors, list) and len(sensors) > 0:
        if isinstance(sensors[0], dict):
            sensors = [s.get('name', s) for s in sensors]
    
    preprocessing = data_config.get('preprocessing', {})
    window_size = preprocessing.get('window_size', 200)
    stride = preprocessing.get('window_stride', 100)
    
    X, y, metadata = loader.prepare_dataset(
        sensors=sensors if isinstance(sensors, list) else None,
        signals=signals if isinstance(signals, list) else None,
        window_size=window_size,
        stride=stride,
        normalize=False,
        verbose=verbose
    )
    
    return X, y, metadata


def create_dataset(X, y, metadata):
    """Create PyTorch dataset from arrays."""
    dataset = GaitDataset(
        data=X,
        labels=y,
        metadata=metadata
    )
    return dataset


# ============================================================================
# Single Experiment Runner
# ============================================================================

def run_single_experiment(
    exp_name: str,
    exp_config: dict,
    X: np.ndarray,
    y: dict,
    metadata: pd.DataFrame,
    training_config: dict,
    encoder_type: str = 'transformer',
    n_folds: int = 5,
    results_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run a single MTL experiment with cross-validation.
    
    Args:
        exp_name: Experiment name
        exp_config: Experiment configuration
        X: Input data (N, T, C)
        y: Label dictionary
        metadata: Sample metadata
        training_config: Training hyperparameters
        encoder_type: Encoder architecture
        n_folds: Number of CV folds
        results_dir: Directory to save results
        verbose: Print progress
        
    Returns:
        Results dictionary
    """
    print(f"\n{'#'*60}")
    print(f"# Experiment: {exp_config['description']}")
    if exp_config['model_type'] == 'mmoe':
        print("# Model: mmoe | Shared experts: Conv1D")
    else:
        print(f"# Model: {exp_config['model_type']} | Encoder: {encoder_type}")
    print(f"# Gradient: {exp_config.get('gradient_method', 'none')}")
    print(f"{'#'*60}")
    
    model_type = exp_config['model_type']
    gradient_method_name = exp_config.get('gradient_method', 'none')
    gradient_kwargs = exp_config.get('gradient_kwargs', {})
    
    input_channels = X.shape[2]
    
    # Determine loss configuration
    use_focal = exp_config.get('use_focal_loss', True)
    use_hierarchy = exp_config.get('use_hierarchy_loss', False)
    
    # Compute class distribution for class-balanced loss
    fine_counts = np.bincount(y['fine'].astype(int), minlength=8).tolist()
    
    # Loss function
    loss_fn = MTLLoss(
        use_focal_loss=use_focal,
        focal_gamma=2.0,
        use_hierarchy_loss=use_hierarchy,
        hierarchy_weight=0.1,
        samples_per_class=fine_counts if use_focal else None,
        label_smoothing=training_config.get('label_smoothing', 0.1),
        regression_weight=training_config.get('reg_weight', 2.0),
        use_mse_loss=training_config.get('use_mse_loss', True)
    )
    
    # Augmentor
    aug_p = training_config.get('aug_p', 0.5)
    augmentor = TimeSeriesAugmentor(
        jitter_sigma=training_config.get('jitter_sigma', 0.03),
        scale_sigma=training_config.get('scale_sigma', 0.1),
        rotation_range=training_config.get('rotation_range', 0.05),
        magnitude_warp_sigma=training_config.get('magnitude_warp_sigma', 0.1),
        p=aug_p
    ) if aug_p > 0 else None
    
    # Model factory
    def model_factory():
        if model_type == 'hierarchical':
            return HierarchicalMTLModel(
                encoder_type=encoder_type,
                input_channels=input_channels,
                use_hierarchy_constraint=True,
                use_uncertainty_weighting=True,
                hierarchy_loss_weight=0.1
            )
        else:
            # Collect encoder kwargs for architecture tuning
            extra_kwargs = {}
            if model_type != 'mmoe' and encoder_type == 'transformer':
                for k in ['d_model', 'nhead', 'num_layers', 'dim_feedforward']:
                    if k in training_config:
                        extra_kwargs[k] = training_config[k]
            head_hidden = training_config.get('head_hidden_dim', 64)
            model_kwargs = {
                'model_type': model_type,
                'input_channels': input_channels,
                'tasks': DEFAULT_TASKS,
                'dropout': training_config.get('dropout', 0.35),
                'head_hidden_dim': head_hidden,
                **extra_kwargs,
            }
            if model_type == 'mmoe':
                return create_mtl_model(**model_kwargs)
            return create_mtl_model(encoder_type=encoder_type, **model_kwargs)
    
    # Optimizer factory
    def optimizer_factory(model):
        optimizer_config = training_config.get('optimizer', {})
        if isinstance(optimizer_config, dict):
            optimizer_type = optimizer_config.get('type', 'AdamW')
            lr = training_config.get('lr', optimizer_config.get('lr', 0.001))
            weight_decay = training_config.get(
                'weight_decay', optimizer_config.get('weight_decay', 0.01)
            )
        else:
            optimizer_type = optimizer_config
            lr = training_config.get('lr', 0.001)
            weight_decay = training_config.get('weight_decay', 0.01)

        return create_optimizer(
            model,
            optimizer_type=optimizer_type,
            lr=lr,
            weight_decay=weight_decay
        )
    
    # Scheduler factory
    def scheduler_factory(optimizer):
        scheduler_config = training_config.get('scheduler', {})
        if isinstance(scheduler_config, dict):
            scheduler_type = scheduler_config.get(
                'type', 'CosineAnnealingWarmRestarts'
            )
            scheduler_kwargs = {
                k: v for k, v in scheduler_config.items() if k != 'type'
            }
        else:
            scheduler_type = scheduler_config
            scheduler_kwargs = {}

        if not scheduler_kwargs and scheduler_type == 'CosineAnnealingWarmRestarts':
            scheduler_kwargs = {'T_0': 10, 'T_mult': 2, 'eta_min': 1e-5}
        
        # Add warmup epochs
        warmup_epochs = training_config.get('warmup_epochs', 0)
        if warmup_epochs > 0:
            scheduler_kwargs['warmup_epochs'] = warmup_epochs

        return create_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            **scheduler_kwargs
        )
    
    # Create dataset
    dataset = create_dataset(X, y, metadata)
    
    # Checkpoint directory
    ckpt_dir = None
    if results_dir:
        ckpt_dir = str(Path(results_dir) / exp_name / 'checkpoints')
    
    # Cross-validation
    cv_runner = MTLCrossValidator(
        model_factory=model_factory,
        optimizer_factory=optimizer_factory,
        scheduler_factory=scheduler_factory,
        loss_fn=loss_fn,
        gradient_method_name=gradient_method_name,
        gradient_method_kwargs=gradient_kwargs,
        n_folds=n_folds,
        device=str(get_device()),
        augmentor=augmentor,
        use_mixup=training_config.get('use_mixup', False),
        checkpoint_dir=ckpt_dir,
        experiment_name=exp_name,
        split_mode=training_config.get('split_mode', 'subject_wise'),
        use_amp=training_config.get('use_amp', False),
        num_workers=training_config.get('num_workers', 0),
        normalize_inputs=training_config.get('normalize_inputs', True)
    )
    
    results = cv_runner.run(
        dataset=dataset,
        metadata=metadata,
        batch_size=training_config.get('batch_size', 32),
        n_epochs=training_config.get('max_epochs', 100),
        early_stopping_patience=training_config.get('early_stopping_patience', 15),
        monitor_metric=training_config.get('monitor_metric', 'val_binary_accuracy'),
        monitor_mode=training_config.get('monitor_mode', 'max'),
        verbose=verbose
    )
    
    # Save results
    if results_dir:
        save_dir = Path(results_dir) / exp_name
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_data = {
            'experiment': exp_name,
            'description': exp_config['description'],
            'model_type': model_type,
            'encoder_type': encoder_type,
            'gradient_method': gradient_method_name,
            'n_folds': n_folds,
            'summary': {
                k: {'mean': float(v[0]), 'std': float(v[1])}
                for k, v in results['summary'].items()
            },
            'fold_results': [
                {k: float(v) if isinstance(v, (float, np.floating)) else v
                 for k, v in fold.items()}
                for fold in results['fold_results']
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        save_results(summary_data, str(save_dir / 'results.json'))
        
        # Save summary CSV
        summary_rows = []
        for k, v in results['summary'].items():
            summary_rows.append({
                'metric': k,
                'mean': v[0],
                'std': v[1]
            })
        pd.DataFrame(summary_rows).to_csv(
            save_dir / 'summary.csv', index=False
        )
        
        print(f"\nResults saved to: {save_dir}")
    
    return results


# ============================================================================
# Single Task Learning Baseline
# ============================================================================

def run_stl_experiment(
    X: np.ndarray,
    y: dict,
    metadata: pd.DataFrame,
    training_config: dict,
    encoder_type: str = 'transformer',
    n_folds: int = 5,
    results_dir: str = None,
    verbose: bool = True
) -> dict:
    """
    Run Single Task Learning baseline.
    Trains a separate model for each task independently.
    """
    from src.models.deep_models import SingleTaskModel
    from torch.utils.data import DataLoader
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
    
    print(f"\n{'#'*60}")
    print(f"# Experiment: Single Task Learning (STL) Baseline")
    print(f"# Trains a separate model for each task independently")
    print(f"{'#'*60}")
    
    input_channels = X.shape[2]
    device = get_device()
    
    task_configs = {
        'binary':         {'num_classes': 2, 'type': 'classification'},
        'coarse':         {'num_classes': 3, 'type': 'classification'},
        'fine':           {'num_classes': 8, 'type': 'classification'},
        'regression':     {'num_classes': 1, 'type': 'regression'},
        'vga_class':      {'num_classes': 5, 'type': 'classification'},
        'vga_regression': {'num_classes': 1, 'type': 'regression'},
        'gender':         {'num_classes': 2, 'type': 'classification'},
        'age':            {'num_classes': 1, 'type': 'regression'},
        'tug':            {'num_classes': 1, 'type': 'regression'},
        'neuro_fine':     {'num_classes': 4, 'type': 'classification'},
    }
    
    all_results = {}
    
    for task_name, task_config in task_configs.items():
        is_regression = (task_config['type'] == 'regression')
        print(f"\n--- Training STL for: {task_name} ({'regression' if is_regression else str(task_config['num_classes']) + ' classes'}) ---")
        
        subjects = metadata['subject'].unique()
        subject_labels = []
        for subj in subjects:
            mask = metadata['subject'] == subj
            label = metadata.loc[mask, 'pathology'].iloc[0]
            subject_labels.append(label)
        subject_labels = np.array(subject_labels)
        
        if n_folds == 1:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            fold_splits = list(sss.split(subjects, subject_labels))
        else:
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            fold_splits = list(skf.split(subjects, subject_labels))
        fold_metrics = []
        
        for fold, (train_subj_idx, test_subj_idx) in enumerate(fold_splits):
            print(f"  Fold {fold + 1}/{n_folds}")
            
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
            
            train_idx = np.where(metadata['subject'].isin(train_subjects).values)[0]
            val_idx = np.where(metadata['subject'].isin(val_subjects).values)[0]
            test_idx = np.where(metadata['subject'].isin(test_subjects).values)[0]
            
            train_data, val_data, test_data = X[train_idx], X[val_idx], X[test_idx]
            if training_config.get('normalize_inputs', True):
                standardizer = FoldStandardizer()
                train_data = standardizer.fit_transform(train_data)
                val_data = standardizer.transform(val_data)
                test_data = standardizer.transform(test_data)

            train_dataset = GaitDataset(
                train_data, {task_name: y[task_name][train_idx]},
                metadata.iloc[train_idx].reset_index(drop=True)
            )
            val_dataset = GaitDataset(
                val_data, {task_name: y[task_name][val_idx]},
                metadata.iloc[val_idx].reset_index(drop=True)
            )
            test_dataset = GaitDataset(
                test_data, {task_name: y[task_name][test_idx]},
                metadata.iloc[test_idx].reset_index(drop=True)
            )
            batch_size = training_config.get('batch_size', 32)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
            
            # Create model
            model = SingleTaskModel(
                encoder_type=encoder_type,
                input_channels=input_channels,
                num_classes=task_config['num_classes'],
                task_type=task_config['type']
            ).to(device)
            
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config.get('lr', 0.001),
                weight_decay=training_config.get('weight_decay', 0.01)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=2, eta_min=1e-5
            )
            
            # Training loop
            best_val_loss = float('inf')
            patience = 0
            max_patience = training_config.get('early_stopping_patience', 15)
            best_state = None
            
            for epoch in range(1, training_config.get('max_epochs', 100) + 1):
                # Train
                model.train()
                for data, targets in train_loader:
                    data = data.to(device)
                    target = targets[task_name].to(device)
                    
                    # Mask out missing labels (-1)
                    mask = target >= 0
                    if not mask.any():
                        continue
                    
                    optimizer.zero_grad()
                    output = model(data)
                    if is_regression:
                        loss = F.smooth_l1_loss(output[mask].squeeze(-1), target[mask].float())
                    else:
                        loss = F.cross_entropy(output[mask], target[mask])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                scheduler.step()
                
                # Validate
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for data, targets in val_loader:
                        data = data.to(device)
                        target = targets[task_name].to(device)
                        output = model(data)
                        mask = target >= 0
                        if not mask.any():
                            continue
                        if is_regression:
                            val_loss += F.smooth_l1_loss(output[mask].squeeze(-1), target[mask].float()).item()
                        else:
                            val_loss += F.cross_entropy(output[mask], target[mask]).item()
                val_loss /= len(val_loader)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= max_patience:
                        break
            
            # Restore best and test
            if best_state:
                model.load_state_dict(best_state)
            
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for data, targets in test_loader:
                    data = data.to(device)
                    output = model(data)
                    if is_regression:
                        preds = output.squeeze(-1)
                    else:
                        preds = output.argmax(dim=-1)
                    all_preds.append(preds.cpu())
                    all_targets.append(targets[task_name])
            
            all_preds = torch.cat(all_preds).numpy()
            all_targets = torch.cat(all_targets).numpy()
            
            evaluator = Evaluator()
            if is_regression:
                # Only evaluate on valid regression targets (>= 0)
                valid_mask = all_targets >= 0
                metrics = evaluator.evaluate_regression(all_targets, all_preds, mask=valid_mask)
                fold_result = {
                    f'test_{task_name}_mae': metrics.get('mae', np.nan),
                    f'test_{task_name}_rmse': metrics.get('rmse', np.nan),
                    f'test_{task_name}_r2': metrics.get('r2', np.nan),
                }
            else:
                metrics = evaluator.evaluate_classification(all_targets, all_preds)
                fold_result = {
                    f'test_{task_name}_accuracy': metrics['accuracy'],
                    f'test_{task_name}_f1_macro': metrics['f1_macro'],
                    f'test_{task_name}_f1_weighted': metrics['f1_weighted'],
                }
            fold_metrics.append(fold_result)
            
            if verbose:
                if is_regression:
                    print(f"    MAE: {metrics.get('mae', float('nan')):.4f}, "
                          f"R2: {metrics.get('r2', float('nan')):.4f}")
                else:
                    print(f"    Acc: {metrics['accuracy']:.4f}, "
                          f"F1: {metrics['f1_macro']:.4f}")
        
        # Aggregate
        for key in fold_metrics[0]:
            values = [fm[key] for fm in fold_metrics]
            all_results[f'{key}_mean'] = np.mean(values)
            all_results[f'{key}_std'] = np.std(values)
    
    # Save
    if results_dir:
        save_dir = Path(results_dir) / 'single_task'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_results(all_results, str(save_dir / 'results.json'))
        print(f"\nSTL results saved to: {save_dir}")
    
    return all_results


# ============================================================================
# Main: Run All Experiments
# ============================================================================

def run_all_experiments(args):
    """Run all configured MTL experiments."""
    start_time = time.time()
    
    # Set seed
    set_seed(args.seed)
    
    # Load configs
    data_config = load_config(str(project_root / 'configs' / 'data_config.yaml'))
    model_config = load_config(str(project_root / 'configs' / 'model_config.yaml'))
    
    # Training config - start from YAML defaults, then force CLI overrides
    training_config = model_config.get('training', {})
    # Force-override ALL CLI parameters (setdefault won't override yaml values)
    training_config['batch_size'] = args.batch_size
    training_config['max_epochs'] = args.epochs
    training_config['early_stopping_patience'] = args.patience
    training_config['lr'] = args.lr
    training_config['weight_decay'] = args.weight_decay
    training_config['dropout'] = args.dropout
    training_config['warmup_epochs'] = args.warmup_epochs
    training_config['label_smoothing'] = args.label_smoothing
    training_config['aug_p'] = args.aug_p
    training_config['split_mode'] = args.split_mode
    training_config['reg_weight'] = args.reg_weight
    training_config['use_mse_loss'] = args.use_mse_loss
    training_config['monitor_metric'] = args.monitor_metric
    training_config['monitor_mode'] = args.monitor_mode
    training_config['use_amp'] = args.use_amp
    training_config['num_workers'] = args.num_workers
    training_config['normalize_inputs'] = data_config.get('data', {}).get(
        'preprocessing', {}
    ).get('normalize', True)
    if args.scheduler:
        training_config['scheduler'] = {'type': args.scheduler}
        if args.scheduler == 'CosineAnnealing':
            training_config['scheduler']['T_max'] = args.epochs
            training_config['scheduler']['eta_min'] = 1e-6
    # Model architecture params
    for k in ['d_model', 'num_layers', 'dim_feedforward', 'nhead', 'head_hidden_dim']:
        v = getattr(args, k, None)
        if v is not None:
            training_config[k] = v
    if args.use_mixup:
        training_config['use_mixup'] = True
    
    # Results directory
    if args.output_dir:
        results_dir = str(Path(args.output_dir).resolve())
    else:
        results_dir = str(project_root / 'results' / 'mtl_experiments')
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    X, y, metadata = load_data(
        data_config, data_path=args.data_path, verbose=True
    )
    
    print(f"\nDataset: X={X.shape}, "
          f"binary={np.bincount(y['binary'].astype(int))}, "
          f"coarse={np.bincount(y['coarse'].astype(int))}, "
          f"fine={np.bincount(y['fine'].astype(int))}, "
          f"tasks={list(y.keys())}")
    
    # Determine which experiments to run
    if args.experiment:
        experiments = {args.experiment: EXPERIMENT_CONFIGS[args.experiment]}
    elif args.ablation:
        # Run hierarchical MTL ablation
        experiments = {
            k: v for k, v in EXPERIMENT_CONFIGS.items()
            if k in ['hard_sharing', 'hierarchical_mtl']
        }
        # Add ablation variants
        experiments['hierarchical_no_hierarchy'] = {
            'model_type': 'hierarchical',
            'gradient_method': 'pcgrad',
            'use_hierarchy_loss': False,
            'use_focal_loss': True,
            'description': 'Ablation: No hierarchy constraint'
        }
        experiments['hierarchical_no_pcgrad'] = {
            'model_type': 'hierarchical',
            'gradient_method': 'none',
            'use_hierarchy_loss': True,
            'use_focal_loss': True,
            'description': 'Ablation: No gradient surgery'
        }
        experiments['hierarchical_no_focal'] = {
            'model_type': 'hierarchical',
            'gradient_method': 'pcgrad',
            'use_hierarchy_loss': True,
            'use_focal_loss': False,
            'description': 'Ablation: No focal loss'
        }
    else:
        experiments = EXPERIMENT_CONFIGS
    
    # Run experiments
    all_results = {}
    
    for exp_name, exp_config in experiments.items():
        try:
            if exp_name == 'single_task':
                result = run_stl_experiment(
                    X, y, metadata, training_config,
                    encoder_type=args.encoder,
                    n_folds=args.folds,
                    results_dir=results_dir,
                    verbose=args.verbose
                )
            else:
                result = run_single_experiment(
                    exp_name=exp_name,
                    exp_config=exp_config,
                    X=X, y=y, metadata=metadata,
                    training_config=training_config,
                    encoder_type=args.encoder,
                    n_folds=args.folds,
                    results_dir=results_dir,
                    verbose=args.verbose
                )
            
            all_results[exp_name] = result
            
        except Exception as e:
            print(f"\nERROR: Experiment '{exp_name}' failed: {e}")
            import traceback
            traceback.print_exc()
            all_results[exp_name] = {'error': str(e)}
    
    # Generate comparison table
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPARISON")
    print("=" * 60)
    
    comparison = generate_comparison_table(all_results)
    if comparison is not None:
        print(comparison.to_string(index=False))
        
        # Save comparison
        comparison.to_csv(
            Path(results_dir) / 'comparison_table.csv', index=False
        )
    
    # Save all results
    save_results(
        {k: str(v) if isinstance(v, Exception) else v 
         for k, v in all_results.items()},
        str(Path(results_dir) / 'all_results.json')
    )
    
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time / 60:.1f} minutes")
    print(f"Results saved to: {results_dir}")


def generate_comparison_table(all_results: dict) -> pd.DataFrame:
    """Generate a comparison table from all experiment results."""
    rows = []
    
    for exp_name, result in all_results.items():
        if 'error' in result:
            continue
        
        row = {'Method': exp_name}
        
        if 'summary' in result:
            summary = result['summary']
            for key, (mean, std) in summary.items():
                if 'accuracy' in key or 'f1_macro' in key:
                    row[key] = f"{mean:.4f}+/-{std:.4f}"
        elif isinstance(result, dict):
            for key, value in result.items():
                if '_mean' in key and ('accuracy' in key or 'f1_macro' in key):
                    std_key = key.replace('_mean', '_std')
                    std = result.get(std_key, 0)
                    display_key = key.replace('_mean', '')
                    row[display_key] = f"{value:.4f}+/-{std:.4f}"
        
        if len(row) > 1:
            rows.append(row)
    
    if not rows:
        return None
    
    return pd.DataFrame(rows)


# ============================================================================
# Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run Multi-Task Learning experiments'
    )
    parser.add_argument(
        '--experiment', type=str, default=None,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help='Run a specific experiment (default: all)'
    )
    parser.add_argument(
        '--encoder', type=str, default='transformer',
        choices=['cnn1d', 'lstm', 'transformer', 'cnn_lstm'],
        help='Encoder for architectures with a selectable encoder'
    )
    parser.add_argument(
        '--folds', type=int, default=10,
        help='Number of CV folds (default: 10)'
    )
    parser.add_argument(
        '--epochs', type=int, default=80,
        help='Maximum training epochs (default: 80)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--lr', type=float, default=0.0003,
        help='Learning rate (default: 0.0003)'
    )
    parser.add_argument(
        '--dropout', type=float, default=0.35,
        help='Dropout rate (default: 0.35)'
    )
    parser.add_argument(
        '--weight_decay', type=float, default=0.0005,
        help='Weight decay (default: 0.0005)'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=8,
        help='Learning rate warmup epochs (default: 8)'
    )
    parser.add_argument(
        '--use_mixup', action='store_true',
        help='Enable mixup data augmentation'
    )
    parser.add_argument(
        '--patience', type=int, default=20,
        help='Early stopping patience (default: 20)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--ablation', action='store_true',
        help='Run ablation study for hierarchical MTL'
    )
    parser.add_argument(
        '--verbose', action='store_true', default=True,
        help='Print detailed progress'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick test run (2 folds, 10 epochs)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Custom output directory (default: results/mtl_experiments)'
    )
    parser.add_argument(
        '--data_path', type=str, default=None,
        help='Data root (default: HMTL_DATA_PATH, then configs/data_config.yaml)'
    )
    # Model architecture tuning
    parser.add_argument('--d_model', type=int, default=None, help='Transformer d_model')
    parser.add_argument('--num_layers', type=int, default=None, help='Transformer num_layers')
    parser.add_argument('--dim_feedforward', type=int, default=None, help='Transformer FFN dim')
    parser.add_argument('--nhead', type=int, default=None, help='Transformer attention heads')
    parser.add_argument('--head_hidden_dim', type=int, default=64, help='Task head hidden dim (default: 64)')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing')
    parser.add_argument('--aug_p', type=float, default=0.5, help='Augmentation probability (0 to disable)')
    parser.add_argument('--split_mode', type=str, default='subject_wise', 
                        choices=['subject_wise', 'trial_wise'],
                        help='Data splitting mode')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealing',
                        choices=['CosineAnnealingWarmRestarts', 'CosineAnnealing', 'StepLR', 'ReduceLROnPlateau'],
                        help='LR scheduler type (default: CosineAnnealing)')
    parser.add_argument('--reg_weight', type=float, default=2.0,
                        help='Regression loss weight multiplier (default: 2.0)')
    parser.add_argument(
        '--use_mse_loss', action=argparse.BooleanOptionalAction, default=True,
        help='Use MSE regression loss (default: enabled)'
    )
    parser.add_argument('--monitor_metric', type=str, default='val_binary_accuracy',
                        help='Metric to monitor for early stopping (default: val_binary_accuracy)')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='Monitor mode: max for accuracy, min for loss (default: max)')
    parser.add_argument('--use_amp', action='store_true',
                        help='Enable AMP mixed precision training (bf16/fp16)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='DataLoader num_workers for parallel data loading (default: 0)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.folds = 2
        args.epochs = 10
        args.patience = 5
    
    return args


if __name__ == '__main__':
    args = parse_args()
    run_all_experiments(args)
