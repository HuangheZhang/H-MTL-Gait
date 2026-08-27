"""
Loss Functions for Multi-Task Gait Analysis
============================================
Task-specific and MTL-specific loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


# ============================================================================
# Focal Loss
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Down-weights well-classified samples and focuses on hard samples.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        """
        Args:
            gamma: Focusing parameter. Higher gamma means more focus on hard samples.
            alpha: Per-class weight tensor. Shape (num_classes,).
            reduction: 'none', 'mean', or 'sum'
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: (batch, num_classes) raw model outputs
            targets: (batch,) class indices
        """
        num_classes = logits.size(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            target_one_hot = F.one_hot(targets, num_classes).float()
            target_one_hot = (
                target_one_hot * (1 - self.label_smoothing) 
                + self.label_smoothing / num_classes
            )
            log_probs = F.log_softmax(logits, dim=1)
            ce_loss = -(target_one_hot * log_probs).sum(dim=1)
        else:
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute p_t
        probs = F.softmax(logits, dim=1)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha (class weights) if provided
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, targets)
            focal_weight = alpha_t * focal_weight
        
        # Final loss
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ============================================================================
# Class-Balanced Loss
# ============================================================================

class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss based on effective number of samples.
    
    Re-weights the loss for each class based on the effective number
    of samples, providing better handling of long-tailed distributions.
    
    Reference: Cui et al., "Class-Balanced Loss Based on Effective Number 
               of Samples", CVPR 2019
    """
    
    def __init__(
        self,
        samples_per_class: List[int],
        beta: float = 0.9999,
        loss_type: str = 'focal',
        gamma: float = 2.0
    ):
        """
        Args:
            samples_per_class: Number of samples per class
            beta: Hyperparameter for effective number. Default 0.9999.
            loss_type: 'softmax', 'focal', or 'sigmoid'
            gamma: Focal loss gamma (only used if loss_type='focal')
        """
        super().__init__()
        self.loss_type = loss_type
        self.gamma = gamma
        
        # Compute effective number of samples
        effective_num = 1.0 - np.power(beta, np.array(samples_per_class))
        weights = (1.0 - beta) / (effective_num + 1e-8)
        weights = weights / weights.sum() * len(samples_per_class)
        
        self.register_buffer('weights', torch.FloatTensor(weights))
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor
    ) -> torch.Tensor:
        if self.loss_type == 'focal':
            focal = FocalLoss(gamma=self.gamma, alpha=self.weights)
            return focal(logits, targets)
        elif self.loss_type == 'softmax':
            return F.cross_entropy(logits, targets, weight=self.weights)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


# ============================================================================
# Hierarchy Consistency Loss
# ============================================================================

class HierarchyConsistencyLoss(nn.Module):
    """
    Hierarchy Consistency Loss for hierarchical classification.
    
    Ensures predictions at finer granularity levels are consistent 
    with predictions at coarser levels through KL-divergence constraints.
    
    Hierarchy:
      Level 1 (Binary):  Healthy vs Pathological
      Level 2 (Coarse):  Healthy vs Neuro vs Ortho
      Level 3 (Fine):    HS, CVA, PD, CIPN, RIL, KOA, HOA, ACL
    """
    
    # Label hierarchy mappings
    FINE_TO_COARSE = {
        0: 0,  # HS -> Healthy
        1: 1, 2: 1, 3: 1, 4: 1,  # CVA/PD/CIPN/RIL -> Neuro
        5: 2, 6: 2, 7: 2  # KOA/HOA/ACL -> Ortho
    }
    
    COARSE_TO_BINARY = {
        0: 0,  # Healthy -> Healthy
        1: 1, 2: 1  # Neuro/Ortho -> Pathological
    }
    
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight
        
        # Create mapping matrices
        fine_to_coarse = torch.zeros(8, 3)
        for fine, coarse in self.FINE_TO_COARSE.items():
            fine_to_coarse[fine, coarse] = 1
        self.register_buffer('fine_to_coarse_mat', fine_to_coarse)
        
        coarse_to_binary = torch.zeros(3, 2)
        for coarse, binary in self.COARSE_TO_BINARY.items():
            coarse_to_binary[coarse, binary] = 1
        self.register_buffer('coarse_to_binary_mat', coarse_to_binary)
    
    def forward(
        self,
        binary_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bidirectional hierarchy consistency loss.
        
        1. Fine -> Coarse: marginalized fine probs should match coarse probs
        2. Coarse -> Binary: marginalized coarse probs should match binary probs
        """
        fine_probs = F.softmax(fine_logits, dim=-1)
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        binary_probs = F.softmax(binary_logits, dim=-1)
        
        # Marginalize fine probs to coarse
        fine_to_coarse = torch.matmul(fine_probs, self.fine_to_coarse_mat)
        
        # Marginalize coarse probs to binary
        coarse_to_binary = torch.matmul(coarse_probs, self.coarse_to_binary_mat)
        
        # KL divergences (symmetrized)
        loss_fine_coarse = self._symmetric_kl(fine_to_coarse, coarse_probs)
        loss_coarse_binary = self._symmetric_kl(coarse_to_binary, binary_probs)
        
        return self.weight * (loss_fine_coarse + loss_coarse_binary)
    
    def _symmetric_kl(self, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Symmetrized KL divergence."""
        eps = 1e-10
        kl_pq = F.kl_div(
            torch.log(p + eps), q, reduction='batchmean'
        )
        kl_qp = F.kl_div(
            torch.log(q + eps), p, reduction='batchmean'
        )
        return 0.5 * (kl_pq + kl_qp)


# ============================================================================
# Mixup Loss
# ============================================================================

class MixupLoss(nn.Module):
    """
    Loss function compatible with Mixup data augmentation.
    Computes weighted combination of losses for mixed samples.
    """
    
    def __init__(self, criterion: nn.Module = None):
        super().__init__()
        self.criterion = criterion or nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        return lam * self.criterion(logits, targets_a) + (1 - lam) * self.criterion(logits, targets_b)


# ============================================================================
# Multi-Task Loss Combiner
# ============================================================================

# Task type registry for the 10-task setup
TASK_TYPE_REGISTRY = {
    'binary':         'classification',
    'coarse':         'classification',
    'fine':           'classification',
    'vga_class':      'classification',
    'gender':         'classification',
    'neuro_fine':     'classification',
    'regression':     'regression',
    'vga_regression': 'regression',
    'age':            'regression',
    'tug':            'regression',
}

class MTLLoss(nn.Module):
    """
    Data-driven Multi-Task Loss that handles arbitrary tasks.
    
    Automatically creates appropriate loss functions based on task type registry.
    Supports per-task masking for missing labels.
    """
    
    def __init__(
        self,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        use_hierarchy_loss: bool = True,
        hierarchy_weight: float = 0.1,
        samples_per_class: Optional[List[int]] = None,
        label_smoothing: float = 0.0,
        task_configs: Optional[Dict[str, Dict]] = None,
        regression_weight: float = 1.0,
        use_mse_loss: bool = False
    ):
        super().__init__()
        
        self.task_losses = nn.ModuleDict()
        self.task_types = {}
        self.regression_weight = regression_weight
        self.use_mse_loss = use_mse_loss
        
        # Build loss functions for each known task
        for task_name, task_type in TASK_TYPE_REGISTRY.items():
            self.task_types[task_name] = task_type
            if task_type == 'classification':
                if task_name == 'fine' and use_focal_loss:
                    if samples_per_class is not None:
                        self.task_losses[task_name] = ClassBalancedLoss(
                            samples_per_class, loss_type='focal', gamma=focal_gamma
                        )
                    else:
                        self.task_losses[task_name] = FocalLoss(gamma=focal_gamma)
                else:
                    self.task_losses[task_name] = nn.CrossEntropyLoss(
                        label_smoothing=label_smoothing
                    )
            else:  # regression
                if self.use_mse_loss:
                    self.task_losses[task_name] = nn.MSELoss()
                else:
                    self.task_losses[task_name] = nn.SmoothL1Loss()
        
        # Hierarchy consistency loss
        self.use_hierarchy_loss = use_hierarchy_loss
        if use_hierarchy_loss:
            self.hierarchy_loss = HierarchyConsistencyLoss(weight=hierarchy_weight)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        regression_mask: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Compute all task losses with automatic masking for missing labels.
        """
        losses = {}
        log_dict = {}
        device = next(iter(outputs.values())).device
        
        for task_name, output in outputs.items():
            if task_name not in targets or task_name not in self.task_losses:
                continue
            
            target = targets[task_name]
            task_type = self.task_types.get(task_name, 'classification')
            loss_fn = self.task_losses[task_name]
            
            if task_type == 'regression':
                # For regression tasks, mask out missing values (labeled as -1)
                mask = target >= 0
                if mask.any():
                    reg_out = output[mask].squeeze(-1)
                    reg_target = target[mask]
                    losses[task_name] = loss_fn(reg_out, reg_target) * self.regression_weight
                else:
                    losses[task_name] = torch.tensor(0.0, device=device)
            else:
                # For classification tasks, mask out missing values (labeled as -1)
                mask = target >= 0
                if mask.any():
                    cls_out = output[mask]
                    cls_target = target[mask]
                    losses[task_name] = loss_fn(cls_out, cls_target)
                else:
                    losses[task_name] = torch.tensor(0.0, device=device)
            
            log_dict[f'{task_name}_loss'] = losses[task_name].item()
        
        # Hierarchy consistency loss
        if (self.use_hierarchy_loss and 
            all(k in outputs for k in ['binary', 'coarse', 'fine'])):
            h_loss = self.hierarchy_loss(
                outputs['binary'], outputs['coarse'], outputs['fine']
            )
            losses['hierarchy'] = h_loss
            log_dict['hierarchy_loss'] = h_loss.item()
        
        # Total for logging
        log_dict['total_loss'] = sum(l.item() for l in losses.values())
        
        return losses, log_dict


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing loss functions...")
    
    batch_size = 16
    
    # Test Focal Loss
    print("\n  FocalLoss:")
    fl = FocalLoss(gamma=2.0)
    logits = torch.randn(batch_size, 8)
    targets = torch.randint(0, 8, (batch_size,))
    loss = fl(logits, targets)
    print(f"    Loss: {loss.item():.4f} OK")
    
    # Test Class-Balanced Loss
    print("\n  ClassBalancedLoss:")
    samples = [7504, 2000, 3000, 500, 800, 3000, 1500, 1200]
    cbl = ClassBalancedLoss(samples, loss_type='focal')
    loss = cbl(logits, targets)
    print(f"    Loss: {loss.item():.4f} OK")
    
    # Test Hierarchy Consistency Loss
    print("\n  HierarchyConsistencyLoss:")
    hcl = HierarchyConsistencyLoss(weight=0.1)
    binary_logits = torch.randn(batch_size, 2)
    coarse_logits = torch.randn(batch_size, 3)
    fine_logits = torch.randn(batch_size, 8)
    loss = hcl(binary_logits, coarse_logits, fine_logits)
    print(f"    Loss: {loss.item():.4f} OK")
    
    # Test MTLLoss
    print("\n  MTLLoss:")
    mtl_loss = MTLLoss(use_focal_loss=True, use_hierarchy_loss=True)
    outputs = {
        'binary': torch.randn(batch_size, 2),
        'coarse': torch.randn(batch_size, 3),
        'fine': torch.randn(batch_size, 8),
        'regression': torch.randn(batch_size, 1),
        'vga_class': torch.randn(batch_size, 5),
        'vga_regression': torch.randn(batch_size, 1),
        'gender': torch.randn(batch_size, 2),
        'age': torch.randn(batch_size, 1),
        'tug': torch.randn(batch_size, 1),
        'neuro_fine': torch.randn(batch_size, 4),
    }
    targets = {
        'binary': torch.randint(0, 2, (batch_size,)),
        'coarse': torch.randint(0, 3, (batch_size,)),
        'fine': torch.randint(0, 8, (batch_size,)),
        'regression': torch.rand(batch_size),
        'vga_class': torch.randint(0, 5, (batch_size,)),
        'vga_regression': torch.rand(batch_size),
        'gender': torch.randint(0, 2, (batch_size,)),
        'age': torch.rand(batch_size),
        'tug': torch.rand(batch_size),
        'neuro_fine': torch.randint(0, 4, (batch_size,)),
    }
    reg_mask = torch.ones(batch_size, dtype=torch.bool)
    losses, log_dict = mtl_loss(outputs, targets, reg_mask)
    print(f"    Losses: {list(losses.keys())}")
    print(f"    Total: {log_dict['total_loss']:.4f} OK")
    
    print("\nAll loss function tests passed!")
