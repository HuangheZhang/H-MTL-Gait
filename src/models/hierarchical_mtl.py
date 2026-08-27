"""
Hierarchical Multi-Task Learning Model
======================================
Core contribution: Hierarchical MTL with label and feature constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from .deep_models import (
    CNN1DEncoder, LSTMEncoder, TransformerEncoder, CNN_LSTMEncoder,
    TaskHead
)


class HierarchyConstraint(nn.Module):
    """
    Enforces hierarchical consistency between task predictions.
    
    Hierarchy:
    Level 1: Binary (Healthy vs Pathological)
    Level 2: Coarse (Healthy vs Neuro vs Ortho)
    Level 3: Fine (8 specific diseases)
    """
    
    # Label hierarchy mapping
    FINE_TO_COARSE = {
        0: 0,  # HS -> Healthy
        1: 1, 2: 1, 3: 1, 4: 1,  # CVA, PD, CIPN, RIL -> Neuro
        5: 2, 6: 2, 7: 2  # KOA, HOA, ACL -> Ortho
    }
    
    COARSE_TO_BINARY = {
        0: 0,  # Healthy -> Healthy
        1: 1, 2: 1  # Neuro, Ortho -> Pathological
    }
    
    def __init__(self):
        super().__init__()
        
        # Create mapping tensors
        fine_to_coarse = torch.zeros(8, 3)
        for fine, coarse in self.FINE_TO_COARSE.items():
            fine_to_coarse[fine, coarse] = 1
        self.register_buffer('fine_to_coarse', fine_to_coarse)
        
        coarse_to_binary = torch.zeros(3, 2)
        for coarse, binary in self.COARSE_TO_BINARY.items():
            coarse_to_binary[coarse, binary] = 1
        self.register_buffer('coarse_to_binary', coarse_to_binary)
        
    def get_consistency_loss(
        self,
        binary_logits: torch.Tensor,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hierarchical consistency loss.
        
        The idea: predictions at finer levels should be consistent with
        predictions at coarser levels.
        """
        # Marginalize fine predictions to coarse
        fine_probs = F.softmax(fine_logits, dim=-1)
        expected_coarse = torch.matmul(fine_probs, self.fine_to_coarse)
        
        # Marginalize coarse predictions to binary
        coarse_probs = F.softmax(coarse_logits, dim=-1)
        expected_binary = torch.matmul(coarse_probs, self.coarse_to_binary)
        
        # KL divergence losses
        coarse_target = F.softmax(coarse_logits, dim=-1)
        binary_target = F.softmax(binary_logits, dim=-1)
        
        # Fine -> Coarse consistency
        fine_coarse_loss = F.kl_div(
            torch.log(expected_coarse + 1e-10),
            coarse_target,
            reduction='batchmean'
        )
        
        # Coarse -> Binary consistency
        coarse_binary_loss = F.kl_div(
            torch.log(expected_binary + 1e-10),
            binary_target,
            reduction='batchmean'
        )
        
        return fine_coarse_loss + coarse_binary_loss


class UncertaintyWeighting(nn.Module):
    """
    Automatic task weighting via homoscedastic uncertainty.
    
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to 
               Weigh Losses for Scene Geometry and Semantics", CVPR 2018
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        # Log variance for each task (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted loss with uncertainty.
        
        For classification: L_i / (2 * sigma_i^2) + log(sigma_i)
        For regression: L_i / (2 * sigma_i^2) + log(sigma_i)
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
            
        return total_loss
    
    def get_weights(self) -> torch.Tensor:
        """Get current task weights (inverse variance)."""
        return torch.exp(-self.log_vars)


class HierarchicalMTLModel(nn.Module):
    """
    Hierarchical Multi-Task Learning Model.
    
    Key features:
    1. Shared encoder for all tasks
    2. Task-specific heads
    3. Hierarchical consistency constraint
    4. Uncertainty-based task weighting
    """
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_channels: int = 36,
        use_hierarchy_constraint: bool = True,
        use_uncertainty_weighting: bool = True,
        hierarchy_loss_weight: float = 0.1,
        **encoder_kwargs
    ):
        super().__init__()
        
        self.use_hierarchy_constraint = use_hierarchy_constraint
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.hierarchy_loss_weight = hierarchy_loss_weight
        
        # Shared encoder
        if encoder_type == 'cnn1d':
            self.encoder = CNN1DEncoder(input_channels, **encoder_kwargs)
        elif encoder_type == 'lstm':
            self.encoder = LSTMEncoder(input_channels, **encoder_kwargs)
        elif encoder_type == 'transformer':
            self.encoder = TransformerEncoder(input_channels, **encoder_kwargs)
        elif encoder_type == 'cnn_lstm':
            self.encoder = CNN_LSTMEncoder(input_channels, **encoder_kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
            
        hidden_dim = self.encoder.output_channels
        
        # Task-specific heads
        self.binary_head = TaskHead(hidden_dim, output_dim=2, task_type='classification')
        self.coarse_head = TaskHead(hidden_dim, output_dim=3, task_type='classification')
        self.fine_head = TaskHead(hidden_dim, output_dim=8, task_type='classification')
        self.regression_head = TaskHead(hidden_dim, output_dim=1, task_type='regression')
        
        # Hierarchy constraint
        if use_hierarchy_constraint:
            self.hierarchy_constraint = HierarchyConstraint()
            
        # Uncertainty weighting (4 tasks)
        if use_uncertainty_weighting:
            self.uncertainty_weighting = UncertaintyWeighting(4)
            
    def forward(self, x) -> Dict[str, torch.Tensor]:
        """Forward pass returning all task outputs."""
        features = self.encoder(x)
        
        return {
            'binary': self.binary_head(features),
            'coarse': self.coarse_head(features),
            'fine': self.fine_head(features),
            'regression': self.regression_head(features)
        }
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        regression_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss with hierarchy constraint and uncertainty weighting.
        
        Args:
            outputs: Model outputs for each task
            targets: Ground truth labels for each task
            regression_mask: Mask for valid regression targets (some may be missing)
            
        Returns:
            total_loss: Combined loss
            loss_dict: Individual losses for logging
        """
        # Individual task losses
        binary_loss = F.cross_entropy(outputs['binary'], targets['binary'])
        coarse_loss = F.cross_entropy(outputs['coarse'], targets['coarse'])
        fine_loss = F.cross_entropy(outputs['fine'], targets['fine'])
        
        # Regression loss (only for valid samples)
        if regression_mask is not None and regression_mask.any():
            valid_outputs = outputs['regression'][regression_mask]
            valid_targets = targets['regression'][regression_mask]
            regression_loss = F.mse_loss(valid_outputs.squeeze(), valid_targets)
        else:
            regression_loss = torch.tensor(0.0, device=outputs['binary'].device)
            
        # Hierarchy consistency loss
        if self.use_hierarchy_constraint:
            hierarchy_loss = self.hierarchy_constraint.get_consistency_loss(
                outputs['binary'], outputs['coarse'], outputs['fine']
            )
        else:
            hierarchy_loss = torch.tensor(0.0, device=outputs['binary'].device)
            
        # Combine losses
        task_losses = [binary_loss, coarse_loss, fine_loss, regression_loss]
        
        if self.use_uncertainty_weighting:
            main_loss = self.uncertainty_weighting(task_losses)
        else:
            # Simple sum with fixed weights
            main_loss = binary_loss + coarse_loss + fine_loss + 0.1 * regression_loss
            
        # Add hierarchy loss
        total_loss = main_loss + self.hierarchy_loss_weight * hierarchy_loss
        
        # Loss dictionary for logging
        loss_dict = {
            'binary_loss': binary_loss.item(),
            'coarse_loss': coarse_loss.item(),
            'fine_loss': fine_loss.item(),
            'regression_loss': regression_loss.item(),
            'hierarchy_loss': hierarchy_loss.item(),
            'total_loss': total_loss.item()
        }
        
        if self.use_uncertainty_weighting:
            weights = self.uncertainty_weighting.get_weights()
            for i, name in enumerate(['binary', 'coarse', 'fine', 'regression']):
                loss_dict[f'{name}_weight'] = weights[i].item()
                
        return total_loss, loss_dict
    
    def get_features(self, x) -> torch.Tensor:
        """Get shared features for analysis."""
        return self.encoder(x)


class GradientSurgery:
    """
    Gradient manipulation methods for MTL.
    
    Implements:
    - PCGrad: Projecting Conflicting Gradients
    - CAGrad: Conflict-Averse Gradient descent
    """
    
    @staticmethod
    def pcgrad(grads: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        PCGrad: Project conflicting gradients.
        
        Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
        """
        num_tasks = len(grads)
        projected_grads = [g.clone() for g in grads]
        
        for i in range(num_tasks):
            for j in range(num_tasks):
                if i != j:
                    # Project grad_i onto grad_j if they conflict
                    dot = torch.dot(projected_grads[i].flatten(), grads[j].flatten())
                    if dot < 0:
                        projected_grads[i] -= (dot / (torch.norm(grads[j])**2 + 1e-10)) * grads[j]
                        
        return projected_grads
    
    @staticmethod
    def cagrad(grads: List[torch.Tensor], c: float = 0.4) -> torch.Tensor:
        """
        CAGrad: Conflict-Averse Gradient descent.
        
        Reference: Liu et al., "Conflict-Averse Gradient Descent for MTL", NeurIPS 2021
        """
        # Stack gradients
        G = torch.stack([g.flatten() for g in grads])
        
        # Average gradient
        g_avg = G.mean(dim=0)
        
        # Compute gradient direction that minimizes worst-case loss
        # Simplified version: weighted average based on gradient magnitudes
        norms = torch.norm(G, dim=1)
        weights = F.softmax(norms * c, dim=0)
        
        g_cagrad = (weights.unsqueeze(1) * G).sum(dim=0)
        
        return g_cagrad


if __name__ == "__main__":
    # Test Hierarchical MTL Model
    print("Testing Hierarchical MTL Model...")
    
    batch_size = 8
    seq_len = 200
    channels = 36
    
    model = HierarchicalMTLModel(
        encoder_type='transformer',
        input_channels=channels,
        use_hierarchy_constraint=True,
        use_uncertainty_weighting=True
    )
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, channels)
    outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    for task, out in outputs.items():
        print(f"  {task}: {out.shape}")
        
    # Test loss computation
    targets = {
        'binary': torch.randint(0, 2, (batch_size,)),
        'coarse': torch.randint(0, 3, (batch_size,)),
        'fine': torch.randint(0, 8, (batch_size,)),
        'regression': torch.rand(batch_size)
    }
    
    total_loss, loss_dict = model.compute_loss(outputs, targets)
    print(f"\nLoss: {total_loss.item():.4f}")
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
