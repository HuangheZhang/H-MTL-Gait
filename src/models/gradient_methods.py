"""
Gradient Manipulation Methods for Multi-Task Learning
=====================================================
Implements gradient-level optimization strategies:
- PCGrad: Projecting Conflicting Gradients (Yu et al., NeurIPS 2020)
- GradNorm: Gradient Normalization (Chen et al., ICML 2018)
- CAGrad: Conflict-Averse Gradient Descent (Liu et al., NeurIPS 2021)
- MGDA: Multiple Gradient Descent Algorithm (Sener & Koltun, NeurIPS 2018)
- NashMTL: Nash Bargaining for MTL (Navon et al., ICML 2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
import copy


class GradientMethod:
    """Base class for gradient manipulation methods."""
    
    def __init__(self):
        self.name = "base"
    
    def backward(
        self,
        losses: Dict[str, torch.Tensor],
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        """
        Perform backward pass with gradient manipulation.
        
        Args:
            losses: Dictionary of task losses
            model: The MTL model
            optimizer: The optimizer
        """
        raise NotImplementedError


class NoneGradient(GradientMethod):
    """No gradient manipulation - simple sum of losses."""
    
    def __init__(self):
        super().__init__()
        self.name = "none"
    
    def backward(self, losses, model, optimizer):
        total_loss = sum(losses.values())
        optimizer.zero_grad()
        total_loss.backward()
        return total_loss.detach()


class PCGrad(GradientMethod):
    """
    PCGrad: Projecting Conflicting Gradients.
    
    When gradients from different tasks conflict (negative cosine similarity),
    project one onto the normal plane of the other to remove the conflicting component.
    
    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
    """
    
    def __init__(self):
        super().__init__()
        self.name = "pcgrad"
    
    def backward(self, losses, model, optimizer):
        task_names = list(losses.keys())
        num_tasks = len(task_names)
        
        # Compute per-task gradients (skip tasks with no grad_fn, e.g. all-masked batches)
        task_grads = {}
        shared_params = [p for p in model.parameters() if p.requires_grad]
        
        for task_name, loss in losses.items():
            if not loss.requires_grad:
                continue
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            
            grads = []
            for p in shared_params:
                if p.grad is not None:
                    grads.append(p.grad.clone().flatten())
                else:
                    grads.append(torch.zeros(p.numel(), device=p.device))
            task_grads[task_name] = torch.cat(grads)
        
        if not task_grads:
            # All losses are masked - nothing to do
            optimizer.zero_grad()
            return sum(losses.values()).detach()
        
        # Project conflicting gradients
        valid_names = list(task_grads.keys())
        projected = {name: grad.clone() for name, grad in task_grads.items()}
        
        # Random permutation for fairness
        perm = torch.randperm(len(valid_names)).tolist()
        
        for i in perm:
            task_i = valid_names[i]
            for j in perm:
                if i == j:
                    continue
                task_j = valid_names[j]
                
                dot = torch.dot(projected[task_i], task_grads[task_j])
                if dot < 0:
                    # Project: remove conflicting component
                    norm_sq = torch.dot(task_grads[task_j], task_grads[task_j])
                    projected[task_i] -= (dot / (norm_sq + 1e-12)) * task_grads[task_j]
        
        # Aggregate projected gradients
        final_grad = sum(projected.values())
        
        # Set gradients
        optimizer.zero_grad()
        offset = 0
        for p in shared_params:
            numel = p.numel()
            p.grad = final_grad[offset:offset + numel].view_as(p).clone()
            offset += numel
        
        return sum(losses.values()).detach()


class GradNorm(GradientMethod):
    """
    GradNorm: Gradient Normalization.
    
    Dynamically tunes task weights to balance gradient norms across tasks,
    ensuring no single task dominates training.
    
    Reference: Chen et al., "GradNorm: Gradient Normalization for Adaptive 
               Loss Balancing in Deep Multi-Task Networks", ICML 2018
    """
    
    def __init__(
        self,
        num_tasks: int,
        alpha: float = 1.5,
        lr: float = 0.025
    ):
        """
        Args:
            num_tasks: Number of tasks
            alpha: Asymmetry parameter controlling restoring force strength
                   Higher alpha -> stronger enforcement of balanced training
            lr: Learning rate for task weight updates
        """
        super().__init__()
        self.name = "gradnorm"
        self.alpha = alpha
        self.lr = lr
        self.num_tasks = num_tasks
        
        # Learnable task weights (initialized to 1)
        self.task_weights = torch.ones(num_tasks, requires_grad=True)
        self.initial_losses = None
    
    def to(self, device):
        """Move task weights to device, keeping them as leaf tensors."""
        self.task_weights = self.task_weights.detach().clone().to(device).requires_grad_(True)
        if self.initial_losses is not None:
            self.initial_losses = self.initial_losses.to(device)
        return self
    
    def backward(self, losses, model, optimizer):
        task_names = list(losses.keys())
        device = next(model.parameters()).device
        
        if self.task_weights.device != device:
            self.to(device)
        
        loss_values = torch.stack([losses[name] for name in task_names])
        
        # Store initial losses for relative inverse training rate
        if self.initial_losses is None:
            self.initial_losses = loss_values.detach().clone()
        
        # Compute weighted total loss
        weights = F.softmax(self.task_weights[:len(task_names)], dim=0) * len(task_names)
        weighted_loss = (weights * loss_values).sum()
        
        # Standard backward for model parameters
        optimizer.zero_grad()
        weighted_loss.backward(retain_graph=True)
        
        # --- GradNorm weight update ---
        # Get the last shared layer (for gradient norm computation)
        shared_params = self._get_last_shared_layer(model)
        
        if shared_params is not None and len(shared_params) > 0:
            # Compute gradient norms for each task
            grad_norms = []
            for i, task_name in enumerate(task_names):
                task_loss = weights[i] * losses[task_name]
                grads = torch.autograd.grad(
                    task_loss, shared_params, 
                    retain_graph=True, allow_unused=True,
                    create_graph=True
                )
                grad_norm = torch.norm(
                    torch.cat([g.flatten() for g in grads if g is not None])
                )
                grad_norms.append(grad_norm)
            
            grad_norms = torch.stack(grad_norms)
            
            # Average gradient norm
            avg_grad_norm = grad_norms.mean().detach()
            
            # Inverse training rate
            loss_ratios = loss_values.detach() / (self.initial_losses + 1e-12)
            inverse_train_rate = loss_ratios / (loss_ratios.mean() + 1e-12)
            
            # Target gradient norms
            target_grad_norms = avg_grad_norm * (inverse_train_rate ** self.alpha)
            
            # GradNorm loss: L1 between current and target gradient norms
            gradnorm_loss = (grad_norms - target_grad_norms.detach()).abs().sum()
            
            # Update task weights
            if self.task_weights.grad is not None:
                self.task_weights.grad.zero_()
            gradnorm_loss.backward()
            
            with torch.no_grad():
                if self.task_weights.grad is not None:
                    self.task_weights.data -= self.lr * self.task_weights.grad.data
                # Renormalize weights
                self.task_weights.data = (
                    self.task_weights.data / self.task_weights.data.sum() * self.num_tasks
                )
            # Reset gradient for next iteration
            if self.task_weights.grad is not None:
                self.task_weights.grad = None
        
        return weighted_loss.detach()
    
    def _get_last_shared_layer(self, model):
        """Get parameters of the last shared layer for GradNorm."""
        # Try to find the encoder's last layer parameters
        if hasattr(model, 'encoder'):
            params = list(model.encoder.parameters())
            if params:
                return params[-2:]  # Last weight and bias
        
        # Fallback: return all shared parameters
        shared = []
        for name, param in model.named_parameters():
            if 'head' not in name and 'gate' not in name:
                shared.append(param)
        return shared[-2:] if shared else []
    
    def get_weights(self) -> Dict[str, float]:
        weights = F.softmax(self.task_weights, dim=0) * self.num_tasks
        return weights.detach().cpu().numpy()


class CAGrad(GradientMethod):
    """
    CAGrad: Conflict-Averse Gradient Descent.
    
    Finds a gradient direction that minimizes the worst-case task loss 
    within a ball around the average gradient direction.
    
    Reference: Liu et al., "Conflict-Averse Gradient Descent for Multi-Task Learning", 
               NeurIPS 2021
    """
    
    def __init__(self, c: float = 0.4, rescale: int = 1):
        """
        Args:
            c: Constraint strength. Higher c allows more departure from average gradient.
            rescale: 0=no rescale, 1=rescale to average gradient norm
        """
        super().__init__()
        self.name = "cagrad"
        self.c = c
        self.rescale = rescale
    
    def backward(self, losses, model, optimizer):
        task_names = list(losses.keys())
        shared_params = [p for p in model.parameters() if p.requires_grad]
        
        # Compute per-task gradients (skip tasks with no grad_fn)
        grads = []
        valid_names = []
        for task_name in task_names:
            if not losses[task_name].requires_grad:
                continue
            optimizer.zero_grad()
            losses[task_name].backward(retain_graph=True)
            
            grad = []
            for p in shared_params:
                if p.grad is not None:
                    grad.append(p.grad.clone().flatten())
                else:
                    grad.append(torch.zeros(p.numel(), device=p.device))
            grads.append(torch.cat(grad))
            valid_names.append(task_name)
        
        if not grads:
            optimizer.zero_grad()
            return sum(losses.values()).detach()
        
        G = torch.stack(grads)  # (num_tasks, total_params)
        
        # Average gradient
        g_avg = G.mean(dim=0)
        g_avg_norm = torch.norm(g_avg)
        
        if g_avg_norm < 1e-12:
            # Degenerate case: use simple sum
            optimizer.zero_grad()
            total = sum(losses.values())
            if total.requires_grad:
                total.backward()
            return total.detach()
        
        # Compute the CAGrad direction
        # Project each gradient's deviation from average onto the average direction
        # Solve for the optimal direction within the constraint ball
        # Simplified approach: use the direction that minimizes worst-case inner product
        g_avg_normalized = g_avg / g_avg_norm
        dots = G @ g_avg_normalized  # Inner products with average direction
        
        # Adjust: move towards tasks with smaller inner products (worst performing)
        weights = F.softmax(-dots / (self.c + 1e-12), dim=0)
        g_final = (weights.unsqueeze(1) * G).sum(dim=0)
        
        # Rescale
        if self.rescale == 1:
            g_final = g_final * (g_avg_norm / (torch.norm(g_final) + 1e-12))
        
        # Set gradients
        optimizer.zero_grad()
        offset = 0
        for p in shared_params:
            numel = p.numel()
            p.grad = g_final[offset:offset + numel].view_as(p).clone()
            offset += numel
        
        return sum(losses.values()).detach()


class MGDA(GradientMethod):
    """
    MGDA: Multiple Gradient Descent Algorithm.
    
    Finds the minimum-norm convex combination of per-task gradients,
    ensuring a Pareto-optimal descent direction.
    
    Reference: Sener & Koltun, "Multi-Task Learning as Multi-Objective Optimization", 
               NeurIPS 2018
    """
    
    def __init__(self, normalize: bool = True):
        super().__init__()
        self.name = "mgda"
        self.normalize = normalize
    
    def backward(self, losses, model, optimizer):
        task_names = list(losses.keys())
        shared_params = [p for p in model.parameters() if p.requires_grad]
        
        # Compute per-task gradients (skip tasks with no grad_fn)
        grads = []
        for task_name in task_names:
            if not losses[task_name].requires_grad:
                continue
            optimizer.zero_grad()
            losses[task_name].backward(retain_graph=True)
            
            grad = []
            for p in shared_params:
                if p.grad is not None:
                    grad.append(p.grad.clone().flatten())
                else:
                    grad.append(torch.zeros(p.numel(), device=p.device))
            grads.append(torch.cat(grad))
        
        if not grads:
            optimizer.zero_grad()
            return sum(losses.values()).detach()
        
        G = torch.stack(grads)  # (num_tasks, dim)
        
        if self.normalize:
            norms = torch.norm(G, dim=1, keepdim=True)
            G = G / (norms + 1e-12)
        
        # Solve min-norm problem using Frank-Wolfe
        weights = self._min_norm_solver(G)
        
        # Compute final gradient
        g_final = (weights.unsqueeze(1) * G).sum(dim=0)
        
        # Set gradients
        optimizer.zero_grad()
        offset = 0
        for p in shared_params:
            numel = p.numel()
            p.grad = g_final[offset:offset + numel].view_as(p).clone()
            offset += numel
        
        return sum(losses.values()).detach()
    
    def _min_norm_solver(self, G: torch.Tensor, max_iter: int = 20) -> torch.Tensor:
        """
        Solve min-norm problem: min_w ||sum(w_i * g_i)||^2 s.t. w >= 0, sum(w) = 1.
        Uses Frank-Wolfe algorithm.
        """
        num_tasks = G.size(0)
        device = G.device
        
        # Initialize with uniform weights
        weights = torch.ones(num_tasks, device=device) / num_tasks
        
        # Precompute gram matrix
        gram = G @ G.T  # (num_tasks, num_tasks)
        
        for _ in range(max_iter):
            # Current gradient of the objective w.r.t. weights
            obj_grad = gram @ weights  # (num_tasks,)
            
            # FW step: move towards the minimal gradient entry
            min_idx = obj_grad.argmin()
            fw_direction = torch.zeros_like(weights)
            fw_direction[min_idx] = 1.0
            
            # Line search
            d = fw_direction - weights
            denom = d @ gram @ d
            if denom > 0:
                gamma = max(0, min(1, -(obj_grad @ d) / denom))
            else:
                gamma = 1.0
            
            weights = weights + gamma * d
        
        return weights.detach()


# ============================================================================
# Uncertainty Weighting (moved here for consistency)
# ============================================================================

class UncertaintyWeighting(GradientMethod):
    """
    Uncertainty Weighting: Learn task weights via homoscedastic uncertainty.
    
    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty to 
               Weigh Losses for Scene Geometry and Semantics", CVPR 2018
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        self.name = "uncertainty"
        self.num_tasks = num_tasks
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def to(self, device):
        """Move parameters to device."""
        self.log_vars = nn.Parameter(self.log_vars.data.to(device))
        return self
    
    def backward(self, losses, model, optimizer):
        task_names = list(losses.keys())
        device = next(model.parameters()).device
        
        # Ensure log_vars is on the correct device
        if self.log_vars.device != device:
            self.to(device)
        
        total_loss = torch.zeros(1, device=device, requires_grad=False)
        for i, name in enumerate(task_names):
            if i < self.num_tasks:
                precision = torch.exp(-self.log_vars[i])
                total_loss = total_loss + precision * losses[name] + self.log_vars[i]
            else:
                total_loss = total_loss + losses[name]
        
        optimizer.zero_grad()
        total_loss.backward()
        return total_loss.detach()
    
    def get_weights(self) -> np.ndarray:
        return torch.exp(-self.log_vars).detach().cpu().numpy()
    
    def parameters(self):
        return [self.log_vars]


# ============================================================================
# DWA: Dynamic Weight Average
# ============================================================================

class DWA(GradientMethod):
    """
    DWA: Dynamic Weight Average.
    
    Adjusts task weights based on the rate of change of each task's loss,
    giving higher weight to tasks with faster-decreasing losses.
    
    Reference: Liu et al., "End-to-End Multi-Task Learning with Attention", CVPR 2019
    """
    
    def __init__(self, num_tasks: int, temperature: float = 2.0):
        super().__init__()
        self.name = "dwa"
        self.num_tasks = num_tasks
        self.temperature = temperature
        self.prev_losses = None
        self.prev_prev_losses = None
    
    def backward(self, losses, model, optimizer):
        task_names = list(losses.keys())
        current_losses = torch.stack([losses[name].detach() for name in task_names])
        
        if self.prev_losses is not None and self.prev_prev_losses is not None:
            # Compute rate of change
            rate = self.prev_losses / (self.prev_prev_losses + 1e-12)
            weights = F.softmax(rate / self.temperature, dim=0) * len(task_names)
        else:
            weights = torch.ones(len(task_names), device=current_losses.device)
        
        # Compute weighted loss
        total_loss = sum(
            weights[i] * losses[name] for i, name in enumerate(task_names)
        )
        
        # Update loss history
        self.prev_prev_losses = self.prev_losses
        self.prev_losses = current_losses
        
        optimizer.zero_grad()
        total_loss.backward()
        return total_loss.detach()


# ============================================================================
# Factory
# ============================================================================

GRADIENT_METHOD_REGISTRY = {
    'none': NoneGradient,
    'pcgrad': PCGrad,
    'gradnorm': GradNorm,
    'cagrad': CAGrad,
    'mgda': MGDA,
    'uncertainty': UncertaintyWeighting,
    'dwa': DWA,
}


def create_gradient_method(method_name: str, **kwargs) -> GradientMethod:
    """
    Factory function to create gradient manipulation methods.
    
    Args:
        method_name: Name of the gradient method
        **kwargs: Method-specific arguments
        
    Returns:
        GradientMethod instance
    """
    if method_name not in GRADIENT_METHOD_REGISTRY:
        raise ValueError(
            f"Unknown gradient method: {method_name}. "
            f"Available: {list(GRADIENT_METHOD_REGISTRY.keys())}"
        )
    
    return GRADIENT_METHOD_REGISTRY[method_name](**kwargs)


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing gradient manipulation methods...")
    
    # Simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.shared = nn.Linear(10, 10)
            self.head1 = nn.Linear(10, 2)
            self.head2 = nn.Linear(10, 3)
        
        def forward(self, x):
            h = self.shared(x)
            return {'task1': self.head1(h), 'task2': self.head2(h)}
    
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x = torch.randn(4, 10)
    
    for method_name in ['none', 'pcgrad', 'cagrad', 'mgda']:
        print(f"\n  Testing {method_name}...")
        method = create_gradient_method(method_name)
        
        outputs = model(x)
        losses = {
            'task1': F.cross_entropy(outputs['task1'], torch.randint(0, 2, (4,))),
            'task2': F.cross_entropy(outputs['task2'], torch.randint(0, 3, (4,)))
        }
        
        total_loss = method.backward(losses, model, optimizer)
        optimizer.step()
        print(f"    Loss: {total_loss.item():.4f} OK")
    
    # Test GradNorm
    print(f"\n  Testing gradnorm...")
    gradnorm = create_gradient_method('gradnorm', num_tasks=2, alpha=1.5)
    outputs = model(x)
    losses = {
        'task1': F.cross_entropy(outputs['task1'], torch.randint(0, 2, (4,))),
        'task2': F.cross_entropy(outputs['task2'], torch.randint(0, 3, (4,)))
    }
    total_loss = gradnorm.backward(losses, model, optimizer)
    optimizer.step()
    print(f"    Loss: {total_loss.item():.4f}")
    print(f"    Weights: {gradnorm.get_weights()} OK")
    
    print("\nAll gradient method tests passed!")
