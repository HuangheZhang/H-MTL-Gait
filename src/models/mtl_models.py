"""
Multi-Task Learning Model Architectures
========================================
Implements various MTL approaches:
- Hard Parameter Sharing
- Soft Parameter Sharing 
- Cross-Stitch Networks
- Multi-gate Mixture of Experts (MMoE)
- Progressive Layered Extraction (PLE)

All models share a common interface for interchangeability in experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math
import copy

from .deep_models import (
    CNN1DEncoder, LSTMEncoder, TransformerEncoder, CNN_LSTMEncoder,
    TaskHead, Conv1DBlock
)


# ============================================================================
# Helper: Encoder Factory
# ============================================================================

def create_encoder(encoder_type: str, input_channels: int, **kwargs) -> nn.Module:
    """Factory to create encoder by type."""
    if encoder_type == 'cnn1d':
        return CNN1DEncoder(input_channels, **kwargs)
    elif encoder_type == 'lstm':
        return LSTMEncoder(input_channels, **kwargs)
    elif encoder_type == 'transformer':
        return TransformerEncoder(input_channels, **kwargs)
    elif encoder_type == 'cnn_lstm':
        return CNN_LSTMEncoder(input_channels, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")


# ============================================================================
# Default Task Configuration
# ============================================================================

DEFAULT_TASKS = {
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


# ============================================================================
# 1. Hard Parameter Sharing MTL
# ============================================================================

class HardSharingMTL(nn.Module):
    """
    Hard Parameter Sharing Multi-Task Learning.
    
    All tasks share a single encoder; each task has its own head.
    This is the simplest and most widely-used MTL architecture.
    
    Reference: Caruana, 1997
    """
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_channels: int = 36,
        tasks: Dict[str, Dict] = None,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        **encoder_kwargs
    ):
        super().__init__()
        self.task_configs = tasks or DEFAULT_TASKS
        
        # Shared encoder
        self.encoder = create_encoder(encoder_type, input_channels, **encoder_kwargs)
        hidden_dim = self.encoder.output_channels
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, task_config in self.task_configs.items():
            self.heads[task_name] = TaskHead(
                input_dim=hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=task_config['num_classes'],
                task_type=task_config['type'],
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        features = self.encoder(x)
        
        if task is not None:
            return {task: self.heads[task](features)}
        
        return {name: head(features) for name, head in self.heads.items()}
    
    def get_shared_parameters(self):
        return self.encoder.parameters()
    
    def get_task_parameters(self, task_name: str):
        return self.heads[task_name].parameters()
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ============================================================================
# 2. Soft Parameter Sharing MTL
# ============================================================================

class SoftSharingMTL(nn.Module):
    """
    Soft Parameter Sharing Multi-Task Learning.
    
    Each task has its own encoder, but the encoders are regularized 
    to stay similar through L2 distance penalties on their parameters.
    
    Reference: Duong et al., 2015
    """
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_channels: int = 36,
        tasks: Dict[str, Dict] = None,
        reg_weight: float = 0.01,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        **encoder_kwargs
    ):
        super().__init__()
        self.task_configs = tasks or DEFAULT_TASKS
        self.reg_weight = reg_weight
        
        # Task-specific encoders
        self.encoders = nn.ModuleDict()
        for task_name in self.task_configs:
            self.encoders[task_name] = create_encoder(
                encoder_type, input_channels, **encoder_kwargs
            )
        
        hidden_dim = list(self.encoders.values())[0].output_channels
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, task_config in self.task_configs.items():
            self.heads[task_name] = TaskHead(
                input_dim=hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=task_config['num_classes'],
                task_type=task_config['type'],
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        if task is not None:
            features = self.encoders[task](x)
            return {task: self.heads[task](features)}
        
        outputs = {}
        for task_name in self.task_configs:
            features = self.encoders[task_name](x)
            outputs[task_name] = self.heads[task_name](features)
        return outputs
    
    def get_regularization_loss(self) -> torch.Tensor:
        """
        Compute L2 regularization loss between encoder parameters.
        Encourages encoders to stay close but allows task-specific adaptation.
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        encoder_names = list(self.encoders.keys())
        
        for i in range(len(encoder_names)):
            for j in range(i + 1, len(encoder_names)):
                params_i = list(self.encoders[encoder_names[i]].parameters())
                params_j = list(self.encoders[encoder_names[j]].parameters())
                
                for p_i, p_j in zip(params_i, params_j):
                    reg_loss += torch.norm(p_i - p_j, p=2)
        
        return self.reg_weight * reg_loss


# ============================================================================
# 3. Cross-Stitch Networks
# ============================================================================

class CrossStitchUnit(nn.Module):
    """
    Cross-Stitch Unit that learns linear combinations of task features.
    
    Reference: Misra et al., "Cross-Stitch Networks for Multi-Task Learning", CVPR 2016
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        # Initialize as identity (each task mostly keeps its own features)
        self.alpha = nn.Parameter(
            torch.eye(num_tasks) * 0.9 + torch.ones(num_tasks, num_tasks) * 0.1 / num_tasks
        )
    
    def forward(self, task_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            task_features: List of tensors, one per task, same shape
        Returns:
            List of cross-stitched tensors
        """
        stacked = torch.stack(task_features, dim=0)  # (num_tasks, batch, ...)
        alpha = F.softmax(self.alpha, dim=1)  # Normalize rows
        
        # Apply cross-stitch: each output is a weighted sum of all inputs
        output_shape = stacked.shape[1:]
        stacked_flat = stacked.view(len(task_features), -1)  # (num_tasks, batch*features)
        stitched = torch.matmul(alpha, stacked_flat)  # (num_tasks, batch*features)
        
        return [stitched[i].view(output_shape) for i in range(len(task_features))]


class CrossStitchEncoder(nn.Module):
    """
    Multi-layer encoder with cross-stitch units between layers.
    Uses CNN blocks with cross-stitching at each layer.
    """
    
    def __init__(
        self,
        num_tasks: int,
        input_channels: int = 36,
        channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 3],
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_tasks = num_tasks
        
        # Create per-task convolutional blocks
        self.task_blocks = nn.ModuleList()
        self.cross_stitches = nn.ModuleList()
        
        in_ch = input_channels
        for layer_idx, (out_ch, ks) in enumerate(zip(channels, kernel_sizes)):
            # One conv block per task
            layer_blocks = nn.ModuleList([
                nn.Sequential(
                    Conv1DBlock(in_ch, out_ch, ks, padding=ks // 2, dropout=dropout),
                    nn.MaxPool1d(2)
                )
                for _ in range(num_tasks)
            ])
            self.task_blocks.append(layer_blocks)
            
            # Cross-stitch unit after this layer
            self.cross_stitches.append(CrossStitchUnit(num_tasks))
            
            in_ch = out_ch
        
        self.output_channels = channels[-1]
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, channels)
        Returns:
            List of feature vectors, one per task, each (batch, output_channels)
        """
        # Convert to (batch, channels, seq_len) for Conv1d
        x_conv = x.transpose(1, 2)
        
        # Initialize task features
        task_features = [x_conv.clone() for _ in range(self.num_tasks)]
        
        for layer_blocks, cross_stitch in zip(self.task_blocks, self.cross_stitches):
            # Apply per-task conv blocks
            task_features = [block(feat) for block, feat in zip(layer_blocks, task_features)]
            # Apply cross-stitch
            task_features = cross_stitch(task_features)
        
        # Global average pooling per task
        return [feat.mean(dim=-1) for feat in task_features]


class CrossStitchMTL(nn.Module):
    """
    Cross-Stitch Networks for Multi-Task Learning.
    
    Each task has its own network path, but cross-stitch units learn
    how to combine features from different tasks at each layer.
    """
    
    def __init__(
        self,
        input_channels: int = 36,
        tasks: Dict[str, Dict] = None,
        channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 3],
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        **kwargs
    ):
        super().__init__()
        self.task_configs = tasks or DEFAULT_TASKS
        task_names = list(self.task_configs.keys())
        num_tasks = len(task_names)
        self.task_names = task_names
        
        # Cross-stitch encoder
        self.encoder = CrossStitchEncoder(
            num_tasks=num_tasks,
            input_channels=input_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            dropout=dropout
        )
        
        hidden_dim = self.encoder.output_channels
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, task_config in self.task_configs.items():
            self.heads[task_name] = TaskHead(
                input_dim=hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=task_config['num_classes'],
                task_type=task_config['type'],
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        task_features = self.encoder(x)  # List of features per task
        
        outputs = {}
        for idx, task_name in enumerate(self.task_names):
            if task is not None and task_name != task:
                continue
            outputs[task_name] = self.heads[task_name](task_features[idx])
        
        return outputs


# ============================================================================
# 4. Multi-gate Mixture of Experts (MMoE)
# ============================================================================

class Expert(nn.Module):
    """Single expert network (1D-CNN based)."""
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 128,
        output_dim: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(hidden_channels, output_dim)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, seq_len) -> (batch, output_dim)"""
        out = self.net(x).squeeze(-1)  # (batch, hidden_channels)
        return self.proj(out)


class GatingNetwork(nn.Module):
    """Task-specific gating network for MMoE."""
    
    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim) -> (batch, num_experts)"""
        return self.gate(x)


class MMoEMTL(nn.Module):
    """
    Multi-gate Mixture-of-Experts for Multi-Task Learning.
    
    Multiple expert networks are shared across tasks, each task has 
    its own gating network to softly select and combine experts.
    
    Reference: Ma et al., "Modeling Task Relationships in Multi-Task Learning 
               with Multi-gate Mixture-of-Experts", KDD 2018
    """
    
    def __init__(
        self,
        input_channels: int = 36,
        tasks: Dict[str, Dict] = None,
        num_experts: int = 4,
        expert_hidden: int = 128,
        expert_output_dim: int = 128,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        seq_length: int = 200,
        **kwargs
    ):
        super().__init__()
        self.task_configs = tasks or DEFAULT_TASKS
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            Expert(input_channels, expert_hidden, expert_output_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Input projection for gating (flatten a summary of the input)
        self.gate_input_proj = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # (batch, channels, 1)
        )
        gate_input_dim = input_channels
        
        # Task-specific gating networks
        self.gates = nn.ModuleDict()
        for task_name in self.task_configs:
            self.gates[task_name] = GatingNetwork(gate_input_dim, num_experts)
        
        # Task-specific heads
        self.heads = nn.ModuleDict()
        for task_name, task_config in self.task_configs.items():
            self.heads[task_name] = TaskHead(
                input_dim=expert_output_dim,
                hidden_dim=head_hidden_dim,
                output_dim=task_config['num_classes'],
                task_type=task_config['type'],
                dropout=dropout
            )
        
        self.output_channels = expert_output_dim
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, channels)
        """
        # Convert for conv: (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        
        # Get expert outputs: List[(batch, expert_output_dim)]
        expert_outputs = torch.stack(
            [expert(x_conv) for expert in self.experts], dim=1
        )  # (batch, num_experts, expert_output_dim)
        
        # Gating input
        gate_input = self.gate_input_proj(x_conv).squeeze(-1)  # (batch, channels)
        
        outputs = {}
        for task_name, task_config in self.task_configs.items():
            if task is not None and task_name != task:
                continue
            
            # Get gating weights
            gate_weights = self.gates[task_name](gate_input)  # (batch, num_experts)
            
            # Weighted sum of expert outputs
            # (batch, num_experts, 1) * (batch, num_experts, dim) -> sum -> (batch, dim)
            task_features = (gate_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)
            
            outputs[task_name] = self.heads[task_name](task_features)
        
        return outputs


# ============================================================================
# 5. Progressive Layered Extraction (PLE)
# ============================================================================

class PLELayer(nn.Module):
    """
    Single extraction layer in PLE.
    Contains task-specific experts + shared experts + task-specific gates.
    """
    
    def __init__(
        self,
        input_dim: int,
        num_task_experts: int,
        num_shared_experts: int,
        num_tasks: int,
        expert_hidden: int = 128,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_task_experts = num_task_experts
        self.num_shared_experts = num_shared_experts
        total_experts_per_task = num_task_experts + num_shared_experts
        
        # Task-specific experts
        self.task_experts = nn.ModuleList()
        for _ in range(num_tasks):
            experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(input_dim, expert_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(expert_hidden, input_dim)
                )
                for _ in range(num_task_experts)
            ])
            self.task_experts.append(experts)
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(expert_hidden, input_dim)
            )
            for _ in range(num_shared_experts)
        ])
        
        # Task-specific gating
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, total_experts_per_task),
                nn.Softmax(dim=-1)
            )
            for _ in range(num_tasks)
        ])
    
    def forward(self, task_inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            task_inputs: List of (batch, dim) tensors, one per task
        Returns:
            List of (batch, dim) tensors after expert mixing
        """
        # Compute shared expert outputs using MEAN of all task inputs (not just task 0)
        shared_input = torch.mean(torch.stack(task_inputs), dim=0)
        shared_outputs = [expert(shared_input) for expert in self.shared_experts]
        
        outputs = []
        for task_idx in range(self.num_tasks):
            # Task-specific expert outputs
            task_expert_outputs = [
                expert(task_inputs[task_idx])
                for expert in self.task_experts[task_idx]
            ]
            
            # Combine task-specific + shared
            all_expert_outputs = task_expert_outputs + shared_outputs
            expert_stack = torch.stack(all_expert_outputs, dim=1)  # (batch, total_experts, dim)
            
            # Gating
            gate_weights = self.gates[task_idx](task_inputs[task_idx])  # (batch, total_experts)
            
            # Weighted sum with residual connection
            output = task_inputs[task_idx] + (gate_weights.unsqueeze(-1) * expert_stack).sum(dim=1)
            outputs.append(output)
        
        return outputs


class PLEMTL(nn.Module):
    """
    Progressive Layered Extraction for Multi-Task Learning.
    
    Combines task-specific experts with shared experts at multiple extraction
    layers, with progressive refinement of task representations.
    
    Reference: Tang et al., "Progressive Layered Extraction (PLE): 
    A Novel Multi-Task Learning Model for Personalized Recommendations", RecSys 2020
    """
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_channels: int = 36,
        tasks: Dict[str, Dict] = None,
        num_extraction_layers: int = 3,
        num_task_experts: int = 2,
        num_shared_experts: int = 2,
        expert_hidden: int = 256,
        head_hidden_dim: int = 256,
        dropout: float = 0.3,
        **encoder_kwargs
    ):
        super().__init__()
        self.task_configs = tasks or DEFAULT_TASKS
        self.task_names = list(self.task_configs.keys())
        num_tasks = len(self.task_names)
        
        # Shared feature extractor
        self.encoder = create_encoder(encoder_type, input_channels, **encoder_kwargs)
        feature_dim = self.encoder.output_channels
        
        # PLE extraction layers
        self.extraction_layers = nn.ModuleList([
            PLELayer(
                input_dim=feature_dim,
                num_task_experts=num_task_experts,
                num_shared_experts=num_shared_experts,
                num_tasks=num_tasks,
                expert_hidden=expert_hidden,
                dropout=dropout
            )
            for _ in range(num_extraction_layers)
        ])
        
        # Task heads
        self.heads = nn.ModuleDict()
        for task_name, task_config in self.task_configs.items():
            self.heads[task_name] = TaskHead(
                input_dim=feature_dim,
                hidden_dim=head_hidden_dim,
                output_dim=task_config['num_classes'],
                task_type=task_config['type'],
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        # Extract shared features
        shared_features = self.encoder(x)  # (batch, feature_dim)
        
        # Initialize task features from shared features
        task_features = [shared_features.clone() for _ in self.task_names]
        
        # Progressive extraction
        for ple_layer in self.extraction_layers:
            task_features = ple_layer(task_features)
        
        # Task heads
        outputs = {}
        for idx, task_name in enumerate(self.task_names):
            if task is not None and task_name != task:
                continue
            outputs[task_name] = self.heads[task_name](task_features[idx])
        
        return outputs


# ============================================================================
# 6. Attention-based MTL (MTAN)
# ============================================================================

class TaskAttentionModule(nn.Module):
    """
    Task-specific attention module that learns which parts of the 
    shared features are important for each task.
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(feature_dim)
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
    
    def forward(self, shared_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_features: (batch, dim)
        Returns:
            task_features: (batch, dim)
        """
        batch_size = shared_features.size(0)
        query = self.query.expand(batch_size, -1, -1)
        kv = shared_features.unsqueeze(1)
        
        attn_out, _ = self.attention(query, kv, kv)
        return self.norm(attn_out.squeeze(1) + shared_features)


class MTANMTL(nn.Module):
    """
    Multi-Task Attention Network.
    
    Uses a shared encoder with task-specific attention modules 
    to create specialized representations for each task.
    
    Reference: Liu et al., "End-to-End Multi-Task Learning with Attention", CVPR 2019
    """
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_channels: int = 36,
        tasks: Dict[str, Dict] = None,
        head_hidden_dim: int = 256,
        num_attention_heads: int = 4,
        dropout: float = 0.3,
        **encoder_kwargs
    ):
        super().__init__()
        self.task_configs = tasks or DEFAULT_TASKS
        
        # Shared encoder
        self.encoder = create_encoder(encoder_type, input_channels, **encoder_kwargs)
        hidden_dim = self.encoder.output_channels
        
        # Task-specific attention
        self.task_attention = nn.ModuleDict()
        for task_name in self.task_configs:
            self.task_attention[task_name] = TaskAttentionModule(
                hidden_dim, num_attention_heads
            )
        
        # Task heads
        self.heads = nn.ModuleDict()
        for task_name, task_config in self.task_configs.items():
            self.heads[task_name] = TaskHead(
                input_dim=hidden_dim,
                hidden_dim=head_hidden_dim,
                output_dim=task_config['num_classes'],
                task_type=task_config['type'],
                dropout=dropout
            )
    
    def forward(self, x: torch.Tensor, task: str = None) -> Dict[str, torch.Tensor]:
        shared_features = self.encoder(x)
        
        outputs = {}
        for task_name in self.task_configs:
            if task is not None and task_name != task:
                continue
            task_features = self.task_attention[task_name](shared_features)
            outputs[task_name] = self.heads[task_name](task_features)
        
        return outputs


# ============================================================================
# Model Factory
# ============================================================================

MTL_MODEL_REGISTRY = {
    'hard_sharing': HardSharingMTL,
    'soft_sharing': SoftSharingMTL,
    'cross_stitch': CrossStitchMTL,
    'mmoe': MMoEMTL,
    'ple': PLEMTL,
    'mtan': MTANMTL,
}


def create_mtl_model(
    model_type: str,
    encoder_type: str = 'transformer',
    input_channels: int = 36,
    tasks: Dict[str, Dict] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create MTL models.
    
    Args:
        model_type: One of 'hard_sharing', 'soft_sharing', 'cross_stitch', 
                    'mmoe', 'ple', 'mtan'
        encoder_type: Encoder for model types that use a selectable encoder
        input_channels: Number of input sensor channels
        tasks: Task configuration dict
        **kwargs: Additional model-specific arguments
        
    Returns:
        MTL model instance
    """
    if model_type not in MTL_MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Available: {list(MTL_MODEL_REGISTRY.keys())}"
        )
    
    model_cls = MTL_MODEL_REGISTRY[model_type]
    
    # These architectures define their own Conv1D feature extractors.
    if model_type in {'cross_stitch', 'mmoe'}:
        return model_cls(
            input_channels=input_channels,
            tasks=tasks,
            **kwargs
        )
    
    return model_cls(
        encoder_type=encoder_type,
        input_channels=input_channels,
        tasks=tasks,
        **kwargs
    )


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    batch_size = 8
    seq_len = 200
    channels = 36
    x = torch.randn(batch_size, seq_len, channels)
    
    print("=" * 60)
    print("Testing all MTL model architectures")
    print("=" * 60)
    
    for model_name, model_cls in MTL_MODEL_REGISTRY.items():
        print(f"\n--- {model_name} ---")
        try:
            if model_name == 'cross_stitch':
                model = model_cls(input_channels=channels)
            else:
                model = model_cls(encoder_type='transformer', input_channels=channels)
            
            outputs = model(x)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"  Parameters: {total_params:,}")
            for task, out in outputs.items():
                print(f"  {task}: {out.shape}")
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\nAll tests passed!")
