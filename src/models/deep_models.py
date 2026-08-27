"""
Deep Learning Models for Gait Analysis
======================================
1D-CNN, LSTM, Transformer, and hybrid architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class Conv1DBlock(nn.Module):
    """1D Convolutional block with BatchNorm and activation."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class CNN1DEncoder(nn.Module):
    """1D CNN Encoder for time series."""
    
    def __init__(
        self,
        input_channels: int = 36,
        channels: List[int] = [64, 128, 256],
        kernel_sizes: List[int] = [7, 5, 3],
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList()
        in_ch = input_channels
        
        for out_ch, ks in zip(channels, kernel_sizes):
            self.blocks.append(
                Conv1DBlock(in_ch, out_ch, ks, padding=ks//2, dropout=dropout)
            )
            self.blocks.append(nn.MaxPool1d(2))
            in_ch = out_ch
            
        self.output_channels = channels[-1]
        
    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        for block in self.blocks:
            x = block(x)
            
        # Global average pooling
        x = x.mean(dim=-1)  # (batch, channels)
        
        return x


class LSTMEncoder(nn.Module):
    """Bidirectional LSTM Encoder."""
    
    def __init__(
        self,
        input_channels: int = 36,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        self.output_channels = hidden_size * (2 if bidirectional else 1)
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        output, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from both directions
        if self.lstm.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        else:
            hidden = h_n[-1]
            
        return hidden


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer Encoder for time series."""
    
    def __init__(
        self,
        input_channels: int = 36,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3,
        max_len: int = 500
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_channels, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_channels = d_model
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        batch_size = x.size(0)
        
        # Project to d_model
        x = self.input_proj(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Return CLS token representation
        return x[:, 0]


class CNN_LSTMEncoder(nn.Module):
    """Hybrid CNN-LSTM Encoder."""
    
    def __init__(
        self,
        input_channels: int = 36,
        cnn_channels: List[int] = [64, 128],
        cnn_kernels: List[int] = [7, 5],
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # CNN layers
        self.cnn_blocks = nn.ModuleList()
        in_ch = input_channels
        
        for out_ch, ks in zip(cnn_channels, cnn_kernels):
            self.cnn_blocks.append(
                Conv1DBlock(in_ch, out_ch, ks, padding=ks//2, dropout=dropout)
            )
            in_ch = out_ch
            
        # LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.output_channels = lstm_hidden * 2
        
    def forward(self, x):
        # x: (batch, seq_len, channels) -> (batch, channels, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        for block in self.cnn_blocks:
            x = block(x)
            
        # (batch, channels, seq_len) -> (batch, seq_len, channels)
        x = x.transpose(1, 2)
        
        # LSTM
        output, (h_n, c_n) = self.lstm(x)
        
        # Concatenate last hidden states
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        
        return hidden


class TaskHead(nn.Module):
    """Task-specific head for classification or regression."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 2,
        task_type: str = 'classification',
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.task_type = task_type
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim if task_type == 'classification' else 1)
        )
        
    def forward(self, x):
        return self.fc(x)


class SingleTaskModel(nn.Module):
    """Single-task model with encoder and task head."""
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_channels: int = 36,
        num_classes: int = 2,
        task_type: str = 'classification',
        **encoder_kwargs
    ):
        super().__init__()
        
        # Select encoder
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
            
        # Task head
        self.head = TaskHead(
            self.encoder.output_channels,
            output_dim=num_classes,
            task_type=task_type
        )
        
    def forward(self, x):
        features = self.encoder(x)
        return self.head(features)


class MultiTaskModel(nn.Module):
    """Multi-task model with shared encoder and task-specific heads."""
    
    def __init__(
        self,
        encoder_type: str = 'transformer',
        input_channels: int = 36,
        tasks: Dict[str, Dict] = None,
        **encoder_kwargs
    ):
        """
        Args:
            encoder_type: Type of encoder (cnn1d, lstm, transformer, cnn_lstm)
            input_channels: Number of input channels
            tasks: Dictionary mapping task names to task configs
                   e.g., {'binary': {'num_classes': 2, 'type': 'classification'}}
        """
        super().__init__()
        
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
            
        # Task-specific heads
        self.tasks = tasks or {
            'binary': {'num_classes': 2, 'type': 'classification'},
            'coarse': {'num_classes': 3, 'type': 'classification'},
            'fine': {'num_classes': 8, 'type': 'classification'},
            'regression': {'num_classes': 1, 'type': 'regression'}
        }
        
        self.heads = nn.ModuleDict()
        for task_name, task_config in self.tasks.items():
            self.heads[task_name] = TaskHead(
                self.encoder.output_channels,
                output_dim=task_config['num_classes'],
                task_type=task_config['type']
            )
            
    def forward(self, x, task: str = None):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            task: If specified, only return output for this task
            
        Returns:
            Dictionary of task outputs or single task output
        """
        features = self.encoder(x)
        
        if task is not None:
            return self.heads[task](features)
        
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(features)
            
        return outputs
    
    def get_features(self, x):
        """Get shared features without task heads."""
        return self.encoder(x)


def create_model(
    model_type: str,
    encoder_type: str = 'transformer',
    input_channels: int = 36,
    **kwargs
) -> nn.Module:
    """Factory function to create models."""
    
    if model_type == 'single_task':
        return SingleTaskModel(encoder_type, input_channels, **kwargs)
    elif model_type == 'multi_task':
        return MultiTaskModel(encoder_type, input_channels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    batch_size = 8
    seq_len = 200
    channels = 36
    
    x = torch.randn(batch_size, seq_len, channels)
    
    # Test CNN1D
    print("Testing CNN1D Encoder...")
    encoder = CNN1DEncoder(channels)
    out = encoder(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    # Test LSTM
    print("Testing LSTM Encoder...")
    encoder = LSTMEncoder(channels)
    out = encoder(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    # Test Transformer
    print("Testing Transformer Encoder...")
    encoder = TransformerEncoder(channels)
    out = encoder(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    # Test Multi-task model
    print("\nTesting Multi-Task Model...")
    model = MultiTaskModel('transformer', channels)
    outputs = model(x)
    for task, out in outputs.items():
        print(f"  {task}: {out.shape}")
