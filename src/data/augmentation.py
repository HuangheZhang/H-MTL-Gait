"""
Time Series Data Augmentation for Gait Analysis
=================================================
Augmentation strategies specifically designed for IMU sensor data.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, List


class TimeSeriesAugmentor:
    """
    Collection of augmentation methods for time series / sensor data.
    
    All transforms operate on tensors of shape (batch, seq_len, channels) 
    or (seq_len, channels).
    """
    
    def __init__(
        self,
        jitter_sigma: float = 0.05,
        scale_sigma: float = 0.1,
        rotation_range: float = 0.1,
        permutation_max_segments: int = 5,
        time_warp_sigma: float = 0.2,
        crop_ratio: float = 0.9,
        cutout_ratio: float = 0.1,
        mixup_alpha: float = 0.4,
        magnitude_warp_sigma: float = 0.2,
        p: float = 0.5
    ):
        """
        Args:
            jitter_sigma: Standard deviation for Gaussian noise
            scale_sigma: Standard deviation for random scaling
            rotation_range: Range for rotation in radians
            permutation_max_segments: Max number of segments for permutation
            time_warp_sigma: Sigma for time warping
            crop_ratio: Ratio for random cropping
            cutout_ratio: Ratio of sequence to cut out
            mixup_alpha: Alpha for beta distribution in mixup
            magnitude_warp_sigma: Sigma for magnitude warping
            p: Probability of applying each augmentation
        """
        self.jitter_sigma = jitter_sigma
        self.scale_sigma = scale_sigma
        self.rotation_range = rotation_range
        self.permutation_max_segments = permutation_max_segments
        self.time_warp_sigma = time_warp_sigma
        self.crop_ratio = crop_ratio
        self.cutout_ratio = cutout_ratio
        self.mixup_alpha = mixup_alpha
        self.magnitude_warp_sigma = magnitude_warp_sigma
        self.p = p
    
    # -----------------------------------------------------------------------
    # Basic Augmentations
    # -----------------------------------------------------------------------
    
    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        if np.random.random() > self.p:
            return x
        noise = torch.randn_like(x) * self.jitter_sigma
        return x + noise
    
    def scaling(self, x: torch.Tensor) -> torch.Tensor:
        """Random amplitude scaling per channel."""
        if np.random.random() > self.p:
            return x
        # One scale factor per channel, broadcast across time
        if x.dim() == 3:
            scales = torch.randn(x.size(0), 1, x.size(2), device=x.device) * self.scale_sigma + 1.0
        else:
            scales = torch.randn(1, x.size(1), device=x.device) * self.scale_sigma + 1.0
        return x * scales
    
    def rotation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Random rotation for 3-axis sensor groups.
        Applies small rotation matrices to each (X,Y,Z) triplet.
        """
        if np.random.random() > self.p:
            return x
        
        result = x.clone()
        ndim = x.dim()
        if ndim == 2:
            result = result.unsqueeze(0)
        
        batch_size, seq_len, channels = result.shape
        
        # Process every 3 channels as an (X, Y, Z) triplet
        for start in range(0, channels - 2, 3):
            # Random small rotation angles
            angles = np.random.uniform(
                -self.rotation_range, self.rotation_range, size=3
            )
            
            # Rotation matrix (Rodrigues' formula, small angle approx)
            Rx = torch.tensor([
                [1, 0, 0],
                [0, np.cos(angles[0]), -np.sin(angles[0])],
                [0, np.sin(angles[0]),  np.cos(angles[0])]
            ], dtype=x.dtype, device=x.device)
            
            Ry = torch.tensor([
                [ np.cos(angles[1]), 0, np.sin(angles[1])],
                [0, 1, 0],
                [-np.sin(angles[1]), 0, np.cos(angles[1])]
            ], dtype=x.dtype, device=x.device)
            
            Rz = torch.tensor([
                [np.cos(angles[2]), -np.sin(angles[2]), 0],
                [np.sin(angles[2]),  np.cos(angles[2]), 0],
                [0, 0, 1]
            ], dtype=x.dtype, device=x.device)
            
            R = Rz @ Ry @ Rx  # Combined rotation
            
            # Apply rotation to (batch, seq_len, 3) chunk
            triplet = result[:, :, start:start + 3]  # (batch, seq_len, 3)
            rotated = torch.einsum('ij,bsj->bsi', R, triplet)
            result[:, :, start:start + 3] = rotated
        
        if ndim == 2:
            result = result.squeeze(0)
        
        return result
    
    def permutation(self, x: torch.Tensor) -> torch.Tensor:
        """Randomly permute temporal segments."""
        if np.random.random() > self.p:
            return x
        
        result = x.clone()
        ndim = x.dim()
        if ndim == 2:
            result = result.unsqueeze(0)
        
        batch_size, seq_len, channels = result.shape
        n_segments = np.random.randint(2, self.permutation_max_segments + 1)
        
        # Create segment boundaries
        boundaries = np.sort(np.random.choice(seq_len - 1, n_segments - 1, replace=False) + 1)
        boundaries = np.concatenate([[0], boundaries, [seq_len]])
        
        # Permute segments
        perm = np.random.permutation(n_segments)
        
        new_data = []
        for idx in perm:
            new_data.append(result[:, boundaries[idx]:boundaries[idx + 1], :])
        
        result = torch.cat(new_data, dim=1)
        
        if ndim == 2:
            result = result.squeeze(0)
        
        return result
    
    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Time warping: stretch/compress random temporal regions.
        Uses smooth random temporal distortion.
        """
        if np.random.random() > self.p:
            return x
        
        result = x.clone()
        ndim = x.dim()
        if ndim == 2:
            result = result.unsqueeze(0)
        
        batch_size, seq_len, channels = result.shape
        
        # Create warping function
        n_knots = 4
        knot_positions = np.linspace(0, 1, n_knots)
        knot_values = knot_positions + np.random.randn(n_knots) * self.time_warp_sigma
        knot_values = np.clip(knot_values, 0, 1)
        knot_values[0] = 0
        knot_values[-1] = 1
        knot_values = np.sort(knot_values)
        
        # Interpolate to get new time indices
        original_indices = np.linspace(0, 1, seq_len)
        warped_indices = np.interp(original_indices, knot_positions, knot_values)
        warped_indices = warped_indices * (seq_len - 1)
        
        # Integer indices for gathering
        idx_floor = np.clip(np.floor(warped_indices).astype(int), 0, seq_len - 1)
        idx_ceil = np.clip(np.ceil(warped_indices).astype(int), 0, seq_len - 1)
        alpha = torch.tensor(
            warped_indices - idx_floor, dtype=x.dtype, device=x.device
        ).unsqueeze(0).unsqueeze(-1)
        
        # Linear interpolation
        result = (
            (1 - alpha) * result[:, idx_floor, :] + 
            alpha * result[:, idx_ceil, :]
        )
        
        if ndim == 2:
            result = result.squeeze(0)
        
        return result
    
    def magnitude_warp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Magnitude warping: smoothly varying amplitude changes over time.
        """
        if np.random.random() > self.p:
            return x
        
        result = x.clone()
        ndim = x.dim()
        if ndim == 2:
            result = result.unsqueeze(0)
        
        batch_size, seq_len, channels = result.shape
        
        # Create smooth magnitude curve
        n_knots = 4
        knot_positions = np.linspace(0, seq_len - 1, n_knots)
        knot_values = 1.0 + np.random.randn(n_knots) * self.magnitude_warp_sigma
        
        # Interpolate to full sequence length
        time_steps = np.arange(seq_len)
        magnitude = np.interp(time_steps, knot_positions, knot_values)
        magnitude = torch.tensor(
            magnitude, dtype=x.dtype, device=x.device
        ).unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
        
        result = result * magnitude
        
        if ndim == 2:
            result = result.squeeze(0)
        
        return result
    
    def cutout(self, x: torch.Tensor) -> torch.Tensor:
        """Random temporal cutout (set a segment to zero)."""
        if np.random.random() > self.p:
            return x
        
        result = x.clone()
        ndim = x.dim()
        if ndim == 2:
            result = result.unsqueeze(0)
        
        _, seq_len, _ = result.shape
        cut_len = max(1, int(seq_len * self.cutout_ratio))
        start = np.random.randint(0, max(1, seq_len - cut_len))
        
        result[:, start:start + cut_len, :] = 0
        
        if ndim == 2:
            result = result.squeeze(0)
        
        return result
    
    def channel_dropout(self, x: torch.Tensor, drop_prob: float = 0.1) -> torch.Tensor:
        """Randomly zero out entire channels."""
        if np.random.random() > self.p:
            return x
        
        result = x.clone()
        ndim = x.dim()
        if ndim == 2:
            channels = x.size(1)
            mask = torch.bernoulli(torch.ones(channels) * (1 - drop_prob)).to(x.device)
            result = result * mask.unsqueeze(0)
        else:
            channels = x.size(2)
            mask = torch.bernoulli(torch.ones(channels) * (1 - drop_prob)).to(x.device)
            result = result * mask.unsqueeze(0).unsqueeze(0)
        
        return result
    
    # -----------------------------------------------------------------------
    # Composite Augmentation
    # -----------------------------------------------------------------------
    
    def __call__(
        self, 
        x: torch.Tensor, 
        augment_types: Optional[List[str]] = None
    ) -> torch.Tensor:
        """
        Apply a random subset of augmentations.
        
        Args:
            x: Input tensor (batch, seq_len, channels) or (seq_len, channels)
            augment_types: List of augmentation names to apply. 
                          If None, applies all with probability p.
        """
        if augment_types is None:
            augment_types = [
                'jitter', 'scaling', 'rotation', 
                'magnitude_warp', 'cutout'
            ]
        
        augment_map = {
            'jitter': self.jitter,
            'scaling': self.scaling,
            'rotation': self.rotation,
            'permutation': self.permutation,
            'time_warp': self.time_warp,
            'magnitude_warp': self.magnitude_warp,
            'cutout': self.cutout,
            'channel_dropout': self.channel_dropout,
        }
        
        for aug_name in augment_types:
            if aug_name in augment_map:
                x = augment_map[aug_name](x)
        
        return x


# ============================================================================
# Mixup Augmentation (batch-level)
# ============================================================================

def mixup_data(
    x: torch.Tensor,
    labels: dict,
    alpha: float = 0.4
) -> Tuple[torch.Tensor, dict, dict, float]:
    """
    Mixup augmentation at the batch level.
    
    Linearly interpolates between pairs of samples and their labels.
    
    Args:
        x: Input batch (batch, seq_len, channels)
        labels: Dictionary of label tensors
        alpha: Beta distribution parameter
        
    Returns:
        mixed_x: Mixed input
        labels_a: Original labels
        labels_b: Shuffled labels
        lam: Mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    labels_b = {k: v[index] for k, v in labels.items()}
    
    return mixed_x, labels, labels_b, lam


# ============================================================================
# CutMix Augmentation (batch-level, temporal)
# ============================================================================

def cutmix_data(
    x: torch.Tensor,
    labels: dict,
    alpha: float = 1.0
) -> Tuple[torch.Tensor, dict, dict, float]:
    """
    CutMix augmentation adapted for time series.
    
    Cuts a temporal segment from one sample and pastes it into another.
    
    Args:
        x: Input batch (batch, seq_len, channels)
        labels: Dictionary of label tensors
        alpha: Beta distribution parameter
        
    Returns:
        mixed_x: Mixed input
        labels_a: Original labels
        labels_b: Shuffled labels
        lam: Effective mixing coefficient
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    
    batch_size, seq_len, _ = x.shape
    index = torch.randperm(batch_size, device=x.device)
    
    # Determine cut boundaries
    cut_len = int(seq_len * (1 - lam))
    start = np.random.randint(0, max(1, seq_len - cut_len))
    end = start + cut_len
    
    # Apply CutMix
    mixed_x = x.clone()
    mixed_x[:, start:end, :] = x[index, start:end, :]
    
    # Adjust lambda to the actual proportion
    lam = 1 - (end - start) / seq_len
    
    labels_b = {k: v[index] for k, v in labels.items()}
    
    return mixed_x, labels, labels_b, lam


# ============================================================================
# Augmented Dataset Wrapper
# ============================================================================

class AugmentedGaitDataset(torch.utils.data.Dataset):
    """
    Wraps a GaitDataset with on-the-fly data augmentation.
    """
    
    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        augmentor: Optional[TimeSeriesAugmentor] = None,
        augment_types: Optional[List[str]] = None
    ):
        self.dataset = dataset
        self.augmentor = augmentor or TimeSeriesAugmentor()
        self.augment_types = augment_types
        self._training = True
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, labels = self.dataset[idx]
        
        if self.training_mode:
            x = self.augmentor(x, self.augment_types)
        
        return x, labels
    
    @property
    def training_mode(self):
        return getattr(self, '_training', True)
    
    def train(self):
        self._training = True
        return self
    
    def eval(self):
        self._training = False
        return self


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing data augmentation...")
    
    # Create sample data
    batch_size = 8
    seq_len = 200
    channels = 36
    x = torch.randn(batch_size, seq_len, channels)
    
    augmentor = TimeSeriesAugmentor(p=1.0)
    
    augmentation_methods = [
        'jitter', 'scaling', 'rotation', 'permutation',
        'time_warp', 'magnitude_warp', 'cutout', 'channel_dropout'
    ]
    
    for method_name in augmentation_methods:
        aug_fn = getattr(augmentor, method_name)
        augmented = aug_fn(x)
        diff = (augmented - x).abs().mean().item()
        print(f"  {method_name}: shape={augmented.shape}, avg_diff={diff:.4f} OK")
    
    # Test composite
    print("\n  Composite augmentation:")
    augmented = augmentor(x)
    print(f"    shape={augmented.shape} OK")
    
    # Test mixup
    print("\n  Mixup:")
    labels = {'binary': torch.randint(0, 2, (batch_size,))}
    mixed, la, lb, lam = mixup_data(x, labels)
    print(f"    shape={mixed.shape}, lam={lam:.4f} OK")
    
    # Test cutmix
    print("\n  CutMix:")
    mixed, la, lb, lam = cutmix_data(x, labels)
    print(f"    shape={mixed.shape}, lam={lam:.4f} OK")
    
    print("\nAll augmentation tests passed!")
