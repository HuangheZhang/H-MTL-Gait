"""Fold-local channel standardization for windowed sensor arrays."""

from typing import Any, Dict

import numpy as np
from sklearn.preprocessing import StandardScaler


class FoldStandardizer:
    """Fit channel statistics on training windows and reuse them unchanged."""

    scope = "training_fold_only"

    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.channels = None
        self.fit_sample_count = 0
        self._fitted = False

    @staticmethod
    def _validate_and_clean(data: np.ndarray) -> np.ndarray:
        array = np.asarray(data)
        if array.ndim != 3:
            raise ValueError(f"Expected data shaped (N, T, C), received {array.shape}")
        if any(size == 0 for size in array.shape):
            raise ValueError("FoldStandardizer requires non-empty N, T, and C dimensions")
        return np.nan_to_num(
            array.astype(np.float64, copy=False),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    def fit(self, training_data: np.ndarray) -> "FoldStandardizer":
        clean = self._validate_and_clean(training_data)
        self.channels = int(clean.shape[2])
        flat = clean.reshape(-1, self.channels)
        self.scaler.fit(flat)
        self.fit_sample_count = int(flat.shape[0])
        self._fitted = True
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("FoldStandardizer must be fitted on training data before transform")
        clean = self._validate_and_clean(data)
        if clean.shape[2] != self.channels:
            raise ValueError(
                f"Expected {self.channels} channels, received {clean.shape[2]}"
            )
        shape = clean.shape
        transformed = self.scaler.transform(clean.reshape(-1, self.channels))
        transformed = np.nan_to_num(
            transformed, nan=0.0, posinf=0.0, neginf=0.0
        )
        return transformed.reshape(shape).astype(np.float32, copy=False)

    def fit_transform(self, training_data: np.ndarray) -> np.ndarray:
        return self.fit(training_data).transform(training_data)

    def state_dict(self) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("FoldStandardizer has no fitted state")
        return {
            "scope": self.scope,
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "channels": self.channels,
            "fit_sample_count": self.fit_sample_count,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> "FoldStandardizer":
        if state.get("scope") != self.scope:
            raise ValueError(f"Expected preprocessing scope {self.scope!r}")
        channels = int(state["channels"])
        mean = np.asarray(state["mean"], dtype=np.float64)
        scale = np.asarray(state["scale"], dtype=np.float64)
        if mean.shape != (channels,) or scale.shape != (channels,):
            raise ValueError("Preprocessing state dimensions do not match channels")
        if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Preprocessing state contains invalid mean or scale values")
        self.channels = channels
        self.fit_sample_count = int(state["fit_sample_count"])
        self.scaler.mean_ = mean
        self.scaler.scale_ = scale
        self.scaler.var_ = scale ** 2
        self.scaler.n_features_in_ = channels
        self.scaler.n_samples_seen_ = self.fit_sample_count
        self._fitted = True
        return self
