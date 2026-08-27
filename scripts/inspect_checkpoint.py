#!/usr/bin/env python3
"""Inspect the trusted Phase-1 checkpoint without requiring gait data."""

import argparse
import hashlib
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = REPO_ROOT / "checkpoints" / "hmtl_mmoe_dwa_example.pt"
EXPECTED_SHA256 = "1eca97d7390988a7c96c411fed446ed2fd8038a981b51d8383cb1b008f7bd89a"
sys.path.insert(0, str(REPO_ROOT))

from src.models.mtl_models import MMoEMTL  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_verified(path: Path):
    digest = sha256(path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Refusing to deserialize unexpected checkpoint SHA-256: {digest}")
    # The hash-pinned release artifact contains NumPy metric objects, so weights_only=True
    # cannot read the complete dictionary on some PyTorch versions.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # compatibility with older supported PyTorch releases
        checkpoint = torch.load(path, map_location="cpu")
    return checkpoint, digest


def build_model() -> MMoEMTL:
    return MMoEMTL(
        input_channels=36,
        num_experts=4,
        expert_hidden=128,
        expert_output_dim=128,
        head_hidden_dim=64,
        dropout=0.35,
        seq_length=200,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    path = args.checkpoint.expanduser().resolve()
    checkpoint, digest = load_verified(path)
    state = checkpoint["model_state_dict"]
    model = build_model()
    model.load_state_dict(state, strict=True)

    print(f"path: {path}")
    print(f"sha256: {digest}")
    print(f"epoch: {checkpoint.get('epoch')}")
    print("artifact_role: example_pretrained_weight")
    print(f"state_dict_tensors: {len(state)}")
    print(f"parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
    print("strict_load: ok")
    print("input_contract: (B, 200, 36), floating-point, historically global-z-scored")
    print("tasks: " + ", ".join(model.task_configs))
    metrics = checkpoint.get("metrics", {})
    print(f"saved_validation_metric_count: {len(metrics)}")
    if "val_binary_accuracy" in metrics:
        print(f"saved_val_binary_accuracy: {float(metrics['val_binary_accuracy']):.16g}")


if __name__ == "__main__":
    main()
