#!/usr/bin/env python3
"""Run exact checkpoint inference on a zero/random tensor or a local .npy array."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.inspect_checkpoint import DEFAULT_CHECKPOINT, build_model, load_verified  # noqa: E402

CLASS_TASKS = {"binary", "coarse", "fine", "vga_class", "gender", "neuro_fine"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--input", type=Path, help="Optional .npy array shaped (B,200,36) or (200,36)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--random", action="store_true", help="Use seeded N(0,1) input instead of zeros")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_input(args: argparse.Namespace) -> torch.Tensor:
    if args.input is not None:
        array = np.load(args.input.expanduser(), allow_pickle=False)
        if array.ndim == 2:
            array = array[None, ...]
        tensor = torch.as_tensor(array, dtype=torch.float32)
    else:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be at least 1")
        generator = torch.Generator().manual_seed(args.seed)
        tensor = torch.randn(args.batch_size, 200, 36, generator=generator) if args.random else torch.zeros(args.batch_size, 200, 36)
    if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (200, 36):
        raise ValueError(f"Expected input shape (B, 200, 36), received {tuple(tensor.shape)}")
    if not torch.isfinite(tensor).all():
        raise ValueError("Input contains NaN or infinity")
    return tensor


def main() -> None:
    args = parse_args()
    inputs = make_input(args)
    checkpoint, digest = load_verified(args.checkpoint.expanduser().resolve())
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    with torch.inference_mode():
        outputs = model(inputs)

    print(f"checkpoint_sha256: {digest}")
    print(f"input_shape: {tuple(inputs.shape)}")
    for task, output in outputs.items():
        values = output.cpu()
        if task in CLASS_TASKS:
            probabilities = torch.softmax(values, dim=-1)
            print(f"{task}: logits={values.tolist()} probabilities={probabilities.tolist()} predicted_class={probabilities.argmax(dim=-1).tolist()}")
        else:
            print(f"{task}: normalized_raw_output={values.squeeze(-1).tolist()}")


if __name__ == "__main__":
    main()
