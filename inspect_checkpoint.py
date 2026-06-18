#!/usr/bin/env python3

"""Inspect a DINO-WM checkpoint and print training metadata.

This utility is intentionally lightweight: it loads a checkpoint, prints the
saved epoch and key names, then dumps any stored PyTorch modules so you can see
the model architecture. If a nearby Hydra config exists, it also prints the
most relevant training fields.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover - optional dependency guard
    OmegaConf = None


def resolve_checkpoint_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def count_parameters(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def summarize_value(value: Any) -> str:
    if isinstance(value, torch.nn.Module):
        return f"{value.__class__.__name__} ({count_parameters(value):,} params)"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if isinstance(value, (list, tuple, set)):
        return f"{type(value).__name__}[{len(value)}]"
    return type(value).__name__


def print_module(name: str, module: Any) -> None:
    print(f"\n[{name}]")
    print(f"Type: {summarize_value(module)}")
    if isinstance(module, torch.nn.Module):
        print(module)
    elif isinstance(module, dict):
        keys = list(module.keys())
        preview = ", ".join(map(str, keys[:12]))
        if len(keys) > 12:
            preview += ", ..."
        print(f"State dict keys ({len(keys)}): {preview}")
    else:
        print(repr(module))


def find_hydra_config(checkpoint_path: Path) -> Path | None:
    candidates = [
        checkpoint_path.parent.parent / "hydra.yaml",
        checkpoint_path.parent.parent / ".hydra" / "hydra.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def print_hydra_summary(hydra_path: Path) -> None:
    if OmegaConf is None:
        print("\n[hydra]")
        print(f"Found config at {hydra_path}, but OmegaConf is unavailable.")
        return

    cfg = OmegaConf.load(hydra_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    print("\n[hydra]")
    print(f"Config path: {hydra_path}")
    for section_name in ("training", "env"):
        section = cfg_dict.get(section_name)
        if isinstance(section, dict):
            print(f"{section_name}:")
            for key in ("epochs", "batch_size", "seed", "name", "dataset"):
                if key in section:
                    value = section[key]
                    if isinstance(value, dict):
                        print(f"  {key}: dict[{len(value)}]")
                    else:
                        print(f"  {key}: {value}")

    for key in ("frameskip", "num_hist", "num_pred", "debug"):
        if key in cfg_dict:
            print(f"{key}: {cfg_dict[key]}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a DINO-WM checkpoint and print training metadata."
    )
    parser.add_argument("checkpoint", help="Path to a .pth checkpoint file")
    parser.add_argument(
        "--no-hydra",
        action="store_true",
        help="Do not try to load a nearby hydra.yaml file.",
    )
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)

    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint type: {type(checkpoint).__name__}")
    print(f"Top-level keys: {', '.join(sorted(map(str, checkpoint.keys())))}")

    epoch = checkpoint.get("epoch")
    if epoch is not None:
        print(f"Epoch: {epoch}")
    else:
        print("Epoch: not found")

    for key in (
        "encoder",
        "predictor",
        "decoder",
        "action_encoder",
        "proprio_encoder",
        "encoder_optimizer",
        "predictor_optimizer",
        "decoder_optimizer",
    ):
        if key in checkpoint:
            print_module(key, checkpoint[key])

    other_keys = [key for key in checkpoint.keys() if key not in {
        "epoch",
        "encoder",
        "predictor",
        "decoder",
        "action_encoder",
        "proprio_encoder",
        "encoder_optimizer",
        "predictor_optimizer",
        "decoder_optimizer",
    }]
    if other_keys:
        print("\n[other keys]")
        for key in other_keys:
            print(f"{key}: {summarize_value(checkpoint[key])}")

    if not args.no_hydra:
        hydra_path = find_hydra_config(checkpoint_path)
        if hydra_path is not None:
            print_hydra_summary(hydra_path)
        else:
            print("\n[hydra]")
            print("No nearby hydra.yaml found.")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())