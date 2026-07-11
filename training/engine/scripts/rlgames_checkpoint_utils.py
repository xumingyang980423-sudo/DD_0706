"""Shared RL-Games checkpoint helpers for Stage3 train/play wrappers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def checkpoint_fixed_sigma(checkpoint_path: Path) -> bool | None:
    """Return True for legacy fixed sigma, False for learnable, None if unknown."""
    if not checkpoint_path.is_file():
        return None
    payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = payload.get("model", payload)
    if not isinstance(state, dict):
        return None
    if "a2c_network.sigma" in state:
        return True
    if "a2c_network.sigma.weight" in state or "a2c_network.sigma.bias" in state:
        return False
    return None


def inject_sigma_hydra_override(tag: str = "Stage3RL") -> None:
    """Match Hydra fixed_sigma to checkpoint layout before RL-Games train/play starts."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    known, _ = parser.parse_known_args()
    if not known.checkpoint:
        return

    fixed_sigma = checkpoint_fixed_sigma(Path(known.checkpoint))
    if fixed_sigma is None:
        return

    override = f"agent.params.network.space.continuous.fixed_sigma={'True' if fixed_sigma else 'False'}"
    joined = " ".join(sys.argv)
    if "fixed_sigma" in joined:
        return
    sys.argv.append(override)
    print(f"[{tag}] Checkpoint sigma layout -> {override}", flush=True)
