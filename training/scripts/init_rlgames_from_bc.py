"""Initialize RL-Games PPO checkpoint actor weights from BC policy."""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch

# RL-Games Adam param order for continuous_a2c (trainable weights only).
_TRAINABLE_KEYS = [
    "a2c_network.sigma",
    "a2c_network.actor_mlp.0.weight",
    "a2c_network.actor_mlp.0.bias",
    "a2c_network.actor_mlp.2.weight",
    "a2c_network.actor_mlp.2.bias",
    "a2c_network.actor_mlp.4.weight",
    "a2c_network.actor_mlp.4.bias",
    "a2c_network.value.weight",
    "a2c_network.value.bias",
    "a2c_network.mu.weight",
    "a2c_network.mu.bias",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_checkpoint", required=True)
    parser.add_argument("--rl_checkpoint", default="", help="Optional existing RL checkpoint to patch")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bc = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=False)
    bc_sd = bc["model_state_dict"]

    if args.rl_checkpoint and Path(args.rl_checkpoint).exists():
        ckpt = torch.load(args.rl_checkpoint, map_location="cpu", weights_only=False)
    else:
        ckpt = {"model": {}, "optimizer": None, "epoch": 0}

    model = ckpt.get("model", ckpt)
    old_obs_dim = _infer_template_obs_dim(model)
    obs_dim = int(bc.get("obs_dim", bc_sd["net.0.weight"].shape[1]))

    mapping = [
        ("net.0.weight", "a2c_network.actor_mlp.0.weight"),
        ("net.0.bias", "a2c_network.actor_mlp.0.bias"),
        ("net.2.weight", "a2c_network.actor_mlp.2.weight"),
        ("net.2.bias", "a2c_network.actor_mlp.2.bias"),
        ("net.4.weight", "a2c_network.actor_mlp.4.weight"),
        ("net.4.bias", "a2c_network.actor_mlp.4.bias"),
        ("net.6.weight", "a2c_network.mu.weight"),
        ("net.6.bias", "a2c_network.mu.bias"),
    ]
    patched = 0
    for bc_key, rl_key in mapping:
        if bc_key in bc_sd and rl_key in model:
            model[rl_key] = bc_sd[bc_key].clone()
            patched += 1

    _reset_obs_normalizer(model, obs_dim)

    resized = 0
    if old_obs_dim is not None and old_obs_dim != obs_dim:
        resized = _resize_non_actor_input_layers(model, old_obs_dim, obs_dim)
        print(
            f"[BC Init] obs_dim {old_obs_dim} -> {obs_dim}: resized critic input layers={resized}",
            flush=True,
        )

    # BC init starts a new PPO run: rebuild Adam state to match patched model shapes.
    template_opt = ckpt.get("optimizer")
    ckpt["optimizer"] = _rebuild_optimizer_state(model, template_opt)
    ckpt["epoch"] = 0
    ckpt["frame"] = 0
    ckpt["last_mean_rewards"] = -1.0e9

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, out)
    print(
        f"[BC Init] Patched {patched} actor tensors, obs_dim={obs_dim}, rebuilt optimizer -> {out}",
        flush=True,
    )


def _infer_template_obs_dim(model: dict) -> int | None:
    for key in (
        "a2c_network.actor_mlp.0.weight",
        "a2c_network.critic_mlp.0.weight",
        "running_mean_std.running_mean",
    ):
        if key not in model:
            continue
        tensor = model[key]
        if key.endswith(".weight"):
            return int(tensor.shape[1])
        return int(tensor.shape[0])
    return None


def _resize_non_actor_input_layers(model: dict, old_dim: int, new_dim: int) -> int:
    """Re-init critic (and any non-actor) first layers that still expect old obs dim."""
    resized = 0
    for key, tensor in list(model.items()):
        if "actor_mlp" in key or "mu." in key:
            continue
        if not key.endswith(".weight") or tensor.ndim != 2:
            continue
        if int(tensor.shape[1]) != old_dim:
            continue
        hidden = int(tensor.shape[0])
        new_w = torch.empty(hidden, new_dim)
        torch.nn.init.kaiming_uniform_(new_w, a=math.sqrt(5))
        model[key] = new_w
        bias_key = key.replace(".weight", ".bias")
        if bias_key in model and model[bias_key].shape[0] == hidden:
            fan_in = new_dim
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(model[bias_key], -bound, bound)
        resized += 1
        print(f"[BC Init] resized {key}: [{hidden},{old_dim}] -> [{hidden},{new_dim}]", flush=True)
    return resized


def _rebuild_optimizer_state(model: dict, template_opt: dict | None) -> dict:
    """Fresh Adam moments with shapes matching the (possibly resized) model tensors."""
    state: dict[int, dict] = {}
    for i, key in enumerate(_TRAINABLE_KEYS):
        if key not in model:
            raise KeyError(f"Missing trainable key in RL model: {key}")
        tensor = model[key]
        state[i] = {
            "step": torch.tensor(0.0, dtype=torch.float32),
            "exp_avg": torch.zeros_like(tensor),
            "exp_avg_sq": torch.zeros_like(tensor),
        }
    if template_opt and isinstance(template_opt, dict) and "param_groups" in template_opt:
        param_groups = copy.deepcopy(template_opt["param_groups"])
    else:
        param_groups = [
            {
                "lr": 3e-4,
                "betas": (0.9, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "params": list(range(len(_TRAINABLE_KEYS))),
            }
        ]
    return {"state": state, "param_groups": param_groups}


def _reset_obs_normalizer(model: dict, obs_dim: int) -> None:
    """Reset RL-Games input normalizer to match current env obs dim (BC uses raw env obs)."""
    mean_key = "running_mean_std.running_mean"
    var_key = "running_mean_std.running_var"
    count_key = "running_mean_std.count"
    if mean_key not in model:
        return
    old_dim = int(model[mean_key].shape[0])
    if old_dim != obs_dim:
        print(f"[BC Init] obs normalizer {old_dim} -> {obs_dim}", flush=True)
    model[mean_key] = torch.zeros(obs_dim)
    model[var_key] = torch.ones(obs_dim)
    model[count_key] = torch.tensor(1.0)


if __name__ == "__main__":
    main()
