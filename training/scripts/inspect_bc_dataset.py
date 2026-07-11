"""Inspect BC dataset transitions.pt (no Isaac Sim required)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()

    path = Path(args.dataset)
    data = torch.load(path, map_location="cpu", weights_only=False)
    obs = data["obs"].float()
    act = data["action"].float()
    sid = data.get("scenario_id")

    print(f"[BC Inspect] file={path}")
    print(f"[BC Inspect] curriculum={data.get('curriculum', '?')}")
    print(f"[BC Inspect] transitions={obs.shape[0]} obs_dim={obs.shape[1]} act_dim={act.shape[1]}")
    print(
        f"[BC Inspect] obs  min={obs.min():.3f} max={obs.max():.3f} mean={obs.mean():.3f} "
        f"std={obs.std():.3f}"
    )
    print(
        f"[BC Inspect] act  min={act.min():.3f} max={act.max():.3f} mean={act.mean():.3f} "
        f"std={act.std():.3f}"
    )
    if sid is not None:
        sid = sid.long()
        counts = torch.bincount(sid, minlength=int(sid.max().item()) + 1)
        covered = int((counts > 0).sum().item())
        print(f"[BC Inspect] scenario_ids covered={covered} (sid 0..{int(sid.max())})")
        top = sorted([(i, int(c)) for i, c in enumerate(counts.tolist()) if c > 0], key=lambda x: -x[1])[:8]
        print(f"[BC Inspect] top scenarios (sid,count): {top}")
    print("[BC Inspect] OK — this is a dataset. Train BC to get bc_policy.pth, then rollout-verify.")


if __name__ == "__main__":
    main()
