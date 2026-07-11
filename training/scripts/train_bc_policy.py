"""Train BC policy (MSE) matching PPO actor architecture [128,128,64]."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn


class BCActor(nn.Module):
    def __init__(self, obs_dim: int = 18, act_dim: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, act_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--obs_dim", type=int, default=0, help="0 = infer from dataset")
    args = parser.parse_args()

    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    obs = data["obs"].float()
    action = data["action"].float()
    obs_dim = args.obs_dim if args.obs_dim > 0 else int(data.get("obs_dim", obs.shape[1]))
    n = obs.shape[0]
    split = int(n * 0.8)
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:split], perm[split:]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BCActor(obs_dim=obs_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        model.train()
        idx = train_idx[torch.randperm(len(train_idx))]
        total_loss = 0.0
        batches = 0
        for start in range(0, len(idx), args.batch_size):
            batch = idx[start : start + args.batch_size]
            pred = model(obs[batch].to(device))
            loss = loss_fn(pred, action[batch].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        model.eval()
        with torch.no_grad():
            val_pred = model(obs[val_idx].to(device))
            val_loss = float(loss_fn(val_pred, action[val_idx].to(device)).cpu())
        print(f"[BC Train] epoch={epoch+1:03d} train_mse={total_loss/max(batches,1):.6f} val_mse={val_loss:.6f}", flush=True)

    out = Path(args.output) if args.output else Path(args.dataset).parent / "bc_policy.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "obs_dim": obs_dim, "act_dim": 2}, out)
    print(f"[BC Train] Saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
