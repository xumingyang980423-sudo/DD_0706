"""Train TrackNet: IR frame -> track state (supports sharded datasets)."""

from __future__ import annotations

import argparse
import gc
import math
import random
from pathlib import Path

import torch
import torch.nn as nn


class TrackNet(nn.Module):
    """Spatial CNN head — keeps 4x4 feature map for centroid regression (no global avg pool)."""

    def __init__(self, in_ch: int = 1, spatial: int = 4, channels: int = 128) -> None:
        super().__init__()
        self.spatial = spatial
        self.channels = channels
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        flat_dim = channels * spatial * spatial
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


def _resolve_shards(dataset_path: Path) -> tuple[list[Path], float]:
    meta = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if isinstance(meta, dict) and meta.get("shards"):
        base = dataset_path.parent
        scale = float(meta.get("frame_scale", 255.0))
        return [base / s for s in meta["shards"]], scale
    return [dataset_path], 1.0


def _split_shards(shard_paths: list[Path], val_shards: int, seed: int) -> tuple[list[Path], list[Path]]:
    if len(shard_paths) <= 1:
        return shard_paths, shard_paths
    n_val = max(1, min(val_shards, len(shard_paths) // 5))
    order = list(shard_paths)
    rng = random.Random(seed)
    rng.shuffle(order)
    val_set = set(order[:n_val])
    train = [p for p in shard_paths if p not in val_set]
    val = [p for p in shard_paths if p in val_set]
    return train, val


def _resolve_device(requested: str) -> torch.device:
    req = requested.strip().lower()
    if req == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dev = torch.device(requested)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA requested ({requested}) but torch.cuda.is_available() is False. "
            "Use Isaac Sim python.bat or install CUDA PyTorch."
        )
    return dev


def _load_shard_cpu(shard_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one shard on CPU. Frames stay uint8 to save RAM; float conversion happens on GPU."""
    data = torch.load(shard_path, map_location="cpu", weights_only=False)
    frames = data["frame"]
    if frames.dtype != torch.uint8:
        frames = (frames.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    if frames.ndim == 3:
        frames = frames.unsqueeze(1)
    track = data["track"].float()
    return frames, track


def _frames_batch_to_gpu(frames: torch.Tensor, idx: torch.Tensor, device: torch.device, scale: float) -> torch.Tensor:
    batch = frames[idx]
    return batch.to(device, non_blocking=True).float() / scale


def _compute_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    bce: nn.Module,
    uv_weight: float,
    rate_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    locked_tgt = tgt[:, 0:1]
    locked_pred = pred[:, 0:1]
    lock_loss = bce(locked_pred, locked_tgt)

    mask = tgt[:, 0] > 0.5
    zero = pred.sum() * 0.0
    if mask.any():
        uv_loss = nn.functional.mse_loss(pred[mask, 1:3], tgt[mask, 1:3])
        rate_loss = nn.functional.mse_loss(pred[mask, 3:5], tgt[mask, 3:5])
        conf_loss = nn.functional.mse_loss(pred[mask, 5:6], tgt[mask, 5:6])
    else:
        uv_loss = zero
        rate_loss = zero
        conf_loss = zero

    loss = 0.05 * lock_loss + uv_weight * uv_loss + rate_weight * rate_loss + 0.1 * conf_loss
    parts = {
        "lock": float(lock_loss.detach().cpu()),
        "uv": float(uv_loss.detach().cpu()) if mask.any() else 0.0,
        "rate": float(rate_loss.detach().cpu()) if mask.any() else 0.0,
    }
    return loss, parts


def _run_epoch(
    model: TrackNet,
    opt: torch.optim.Optimizer | None,
    bce: nn.Module,
    shard_paths: list[Path],
    shard_cache: list[tuple[torch.Tensor, torch.Tensor]] | None,
    device: torch.device,
    batch_size: int,
    scale: float,
    train: bool,
    uv_weight: float,
    rate_weight: float,
    use_amp: bool,
) -> tuple[float, float, float, float]:
    if train:
        model.train()
    else:
        model.eval()

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total_loss = 0.0
    total_uv = 0.0
    batches = 0
    lock_correct = 0.0
    lock_total = 0.0
    uv_sq = 0.0
    uv_count = 0.0

    def _process_shard(frames: torch.Tensor, track: torch.Tensor) -> None:
        nonlocal total_loss, total_uv, batches, lock_correct, lock_total, uv_sq, uv_count
        n = frames.shape[0]
        perm = torch.randperm(n) if train else torch.arange(n)

        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x = _frames_batch_to_gpu(frames, idx, device, scale)
            tgt = track[idx].to(device, non_blocking=True)

            if train:
                assert opt is not None
                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(x)
                    loss, parts = _compute_loss(pred, tgt, bce, uv_weight, rate_weight)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        pred = model(x)
                        loss, parts = _compute_loss(pred, tgt, bce, uv_weight, rate_weight)

            locked_pred = pred[:, 0:1].detach()
            locked_tgt = tgt[:, 0:1]
            lock_correct += float(((locked_pred > 0.0) == (locked_tgt > 0.5)).float().sum().cpu())
            lock_total += float(len(idx))
            mask = tgt[:, 0] > 0.5
            if mask.any():
                err = (pred[mask, 1:3].detach() - tgt[mask, 1:3]) ** 2
                uv_sq += float(err.sum().cpu())
                uv_count += float(mask.sum().cpu()) * 2.0

            total_loss += float(loss.detach().cpu())
            total_uv += parts["uv"]
            batches += 1

    if shard_cache is not None:
        for frames, track in shard_cache:
            _process_shard(frames, track)
    else:
        order = list(shard_paths)
        if train:
            random.shuffle(order)
        for shard in order:
            frames, track = _load_shard_cpu(shard)
            _process_shard(frames, track)
            del frames, track
            gc.collect()

    lock_acc = (lock_correct / lock_total) if lock_total else 0.0
    uv_rmse = (uv_sq / uv_count) ** 0.5 if uv_count else 0.0
    return total_loss / max(batches, 1), lock_acc, uv_rmse, total_uv / max(batches, 1)


def _preload_shards(shard_paths: list[Path]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    cache: list[tuple[torch.Tensor, torch.Tensor]] = []
    total = 0
    for i, shard in enumerate(shard_paths):
        frames, track = _load_shard_cpu(shard)
        cache.append((frames, track))
        total += frames.shape[0]
        print(f"[TrackNet] preload {i+1}/{len(shard_paths)} {shard.name} samples={frames.shape[0]}", flush=True)
    print(f"[TrackNet] preloaded {total} samples (uint8 on CPU, ~{total * 4096 / 1e9:.1f} GB frames)", flush=True)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=0, help="0 = auto (1024 on GPU, 256 on CPU)")
    parser.add_argument("--device", default="cuda:0", help="cuda:0 / cpu / auto")
    parser.add_argument("--preload", action="store_true", help="Cache all shards in CPU RAM (faster, uses ~4GB+)")
    parser.add_argument("--no_amp", action="store_true", help="Disable mixed precision on GPU")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--uv_weight", type=float, default=20.0, help="Weight for u,v centroid MSE")
    parser.add_argument("--rate_weight", type=float, default=2.0, help="Weight for u_dot,v_dot MSE")
    parser.add_argument("--val_shards", type=int, default=3, help="Random held-out shards for validation")
    parser.add_argument("--patience", type=int, default=12, help="Early stop if val_uv_rmse does not improve")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default="", help="Optional checkpoint to resume (same architecture)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    shard_paths, scale = _resolve_shards(dataset_path)
    train_shards, val_shards = _split_shards(shard_paths, args.val_shards, args.seed)

    device = _resolve_device(args.device)
    batch_size = args.batch_size if args.batch_size > 0 else (1024 if device.type == "cuda" else 256)
    use_amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    data_mode = "preload_cpu_uint8" if args.preload else "stream_shards"
    print(
        f"[TrackNet] shards={len(shard_paths)} train={len(train_shards)} val={len(val_shards)} "
        f"scale={scale} uv_weight={args.uv_weight} device={device} batch_size={batch_size} "
        f"amp={use_amp} data_mode={data_mode}",
        flush=True,
    )
    if device.type == "cuda":
        print(f"[TrackNet] GPU: {torch.cuda.get_device_name(device)}", flush=True)
        print("[TrackNet] Frames convert uint8->float on GPU per batch (low CPU RAM by default)", flush=True)
    else:
        print("[TrackNet] WARNING: running on CPU", flush=True)

    train_cache = _preload_shards(train_shards) if args.preload else None
    val_cache = _preload_shards(val_shards) if args.preload else None

    model = TrackNet().to(device)
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[TrackNet] Resumed weights from {args.resume}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    bce = nn.BCEWithLogitsLoss()

    out = Path(args.output) if args.output else dataset_path.parent / "tracknet.pth"
    best_path = out.with_name(out.stem + "_best.pth")
    out.parent.mkdir(parents=True, exist_ok=True)

    best_uv = math.inf
    stale = 0
    for epoch in range(args.epochs):
        train_loss, _, train_uv_rmse, _ = _run_epoch(
            model, opt, bce, train_shards, train_cache, device, batch_size, scale, True,
            args.uv_weight, args.rate_weight, use_amp,
        )
        val_loss, lock_acc, val_uv_rmse, _ = _run_epoch(
            model, None, bce, val_shards, val_cache, device, batch_size, scale, False,
            args.uv_weight, args.rate_weight, use_amp,
        )
        scheduler.step()
        improved = val_uv_rmse < best_uv - 1.0e-5
        if improved:
            best_uv = val_uv_rmse
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "in_ch": 1,
                    "out_dim": 6,
                    "val_uv_rmse": val_uv_rmse,
                    "val_lock_acc": lock_acc,
                    "arch": "spatial_v2",
                },
                best_path,
            )
        else:
            stale += 1

        print(
            f"[TrackNet] epoch={epoch+1:03d} train_loss={train_loss:.5f} train_uv_rmse={train_uv_rmse:.4f} "
            f"val_loss={val_loss:.5f} val_lock_acc={lock_acc:.3f} val_uv_rmse={val_uv_rmse:.4f} "
            f"best_uv={best_uv:.4f} lr={scheduler.get_last_lr()[0]:.2e}"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"[TrackNet] Early stop at epoch {epoch+1} (patience={args.patience})", flush=True)
            break

    if best_path.exists():
        best = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(best["model_state_dict"])
        print(f"[TrackNet] Loaded best checkpoint val_uv_rmse={best.get('val_uv_rmse', best_uv):.4f}", flush=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "in_ch": 1,
            "out_dim": 6,
            "val_uv_rmse": best_uv,
            "arch": "spatial_v2",
        },
        out,
    )
    print(f"[TrackNet] Saved -> {out}", flush=True)
    print(f"[TrackNet] Best -> {best_path}", flush=True)


if __name__ == "__main__":
    main()
