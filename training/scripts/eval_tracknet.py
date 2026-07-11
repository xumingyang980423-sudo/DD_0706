"""Offline TrackNet validation on held-out IR shards."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn

from train_tracknet import TrackNet, _load_shard_cpu, _resolve_shards, _split_shards, _resolve_device, _frames_batch_to_gpu


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_shards", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_png", type=int, default=8, help="Save N sample overlays (0=skip)")
    parser.add_argument("--png_dir", default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    ckpt_path = Path(args.checkpoint)
    shard_paths, scale = _resolve_shards(dataset_path)
    _, val_paths = _split_shards(shard_paths, args.val_shards, args.seed)
    device = _resolve_device(args.device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = TrackNet().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"[TrackNet Eval] checkpoint={ckpt_path}", flush=True)
    print(f"[TrackNet Eval] val_shards={len(val_paths)} device={device}", flush=True)

    lock_correct = 0.0
    lock_total = 0.0
    uv_sq = 0.0
    uv_count = 0.0
    rate_sq = 0.0
    rate_count = 0.0
    saved_png = 0
    png_dir = Path(args.png_dir) if args.png_dir else dataset_path.parent / "eval_debug"
    if args.save_png > 0:
        png_dir.mkdir(parents=True, exist_ok=True)

    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None
        ImageDraw = None

    with torch.no_grad():
        for shard in val_paths:
            frames, track = _load_shard_cpu(shard)
            n = frames.shape[0]
            for start in range(0, n, args.batch_size):
                idx = torch.arange(start, min(start + args.batch_size, n))
                x = _frames_batch_to_gpu(frames, idx, device, scale)
                tgt = track[idx].to(device)
                pred = model(x)

                lock_correct += float(((pred[:, 0] > 0.0) == (tgt[:, 0] > 0.5)).float().sum().cpu())
                lock_total += float(len(idx))
                mask = tgt[:, 0] > 0.5
                if mask.any():
                    uv_err = (pred[mask, 1:3] - tgt[mask, 1:3]) ** 2
                    uv_sq += float(uv_err.sum().cpu())
                    uv_count += float(mask.sum().cpu()) * 2.0
                    rate_err = (pred[mask, 3:5] - tgt[mask, 3:5]) ** 2
                    rate_sq += float(rate_err.sum().cpu())
                    rate_count += float(mask.sum().cpu()) * 2.0

                if args.save_png > 0 and Image is not None and saved_png < args.save_png:
                    for j in range(min(len(idx), args.save_png - saved_png)):
                        if tgt[j, 0] <= 0.5:
                            continue
                        fidx = int(idx[j].item())
                        frame_u8 = frames[fidx, 0].numpy() if frames.ndim == 4 else frames[fidx].numpy()
                        img = Image.fromarray(frame_u8, mode="L").convert("RGB")
                        draw = ImageDraw.Draw(img)
                        res = frame_u8.shape[0]
                        gt_u, gt_v = float(tgt[j, 1].cpu()), float(tgt[j, 2].cpu())
                        pr_u, pr_v = float(pred[j, 1].cpu()), float(pred[j, 2].cpu())
                        for u, v, color in ((gt_u, gt_v, (0, 255, 0)), (pr_u, pr_v, (255, 64, 64))):
                            px = int((u + 1.0) * 0.5 * (res - 1))
                            py = int((1.0 - (v + 1.0) * 0.5) * (res - 1))
                            r = 3
                            draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=2)
                        out = png_dir / f"eval_{saved_png:02d}_gt{gt_u:+.2f}_{gt_v:+.2f}_pr{pr_u:+.2f}_{pr_v:+.2f}.png"
                        img.save(out)
                        saved_png += 1

    lock_acc = lock_correct / lock_total if lock_total else 0.0
    uv_rmse = math.sqrt(uv_sq / uv_count) if uv_count else 0.0
    rate_rmse = math.sqrt(rate_sq / rate_count) if rate_count else 0.0
    gate_uv = uv_rmse < 0.05
    gate_lock = lock_acc > 0.95
    gate = gate_uv and gate_lock

    print(
        f"[TrackNet Eval] lock_acc={lock_acc:.4f} uv_rmse={uv_rmse:.4f} rate_rmse={rate_rmse:.4f} "
        f"gate={gate} (need lock>0.95 uv<0.05)",
        flush=True,
    )
    if saved_png:
        print(f"[TrackNet Eval] saved {saved_png} overlay PNG(s) -> {png_dir}", flush=True)
        print("[TrackNet Eval] green=GT centroid  red=pred centroid", flush=True)


if __name__ == "__main__":
    main()
