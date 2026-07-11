"""CPU reference IR seeker: geometric thermal image synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IRSeekerConfig:
    resolution: int = 64
    fov_deg: float = 4.0
    max_range_m: float = 8000.0
    blob_sigma_px: float = 2.2
    noise_std: float = 0.05
    body_intensity: float = 0.35
    plume_intensity: float = 1.0


def unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    fallback = np.array([1.0, 0.0, 0.0]) if fallback is None else fallback
    n = float(np.linalg.norm(vec))
    if n < 1.0e-9:
        return fallback
    return vec / n


def seeker_basis(missile_vel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = unit(missile_vel)
    world_up = np.array([0.0, 0.0, 1.0])
    right = unit(np.cross(forward, world_up), np.array([0.0, -1.0, 0.0]))
    up = unit(np.cross(right, forward), world_up)
    return forward, right, up


def aspect_tail_factor(missile_pos: np.ndarray, target_pos: np.ndarray, target_vel: np.ndarray) -> float:
    """1.0 = tail aspect (hot plume visible), 0.0 = head-on."""
    to_missile = unit(missile_pos - target_pos)
    target_fwd = unit(target_vel)
    return float(np.clip(np.dot(target_fwd, to_missile), 0.0, 1.0))


def project_los_to_uv(
    missile_pos: np.ndarray,
    missile_vel: np.ndarray,
    target_pos: np.ndarray,
    cfg: IRSeekerConfig,
) -> tuple[float, float, bool]:
    forward, right, up = seeker_basis(missile_vel)
    rel = target_pos - missile_pos
    dist = float(np.linalg.norm(rel))
    if dist < 1.0e-6:
        return 0.0, 0.0, False
    los = rel / dist
    fx = float(np.dot(los, forward))
    fy = float(np.dot(los, right))
    fz = float(np.dot(los, up))
    half_fov = np.deg2rad(cfg.fov_deg * 0.5)
    in_fov = fx > np.cos(half_fov) and abs(fy) < np.tan(half_fov) and abs(fz) < np.tan(half_fov)
    locked = in_fov and dist <= cfg.max_range_m
    if not locked:
        return 0.0, 0.0, False
    az = np.arctan2(fy, fx)
    el = np.arctan2(fz, fx)
    u = float(np.clip(az / half_fov, -1.0, 1.0))
    v = float(np.clip(el / half_fov, -1.0, 1.0))
    return u, v, True


def render_ir_frame(
    missile_pos: np.ndarray,
    missile_vel: np.ndarray,
    target_pos: np.ndarray,
    target_vel: np.ndarray,
    cfg: IRSeekerConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, bool, float, float]:
    cfg = cfg or IRSeekerConfig()
    rng = rng or np.random.default_rng()
    u, v, locked = project_los_to_uv(missile_pos, missile_vel, target_pos, cfg)
    res = cfg.resolution
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, res, dtype=np.float32),
        np.linspace(-1.0, 1.0, res, dtype=np.float32),
        indexing="ij",
    )
    frame = np.zeros((res, res), dtype=np.float32)
    if locked:
        aspect = aspect_tail_factor(missile_pos, target_pos, target_vel)
        intensity = cfg.body_intensity + cfg.plume_intensity * (0.25 + 0.75 * aspect)
        dist2 = (xx - u) ** 2 + (yy - v) ** 2
        sigma = cfg.blob_sigma_px * (2.0 / cfg.resolution)
        frame = intensity * np.exp(-dist2 / (2.0 * sigma * sigma))
    if cfg.noise_std > 0.0:
        frame = frame + rng.normal(0.0, cfg.noise_std, size=frame.shape).astype(np.float32)
    frame = np.clip(frame, 0.0, 1.0)
    return frame, locked, u, v
