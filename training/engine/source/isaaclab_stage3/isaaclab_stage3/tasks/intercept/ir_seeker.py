"""GPU-batched IR seeker image synthesis for Stage3 intercept env."""

from __future__ import annotations

import math
import os

import torch


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class TorchIRSeeker:
    """Render IR frames and GT track labels from missile/target kinematics."""

    def __init__(self, num_envs: int, device: torch.device | str) -> None:
        self.num_envs = num_envs
        self.device = device
        self.res = int(_float_env("STAGE3_IR_RES", 64))
        self.fov_deg = _float_env("STAGE3_IR_FOV_DEG", 4.0)
        self.max_range = _float_env("STAGE3_IR_MAX_RANGE", 8000.0)
        self.noise_std = _float_env("STAGE3_IR_NOISE", 0.05)
        self.blob_sigma = _float_env("STAGE3_IR_BLOB_SIGMA", 2.2)
        self.body_intensity = 0.35
        self.plume_intensity = 1.0

        lin = torch.linspace(-1.0, 1.0, self.res, device=self.device)
        yy, xx = torch.meshgrid(lin, lin, indexing="ij")
        self.grid_x = xx.unsqueeze(0)
        self.grid_y = yy.unsqueeze(0)

        self.prev_u = torch.zeros(num_envs, device=self.device)
        self.prev_v = torch.zeros(num_envs, device=self.device)
        self.prev_locked = torch.zeros(num_envs, dtype=torch.bool, device=self.device)

        self.frame = torch.zeros(num_envs, self.res, self.res, device=self.device)
        self.locked = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.u = torch.zeros(num_envs, device=self.device)
        self.v = torch.zeros(num_envs, device=self.device)
        self.u_dot = torch.zeros(num_envs, device=self.device)
        self.v_dot = torch.zeros(num_envs, device=self.device)
        self.confidence = torch.zeros(num_envs, device=self.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        self.prev_u[env_ids] = 0.0
        self.prev_v[env_ids] = 0.0
        self.prev_locked[env_ids] = False
        self.u_dot[env_ids] = 0.0
        self.v_dot[env_ids] = 0.0

    @staticmethod
    def _unit(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(vec, dim=1, keepdim=True)
        return torch.where(norm > 1.0e-6, vec / norm.clamp_min(1.0e-6), fallback)

    def update(
        self,
        missile_pos: torch.Tensor,
        missile_vel: torch.Tensor,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor,
        dt: float,
    ) -> None:
        forward = self._unit(missile_vel, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(missile_vel))
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(missile_vel)
        right = self._unit(torch.cross(forward, world_up, dim=1), torch.tensor([0.0, -1.0, 0.0], device=self.device).expand_as(missile_vel))
        up = self._unit(torch.cross(right, forward, dim=1), world_up)

        rel = target_pos - missile_pos
        dist = torch.linalg.norm(rel, dim=1).clamp_min(1.0e-6)
        los = rel / dist.unsqueeze(1)
        fx = torch.sum(los * forward, dim=1)
        fy = torch.sum(los * right, dim=1)
        fz = torch.sum(los * up, dim=1)

        half_fov = math.radians(self.fov_deg * 0.5)
        tan_fov = math.tan(half_fov)
        cos_fov = math.cos(half_fov)
        in_fov = (fx > cos_fov) & (fy.abs() < tan_fov) & (fz.abs() < tan_fov)
        locked = in_fov & (dist <= self.max_range)

        az = torch.atan2(fy, fx)
        el = torch.atan2(fz, fx)
        u = (az / half_fov).clamp(-1.0, 1.0)
        v = (el / half_fov).clamp(-1.0, 1.0)

        to_missile = self._unit(missile_pos - target_pos, forward)
        target_fwd = self._unit(target_vel, forward)
        aspect = torch.clamp(torch.sum(target_fwd * to_missile, dim=1), 0.0, 1.0)
        intensity = self.body_intensity + self.plume_intensity * (0.25 + 0.75 * aspect)

        sigma = self.blob_sigma * (2.0 / self.res)
        du = self.grid_x - u.view(-1, 1, 1)
        dv = self.grid_y - v.view(-1, 1, 1)
        dist2 = du * du + dv * dv
        frame = intensity.view(-1, 1, 1) * torch.exp(-dist2 / (2.0 * sigma * sigma))
        frame = torch.where(locked.view(-1, 1, 1), frame, torch.zeros_like(frame))
        if self.noise_std > 0.0:
            frame = frame + torch.randn_like(frame) * self.noise_std
        frame = frame.clamp(0.0, 1.0)

        inv_dt = 1.0 / max(dt, 1.0e-6)
        u_dot = torch.where(self.prev_locked, (u - self.prev_u) * inv_dt, torch.zeros_like(u))
        v_dot = torch.where(self.prev_locked, (v - self.prev_v) * inv_dt, torch.zeros_like(v))
        confidence = locked.float() * (0.5 + 0.5 * aspect)

        self.frame = frame
        self.locked = locked
        self.u = torch.where(locked, u, torch.zeros_like(u))
        self.v = torch.where(locked, v, torch.zeros_like(v))
        self.u_dot = torch.where(locked, u_dot, torch.zeros_like(u_dot))
        self.v_dot = torch.where(locked, v_dot, torch.zeros_like(v_dot))
        self.confidence = confidence

        self.prev_u = self.u.detach().clone()
        self.prev_v = self.v.detach().clone()
        self.prev_locked = locked.detach().clone()

    def track_state(self) -> torch.Tensor:
        """GT track vector for TrackNet labels / debug: [locked,u,v,u_dot,v_dot,confidence]."""
        return torch.stack(
            (self.locked.float(), self.u, self.v, self.u_dot, self.v_dot, self.confidence),
            dim=1,
        )
