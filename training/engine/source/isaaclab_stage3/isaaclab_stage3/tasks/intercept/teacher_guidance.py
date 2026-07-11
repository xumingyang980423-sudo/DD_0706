"""Batched teacher guidance for GPU intercept env (ported from intercept_core.py)."""

from __future__ import annotations

import torch

MISSILE_NOSE_OFFSET = 2.35
LONG_FOLLOW_MIN_SID = 14

GAIN_BY_SID = {
    14: 2.95, 15: 2.95, 16: 2.95, 17: 2.95,
    10: 3.2, 1: 3.2,
    11: 2.8, 4: 3.6, 5: 3.6, 12: 3.6, 13: 3.6,
}
DEFAULT_GAIN = 4.2

ALPHA_BY_SID = {
    14: 0.22, 15: 0.22, 16: 0.22, 17: 0.22,
    10: 0.30, 1: 0.30,
    11: 0.24,
}
DEFAULT_ALPHA = 0.34

TRAIL_START = {
    14: 48.0, 15: 54.0, 16: 58.0, 17: 56.0,
    18: 58.0, 19: 64.0, 20: 66.0, 21: 60.0, 22: 62.0, 23: 68.0,
}
TRAIL_SHRINK = {
    14: 2.20, 15: 1.65, 16: 1.70, 17: 1.75,
    18: 1.85, 19: 1.62, 20: 1.58, 21: 1.78, 22: 1.62, 23: 1.55,
}
PATH_GAIN = {
    14: 0.72, 15: 0.82, 16: 0.84, 17: 0.80,
    18: 0.86, 19: 0.78, 20: 0.80, 21: 0.88, 22: 0.82, 23: 0.86,
}


def _unit(vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(vec, dim=-1, keepdim=True)
    fb = fallback.unsqueeze(0).expand_as(vec) if fallback.ndim == 1 else fallback
    return torch.where(norm > 1e-6, vec / norm.clamp_min(1e-6), fb)


def _is_tail_chase(sid: torch.Tensor) -> torch.Tensor:
    return (sid == 1) | (sid == 8) | (sid == 10) | (sid == 11) | (sid >= LONG_FOLLOW_MIN_SID)


def _is_long_follow(sid: torch.Tensor) -> torch.Tensor:
    return sid >= LONG_FOLLOW_MIN_SID


def aircraft_forward(aircraft_vel: torch.Tensor) -> torch.Tensor:
    return _unit(aircraft_vel, torch.tensor([1.0, 0.0, 0.0], device=aircraft_vel.device))


def mid_tail_point(aircraft_pos: torch.Tensor, aircraft_vel: torch.Tensor) -> torch.Tensor:
    fwd = aircraft_forward(aircraft_vel)
    return aircraft_pos - fwd * 2.2


def missile_contact_point(missile_pos: torch.Tensor, missile_vel: torch.Tensor) -> torch.Tensor:
    fwd = _unit(missile_vel, torch.tensor([1.0, 0.0, 0.0], device=missile_vel.device))
    return missile_pos + fwd * MISSILE_NOSE_OFFSET


def _lookup_sid(sid: torch.Tensor, table: dict[int, float], default: float, device) -> torch.Tensor:
    out = torch.full(sid.shape, default, device=device, dtype=torch.float32)
    for k, v in table.items():
        out = torch.where(sid == k, torch.full_like(out, v), out)
    return out


def guidance_aim_point(
    sid: torch.Tensor,
    missile_pos: torch.Tensor,
    missile_vel: torch.Tensor,
    aircraft_pos: torch.Tensor,
    aircraft_vel: torch.Tensor,
    lead_time: torch.Tensor,
    tail_offset: float,
    closeout_range: float,
) -> torch.Tensor:
    device = sid.device
    rel_vel = aircraft_vel - missile_vel
    mid_tail = mid_tail_point(aircraft_pos, aircraft_vel)
    contact = torch.linalg.norm(mid_tail - missile_contact_point(missile_pos, missile_vel), dim=1)
    long_mask = _is_long_follow(sid)

    trail_dist = (contact * 0.14).clamp(1.2, 8.0)
    lead_scale = ((contact - 5.0) / 45.0).clamp(0.0, 0.10)
    long_aim = mid_tail - aircraft_forward(aircraft_vel) * trail_dist.unsqueeze(1) + aircraft_vel * lead_time.unsqueeze(1) * lead_scale.unsqueeze(1)

    rel_lead = aircraft_pos + rel_vel * lead_time.unsqueeze(1) * 0.16
    tail_lead = mid_tail + aircraft_vel * lead_time.unsqueeze(1) * 0.08
    fighter = aircraft_pos + aircraft_vel * lead_time.unsqueeze(1) * 0.12
    maneuver = aircraft_pos + aircraft_vel * lead_time.unsqueeze(1) * 0.10
    evasion = aircraft_pos + rel_vel * lead_time.unsqueeze(1) * 0.08

    aim = rel_lead
    aim = torch.where(_is_tail_chase(sid).unsqueeze(1), tail_lead, aim)
    aim = torch.where((sid == 12).unsqueeze(1), fighter, aim)
    aim = torch.where((sid == 13).unsqueeze(1), maneuver, aim)
    aim = torch.where(((sid == 6) | (sid == 7)).unsqueeze(1), evasion, aim)
    aim = torch.where(long_mask.unsqueeze(1), long_aim, aim)

    tail_pt = aircraft_pos - aircraft_forward(aircraft_vel) * tail_offset
    closeout = ((closeout_range - contact) / closeout_range).clamp(0.0, 1.0)
    aim = aim * (1.0 - closeout.unsqueeze(1)) + tail_pt * closeout.unsqueeze(1)
    return aim


def baseline_lateral_action(
    sid: torch.Tensor,
    missile_pos: torch.Tensor,
    missile_vel: torch.Tensor,
    aircraft_pos: torch.Tensor,
    aircraft_vel: torch.Tensor,
    forward: torch.Tensor,
    right: torch.Tensor,
    up: torch.Tensor,
    last_action: torch.Tensor,
    tail_gain: float,
    tail_offset: float,
    action_alpha: float,
    closeout_range: float,
    tangent_blend_max: float,
) -> torch.Tensor:
    rel_pos = aircraft_pos - missile_pos
    speed = torch.linalg.norm(missile_vel, dim=1).clamp_min(1.0)
    lead_time = (torch.linalg.norm(rel_pos, dim=1) / speed).clamp(0.2, 2.0)
    aim = guidance_aim_point(sid, missile_pos, missile_vel, aircraft_pos, aircraft_vel, lead_time, tail_offset, closeout_range)
    desired_dir = _unit(aim - missile_pos, forward)

    contact = torch.linalg.norm(mid_tail_point(aircraft_pos, aircraft_vel) - missile_contact_point(missile_pos, missile_vel), dim=1)
    follow_blend = ((contact - 8.0) / 55.0).clamp(0.0, tangent_blend_max)
    ac_dir = aircraft_forward(aircraft_vel)
    long_mask = _is_long_follow(sid)
    blended = _unit(desired_dir * (1.0 - follow_blend.unsqueeze(1)) + ac_dir * follow_blend.unsqueeze(1), desired_dir)
    desired_dir = torch.where(long_mask.unsqueeze(1), blended, desired_dir)

    correction = desired_dir - forward * torch.sum(desired_dir * forward, dim=1, keepdim=True)
    gain_ref = max(DEFAULT_GAIN, 1.0e-6)
    gain = _lookup_sid(sid, GAIN_BY_SID, DEFAULT_GAIN, sid.device) * (tail_gain / gain_ref)
    raw = torch.stack(
        (
            (torch.sum(correction * right, dim=1) * gain).clamp(-1.0, 1.0),
            (torch.sum(correction * up, dim=1) * gain).clamp(-1.0, 1.0),
        ),
        dim=1,
    )
    alpha_ref = max(DEFAULT_ALPHA, 1.0e-6)
    alpha = _lookup_sid(sid, ALPHA_BY_SID, DEFAULT_ALPHA, sid.device) * (action_alpha / alpha_ref)
    alpha = alpha.unsqueeze(1).clamp(0.05, 1.0)
    smoothed = last_action * (1.0 - alpha) + raw * alpha
    return smoothed.clamp(-1.0, 1.0)


def long_follow_desired_velocity(
    sid: torch.Tensor,
    guidance_t: torch.Tensor,
    missile_pos: torch.Tensor,
    missile_vel: torch.Tensor,
    aircraft_pos: torch.Tensor,
    aircraft_vel: torch.Tensor,
    aircraft_history: torch.Tensor,
    history_len: torch.Tensor,
    guidance_speed: torch.Tensor,
    max_lateral_accel: torch.Tensor,
    step_dt: float,
    closeout_range: float,
) -> torch.Tensor:
    device = sid.device
    n = sid.shape[0]
    ac_fwd = aircraft_forward(aircraft_vel)
    mid_tail = mid_tail_point(aircraft_pos, aircraft_vel)
    contact = torch.linalg.norm(mid_tail - missile_contact_point(missile_pos, missile_vel), dim=1)

    trail_start = _lookup_sid(sid, TRAIL_START, 34.0, device)
    shrink = _lookup_sid(sid, TRAIL_SHRINK, 1.9, device)
    trail_dist = (trail_start - guidance_t * shrink).clamp_min(0.8)

    ac_speed = torch.linalg.norm(aircraft_vel, dim=1).clamp_min(1.0)
    delay = (trail_dist / ac_speed).clamp(0.25, 5.2)
    delay_steps = (delay / step_dt).long().clamp(0, aircraft_history.shape[1] - 1)

    idx = (history_len - 1 - delay_steps).clamp(0, aircraft_history.shape[1] - 1)
    batch_idx = torch.arange(n, device=device)
    follow_point = aircraft_history[batch_idx, idx]

    prev_idx = (idx - max(1, int(0.35 / step_dt))).clamp(0, aircraft_history.shape[1] - 1)
    trail_dir = _unit(follow_point - aircraft_history[batch_idx, prev_idx], ac_fwd)
    follow_point = follow_point - trail_dir * (trail_dist * 0.18).clamp(max=5.0)

    closeout = ((closeout_range - contact) / closeout_range).clamp(0.0, 1.0)
    tail_pt = mid_tail - ac_fwd * (trail_dist * 0.12).clamp(0.4, 3.0)
    desired_point = follow_point * (1.0 - closeout.unsqueeze(1)) + tail_pt * closeout.unsqueeze(1)
    to_follow = desired_point - missile_pos
    desired_tangent = trail_dir * (1.0 - closeout.unsqueeze(1)) + ac_fwd * closeout.unsqueeze(1)

    pg = _lookup_sid(sid, PATH_GAIN, 0.76, device)
    target_speed = (ac_speed + 2.0).clamp_min(guidance_speed)
    correction = (to_follow * pg.unsqueeze(1)).clamp(-target_speed.unsqueeze(1) * 0.86, target_speed.unsqueeze(1) * 0.86)
    desired_vel = desired_tangent * (ac_speed + 2.0).unsqueeze(1) + correction
    ds = torch.linalg.norm(desired_vel, dim=1, keepdim=True).clamp_min(1e-6)
    desired_vel = torch.where((ds > target_speed.unsqueeze(1)), desired_vel / ds * target_speed.unsqueeze(1), desired_vel)

    max_dv = max_lateral_accel * step_dt * 1.55
    delta = (desired_vel - missile_vel).clamp(-max_dv.unsqueeze(1), max_dv.unsqueeze(1))
    return missile_vel + delta


def residual_schedule(step: int) -> tuple[float, float]:
    if step < 50_000:
        return 1.0, 0.0
    if step < 150_000:
        return 0.8, 0.2
    if step < 300_000:
        return 0.5, 0.5
    if step < 500_000:
        return 0.2, 0.8
    return 0.0, 1.0


def reward_stage(step: int) -> str:
    if step < 100_000:
        return "A"
    if step < 300_000:
        return "B"
    return "C"
