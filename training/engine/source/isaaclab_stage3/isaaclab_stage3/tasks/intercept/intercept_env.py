from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade

from . import teacher_guidance as tg
from .ir_seeker import TorchIRSeeker
from .tracknet_model import TorchTrackNetRunner

FUSELAGE_CONTACT_RADIUS = 0.95
MISSILE_NOSE_OFFSET = 2.35
DEFAULT_BOOST_CLIMB_WEIGHT = 0.48


@configclass
class Stage3InterceptEnvCfg(DirectRLEnvCfg):
    decimation = 2
    episode_length_s = 45.0
    action_space = 2
    observation_space = 18
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1.0 / 60.0, render_interval=decimation, device="cuda:0")
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=8.0, replicate_physics=True, clone_in_fabric=True
    )

    max_time = 45.0
    hit_radius = 2.5
    center_hit_radius = 6.0
    boost_accel = 11.0
    default_boost_duration = 2.4
    default_guidance_speed = 46.0
    max_altitude = 220.0
    max_range = 620.0
    speed_hold_gain = 2.4
    console_log_interval_steps = 8
    visual_debug_max_envs = 16
    visual_trail_points = 720


class Stage3InterceptEnv(DirectRLEnv):
    """GPU-vectorized abstract guidance task for Isaac Lab/RSL-RL.

    This is an algorithm training environment, not a real weapon model.
    It mirrors the stage-2 two-phase structure: rule-based BOOST and
    policy-controlled GUIDANCE.
    """

    cfg: Stage3InterceptEnvCfg

    def __init__(self, cfg: Stage3InterceptEnvCfg, render_mode: str | None = None, **kwargs):
        self._obs_mode_value = self._obs_mode()
        if self._obs_mode_value == "ir_track":
            os.environ["STAGE3_IR_ENABLE"] = "1"
            cfg.observation_space = self._observation_dim("ir_track")
        super().__init__(cfg, render_mode, **kwargs)
        self.actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.applied_actions = torch.zeros(self.num_envs, 2, device=self.device)
        self.aircraft_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.aircraft_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.missile_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.missile_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.target_base_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.scenario_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.launch_delay = torch.zeros(self.num_envs, device=self.device)
        self.boost_duration = torch.full((self.num_envs,), self.cfg.default_boost_duration, device=self.device)
        self.guidance_speed = torch.full((self.num_envs,), self.cfg.default_guidance_speed, device=self.device)
        self.max_lateral_accel = torch.full((self.num_envs,), 85.0, device=self.device)
        self.aircraft_altitude = torch.full((self.num_envs,), 55.0, device=self.device)
        self.closest_distance = torch.full((self.num_envs,), 1.0e9, device=self.device)
        self.closest_center_distance = torch.full((self.num_envs,), 1.0e9, device=self.device)
        self.last_distance = torch.full((self.num_envs,), 1.0e9, device=self.device)
        self.prev_missile_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_missile_vel = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_aircraft_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.maneuver_amp_scale = torch.ones(self.num_envs, device=self.device)
        self.maneuver_freq_scale = torch.ones(self.num_envs, device=self.device)
        self.maneuver_phase = torch.zeros(self.num_envs, device=self.device)
        self.heading_bias = torch.zeros(self.num_envs, device=self.device)
        self.climb_scale = torch.ones(self.num_envs, device=self.device)
        self.boost_climb_weight = torch.full((self.num_envs,), DEFAULT_BOOST_CLIMB_WEIGHT, device=self.device)
        self.aircraft_turn_amplitude = torch.full((self.num_envs,), 2.2, device=self.device)
        self.aircraft_turn_rate = torch.full((self.num_envs,), 0.33, device=self.device)
        self.launch_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.last_teacher_action = torch.zeros(self.num_envs, 2, device=self.device)
        self._history_steps = 720
        self.aircraft_history = torch.zeros(self.num_envs, self._history_steps, 3, device=self.device)
        self.history_len = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_hit = torch.zeros(self.num_envs, device=self.device)
        # Snapshots for eval/logging after step() resets episode_hit.
        self._last_step_hit = torch.zeros(self.num_envs, device=self.device)
        self._last_episode_hit_snapshot = torch.zeros(self.num_envs, device=self.device)
        self._last_step_contact = torch.zeros(self.num_envs, device=self.device)
        self._last_step_center = torch.zeros(self.num_envs, device=self.device)
        self._last_step_closest = torch.zeros(self.num_envs, device=self.device)
        self._last_step_closest_center = torch.zeros(self.num_envs, device=self.device)
        self._last_step_ground = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_step_bounds = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._interval_hits = 0
        self._last_console_log_step = -1
        self._completed_ep_hits = 0
        self._completed_ep_total = 0
        self._visual_aircraft_trail: list[tuple[float, float, float]] = []
        self._visual_missile_trail: list[tuple[float, float, float]] = []
        self._ir_seeker = TorchIRSeeker(self.num_envs, self.device) if self._ir_enabled() else None
        self._tracknet_runner: TorchTrackNetRunner | None = None
        if self._obs_mode_value == "ir_track":
            ckpt = os.environ.get("STAGE3_TRACKNET_CKPT", "").strip()
            if not ckpt:
                raise RuntimeError("STAGE3_OBS_MODE=ir_track requires STAGE3_TRACKNET_CKPT")
            self._tracknet_runner = TorchTrackNetRunner(ckpt, self.device)
            if self._ir_seeker is None:
                raise RuntimeError("IR seeker must be enabled for ir_track observation mode")

    def _setup_scene(self):
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        light_cfg = sim_utils.DomeLightCfg(intensity=1800.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self._setup_policy_visuals()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)
        self.prev_aircraft_pos[:] = self.aircraft_pos
        self.prev_missile_vel[:] = self.missile_vel
        self.prev_missile_pos[:] = self.missile_pos
        self._update_aircraft()
        self._update_missile()
        self._update_ir_seeker()
        self._update_policy_visuals()

    def _apply_action(self) -> None:
        return None

    def _get_observations(self) -> dict:
        if self._obs_mode_value == "ir_track":
            return self._get_ir_track_observations()
        rel_pos = self.aircraft_pos - self.missile_pos
        rel_vel = self.aircraft_vel - self.missile_vel
        distance = torch.linalg.norm(rel_pos, dim=1, keepdim=True).clamp_min(1.0e-6)
        los_dir = rel_pos / distance
        closing_speed = -torch.sum(rel_vel * los_dir, dim=1, keepdim=True) / 90.0
        lateral_rel_vel = rel_vel - torch.sum(rel_vel * los_dir, dim=1, keepdim=True) * los_dir
        lateral_speed = torch.linalg.norm(lateral_rel_vel, dim=1, keepdim=True) / 90.0
        missile_dir = self._unit(self.missile_vel, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        phase_flag = (self._flight_t().unsqueeze(1) >= self.boost_duration.unsqueeze(1)).float()
        obs = torch.cat(
            (
                rel_pos / torch.tensor([250.0, 250.0, 120.0], device=self.device),
                rel_vel / 90.0,
                missile_dir,
                los_dir,
                distance / 300.0,
                closing_speed,
                lateral_speed,
                self.missile_pos[:, 2:3] / 120.0,
                self._time().unsqueeze(1) / self.cfg.max_time,
                phase_flag,
            ),
            dim=1,
        )
        return {"policy": obs}

    def _get_ir_track_observations(self) -> dict:
        assert self._ir_seeker is not None and self._tracknet_runner is not None
        pred = self._tracknet_runner.predict(self._ir_seeker.frame)
        locked = torch.sigmoid(pred[:, 0:1])
        phase_flag = (self._flight_t().unsqueeze(1) >= self.boost_duration.unsqueeze(1)).float()
        obs = torch.cat(
            (
                locked,
                pred[:, 1:3],
                pred[:, 3:5],
                pred[:, 5:6],
                phase_flag,
                self.missile_pos[:, 2:3] / 120.0,
                self._time().unsqueeze(1) / self.cfg.max_time,
            ),
            dim=1,
        )
        return {"policy": obs}

    def _evaluate_intercept(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return hit mask plus contact/center/scoring distances.

        Hit = nose-to-fuselage contact OR center-to-center within visual envelope.
        Center check uses both current and previous positions to reduce step tunneling.
        """
        contact_distance = self._contact_distance()
        center_distance = torch.linalg.norm(self.aircraft_pos - self.missile_pos, dim=1)
        prev_center_distance = torch.linalg.norm(self.prev_aircraft_pos - self.prev_missile_pos, dim=1)
        scoring_distance = torch.minimum(contact_distance, center_distance)

        contact_radius = self._effective_hit_radius()
        center_radius = torch.full((self.num_envs,), self.cfg.center_hit_radius, device=self.device)

        hit_contact = contact_distance <= contact_radius
        hit_center = (center_distance <= center_radius) | (prev_center_distance <= center_radius)
        hit = hit_contact | hit_center
        return hit, contact_distance, center_distance, prev_center_distance, scoring_distance

    def _get_rewards(self) -> torch.Tensor:
        hit, contact_distance, center_distance, _prev_center_distance, scoring_distance = self._evaluate_intercept()
        distance = contact_distance
        self.closest_distance = torch.minimum(self.closest_distance, scoring_distance)
        self.closest_center_distance = torch.minimum(self.closest_center_distance, center_distance)
        guidance = self._flight_t() >= self.boost_duration
        guidance_f = guidance.float()
        stage = self._active_reward_stage()
        tail_offset = self._float_env("STAGE3_REWARD_TAIL_OFFSET", 2.6)

        closing_reward = (self.last_distance - distance) * 1.0
        target_forward = self._unit(self.aircraft_vel, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        tail_aim_point = self.aircraft_pos - target_forward * tail_offset
        target_dir = self._unit(tail_aim_point - self.missile_pos, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        missile_dir = self._unit(self.missile_vel, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        heading_reward = torch.sum(target_dir * missile_dir, dim=1) * 0.03
        tail_error = torch.linalg.norm(tail_aim_point - self.missile_pos, dim=1)
        tail_reward = torch.exp(-tail_error / 45.0) * 0.04
        along_aircraft = torch.sum((self.missile_pos - self.aircraft_pos) * target_forward, dim=1)
        ahead_penalty = torch.clamp(along_aircraft - 2.0, min=0.0) * 0.004 * (distance < 45.0).float() * guidance_f
        rear_alignment_reward = torch.clamp(-along_aircraft / 35.0, 0.0, 1.0) * 0.02
        near_reward = (
            torch.exp(-distance / 30.0) * 0.08
            + (distance < 10.0).float() * 0.15
            + (distance < 5.0).float() * 0.35
            + (distance < 2.5).float() * 0.70
        ) * guidance_f
        control_penalty = torch.sum(self.actions * self.actions, dim=1) * 0.006 * guidance_f

        if hit.any():
            self._interval_hits += int(hit.sum().item())
            step = int(self.common_step_counter)
            hit_contact = (contact_distance <= self._effective_hit_radius())[hit]
            hit_center = (center_distance <= self.cfg.center_hit_radius)[hit] | (
                torch.linalg.norm(self.prev_aircraft_pos - self.prev_missile_pos, dim=1) <= self.cfg.center_hit_radius
            )[hit]
            kind = "contact" if hit_contact.any() and not hit_center.any() else (
                "center" if hit_center.any() and not hit_contact.any() else "both"
            )
            print(
                f"[Stage3Hit] step={step} n={int(hit.sum().item())} kind={kind} "
                f"contact={contact_distance[hit].mean().item():.2f} center={center_distance[hit].mean().item():.2f} "
                f"sid={self.scenario_id[hit].float().mean().item():.0f}",
                flush=True,
            )

        ground = self.missile_pos[:, 2] <= 0.0
        bounds = (self.missile_pos[:, 2] > self.cfg.max_altitude) | (
            torch.linalg.norm(self.missile_pos[:, :2], dim=1) > self.cfg.max_range
        )
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        if stage == "A":
            reward = closing_reward + hit.float() * 50.0 - ground.float() * 5.0 - timeout.float() * 2.0
        elif stage == "B":
            reward = (
                closing_reward
                + heading_reward
                + rear_alignment_reward
                + hit.float() * 50.0
                - ground.float() * 5.0
                - bounds.float() * 4.0
                - timeout.float() * 2.0
            )
        else:
            reward = (
                closing_reward * 0.08
                + heading_reward
                + tail_reward
                + rear_alignment_reward
                + near_reward
                - ahead_penalty
                - control_penalty
                - 0.002
                + hit.float() * 50.0
                - ground.float() * 5.0
                - bounds.float() * 4.0
                - timeout.float() * 2.0
            )

        self.episode_hit = torch.maximum(self.episode_hit, hit.float())
        self._last_step_hit = hit.detach().clone()
        self._last_episode_hit_snapshot = self.episode_hit.detach().clone()
        self._last_step_contact = contact_distance.detach().clone()
        self._last_step_center = center_distance.detach().clone()
        self._last_step_closest = self.closest_distance.detach().clone()
        self._last_step_closest_center = self.closest_center_distance.detach().clone()
        self._last_step_ground = ground.detach().clone()
        self._last_step_bounds = bounds.detach().clone()
        self.episode_return += reward
        self.last_distance = distance

        hit_rate = hit.float().mean()
        episode_hit_rate = self.episode_hit.mean()
        mean_closest = self.closest_distance.mean()
        mean_ahead = torch.clamp(along_aircraft, min=0.0).mean()
        selection_score = hit_rate * 100.0 - mean_closest * 0.5 - mean_ahead * 0.3

        self.extras["log"] = {
            "Metrics/hit_rate": hit_rate,
            "Metrics/episode_hit_rate": episode_hit_rate,
            "Metrics/selection_score": selection_score,
            "Metrics/mean_closest_distance": mean_closest,
            "Metrics/mean_distance": contact_distance.mean(),
            "Metrics/mean_center_distance": center_distance.mean(),
            "Metrics/mean_scoring_distance": scoring_distance.mean(),
            "Metrics/mean_step_reward": reward.mean(),
            "Metrics/mean_episode_return": self.episode_return.mean(),
            "Metrics/mean_ahead_distance": mean_ahead,
            "Metrics/ground_rate": ground.float().mean(),
            "Metrics/bounds_rate": bounds.float().mean(),
            "Metrics/timeout_rate": timeout.float().mean(),
            "Metrics/reward_stage": torch.tensor(float(ord(stage)), device=self.device),
        }
        self._print_training_progress(
            reward, contact_distance, center_distance, hit, ground, bounds, timeout, selection_score, episode_hit_rate
        )
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        hit, _contact_distance, _center_distance, _prev_center_distance, _scoring_distance = self._evaluate_intercept()
        ground = self.missile_pos[:, 2] <= 0.0
        bounds = (self.missile_pos[:, 2] > self.cfg.max_altitude) | (
            torch.linalg.norm(self.missile_pos[:, :2], dim=1) > self.cfg.max_range
        )
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = hit | ground | bounds
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, dtype=torch.long, device=self.device)
        count = len(env_ids)
        self._completed_ep_hits += int(self.episode_hit[env_ids].sum().item())
        self._completed_ep_total += count
        super()._reset_idx(env_ids)

        sid = self._sample_scenario_ids(count)
        fixed_sid = self._fixed_visual_scenario_id()
        if fixed_sid is not None:
            sid[:] = fixed_sid
        self.scenario_id[env_ids] = sid
        rand = torch.rand(count, 22, device=self.device)

        target_x = 170.0 + rand[:, 0] * 60.0
        target_y = -80.0 + rand[:, 1] * 160.0
        target_z = 42.0 + rand[:, 2] * 30.0
        target_vx = -13.0 + rand[:, 3] * 4.0
        target_vy = 1.0 + rand[:, 4] * 4.0
        launch_vx = 5.0 + rand[:, 19] * 5.0
        launch_vy = -3.0 + rand[:, 20] * 6.0

        self.launch_delay[env_ids] = 0.0
        self.boost_duration[env_ids] = self.cfg.default_boost_duration
        self.guidance_speed[env_ids] = self.cfg.default_guidance_speed
        self.max_lateral_accel[env_ids] = 85.0
        self.aircraft_turn_amplitude[env_ids] = 2.2
        self.aircraft_turn_rate[env_ids] = 0.33
        self.boost_climb_weight[env_ids] = DEFAULT_BOOST_CLIMB_WEIGHT

        self._apply_scenario_resets(env_ids, sid, target_x, target_y, target_z, target_vx, target_vy, launch_vx, launch_vy)
        target_x, target_y, target_z, target_vx, target_vy, launch_vx, launch_vy = self._apply_episode_randomization(
            env_ids, sid, rand, target_x, target_y, target_z, target_vx, target_vy, launch_vx, launch_vy
        )

        delayed = self.launch_delay[env_ids] > 0.0
        launch_vx = torch.where(delayed, torch.zeros_like(launch_vx), launch_vx)
        launch_vy = torch.where(delayed, torch.zeros_like(launch_vy), launch_vy)

        self.aircraft_pos[env_ids] = torch.stack((target_x, target_y, target_z), dim=1)
        self.target_base_vel[env_ids] = torch.stack((target_vx, target_vy, torch.zeros_like(target_vx)), dim=1)
        self.aircraft_vel[env_ids] = self.target_base_vel[env_ids]
        self.aircraft_altitude[env_ids] = target_z
        self.missile_pos[env_ids] = torch.tensor([0.0, 0.0, 2.0], device=self.device)
        launch_vz = torch.full_like(launch_vx, 13.0)
        self.launch_velocity[env_ids] = torch.stack((launch_vx, launch_vy, launch_vz), dim=1)
        init_vel = torch.where(
            delayed.unsqueeze(1),
            torch.zeros(3, device=self.device),
            self.launch_velocity[env_ids],
        )
        self.missile_vel[env_ids] = init_vel
        self.prev_missile_pos[env_ids] = self.missile_pos[env_ids]
        self.prev_missile_vel[env_ids] = init_vel
        self.prev_aircraft_pos[env_ids] = self.aircraft_pos[env_ids]
        self.closest_distance[env_ids] = torch.linalg.norm(self.aircraft_pos[env_ids] - self.missile_pos[env_ids], dim=1)
        self.closest_center_distance[env_ids] = self.closest_distance[env_ids]
        self.last_distance[env_ids] = self.closest_distance[env_ids]
        self.episode_return[env_ids] = 0.0
        self.episode_hit[env_ids] = 0.0
        self.last_teacher_action[env_ids] = 0.0
        self.aircraft_history[env_ids] = 0.0
        self.history_len[env_ids] = 0
        if self._ir_seeker is not None:
            self._ir_seeker.reset(env_ids)

    def _apply_scenario_resets(self, env_ids, sid, target_x, target_y, target_z, target_vx, target_vy, launch_vx, launch_vy):
        def mask(value: int):
            return sid == value

        m = mask(0)
        self.aircraft_turn_amplitude[env_ids[m]] = 1.0

        m = mask(1)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -50.0, 0.0, 58.0, 12.2, 0.0
        launch_vx[m], launch_vy[m] = 8.0, 0.0
        self.launch_delay[env_ids[m]] = 5.4
        self.guidance_speed[env_ids[m]] = 54.0
        self.aircraft_turn_amplitude[env_ids[m]] = 0.15

        m = mask(2)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = 78.0, -160.0, 54.0, -0.8, 14.5
        self.aircraft_turn_amplitude[env_ids[m]] = 0.8
        m = mask(3)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = 78.0, 160.0, 54.0, -0.8, -14.5
        self.aircraft_turn_amplitude[env_ids[m]] = 0.8
        m = mask(4)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = 195.0, -52.0, 42.0, -13.8, 3.2
        self.guidance_speed[env_ids[m]] = 56.0
        self.max_lateral_accel[env_ids[m]] = 76.0
        self.aircraft_turn_amplitude[env_ids[m]] = 1.6
        m = mask(5)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = 198.0, 58.0, 86.0, -13.5, -3.4
        self.guidance_speed[env_ids[m]] = 56.0
        self.max_lateral_accel[env_ids[m]] = 76.0
        self.aircraft_turn_amplitude[env_ids[m]] = 1.6
        m = mask(6)
        self.aircraft_turn_amplitude[env_ids[m]] = 0.0
        self.aircraft_turn_rate[env_ids[m]] = 0.65
        m = mask(7)
        self.aircraft_turn_amplitude[env_ids[m]] = 0.0
        m = mask(8)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = 28.0, 58.0, 59.0, 13.0, -4.2
        launch_vx[m], launch_vy[m] = 7.0, 1.4
        self.launch_delay[env_ids[m]] = 3.8
        self.aircraft_turn_amplitude[env_ids[m]] = 2.2
        m = mask(9)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -94.0, -78.0, 68.0, 22.0, 6.5
        self.guidance_speed[env_ids[m]] = 60.0
        self.max_lateral_accel[env_ids[m]] = 64.0
        self.aircraft_turn_amplitude[env_ids[m]] = 2.6
        self.aircraft_turn_rate[env_ids[m]] = 0.42
        m = mask(10)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -72.0, 46.0, 37.0, 14.5, -2.8
        self.boost_duration[env_ids[m]] = 1.55
        self.guidance_speed[env_ids[m]] = 58.0
        self.max_lateral_accel[env_ids[m]] = 54.0
        self.boost_climb_weight[env_ids[m]] = 0.22
        self.aircraft_turn_amplitude[env_ids[m]] = 0.4
        self.aircraft_turn_rate[env_ids[m]] = 0.28
        m = mask(11)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = 165.0, 95.0, 82.0, 10.2, -2.1
        self.guidance_speed[env_ids[m]] = 55.0
        self.boost_climb_weight[env_ids[m]] = 0.38
        self.aircraft_turn_amplitude[env_ids[m]] = 1.4
        self.aircraft_turn_rate[env_ids[m]] = 0.24
        m = mask(12)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -80.0, -36.0, 65.0, 20.5, 3.0
        self.boost_duration[env_ids[m]] = 1.8
        self.guidance_speed[env_ids[m]] = 68.0
        self.max_lateral_accel[env_ids[m]] = 96.0
        m = mask(13)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -104.0, 0.0, 62.0, 18.0, 0.0
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 62.0
        self.max_lateral_accel[env_ids[m]] = 88.0
        m = mask(14)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -164.0, -8.0, 74.0, 25.5, 0.4
        self.launch_delay[env_ids[m]] = 7.2
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 64.0
        self.max_lateral_accel[env_ids[m]] = 120.0
        self.boost_climb_weight[env_ids[m]] = 0.44
        m = mask(15)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -170.0, 34.0, 74.0, 25.5, -0.5
        self.launch_delay[env_ids[m]] = 7.5
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 66.0
        self.max_lateral_accel[env_ids[m]] = 128.0
        m = mask(16)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -166.0, -30.0, 64.0, 24.5, 2.0
        self.launch_delay[env_ids[m]] = 7.2
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 68.0
        self.max_lateral_accel[env_ids[m]] = 126.0
        m = mask(17)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -176.0, 0.0, 72.0, 26.5, 0.0
        self.launch_delay[env_ids[m]] = 8.0
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 66.0
        self.max_lateral_accel[env_ids[m]] = 116.0
        self.boost_climb_weight[env_ids[m]] = 0.46
        m = mask(18)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -170.0, -4.0, 60.0, 25.5, 0.0
        self.launch_delay[env_ids[m]] = 7.5
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 70.0
        self.max_lateral_accel[env_ids[m]] = 134.0
        self.boost_climb_weight[env_ids[m]] = 0.50
        m = mask(19)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -174.0, 8.0, 72.0, 26.0, 0.2
        self.launch_delay[env_ids[m]] = 7.7
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 70.0
        self.max_lateral_accel[env_ids[m]] = 134.0
        m = mask(20)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -168.0, -20.0, 56.0, 25.0, 2.0
        self.launch_delay[env_ids[m]] = 7.4
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 70.0
        self.max_lateral_accel[env_ids[m]] = 134.0
        self.boost_climb_weight[env_ids[m]] = 0.50
        m = mask(21)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -176.0, 32.0, 70.0, 26.5, -0.4
        self.launch_delay[env_ids[m]] = 7.8
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 70.0
        self.max_lateral_accel[env_ids[m]] = 134.0
        m = mask(22)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -178.0, 0.0, 70.0, 26.2, 0.0
        self.launch_delay[env_ids[m]] = 8.0
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 70.0
        self.max_lateral_accel[env_ids[m]] = 134.0
        m = mask(23)
        target_x[m], target_y[m], target_z[m], target_vx[m], target_vy[m] = -184.0, -10.0, 66.0, 27.0, 0.4
        self.launch_delay[env_ids[m]] = 8.2
        self.boost_duration[env_ids[m]] = 1.9
        self.guidance_speed[env_ids[m]] = 70.0
        self.max_lateral_accel[env_ids[m]] = 134.0
        self.boost_climb_weight[env_ids[m]] = 0.50

    def _apply_episode_randomization(
        self,
        env_ids: torch.Tensor,
        sid: torch.Tensor,
        rand: torch.Tensor,
        target_x: torch.Tensor,
        target_y: torch.Tensor,
        target_z: torch.Tensor,
        target_vx: torch.Tensor,
        target_vy: torch.Tensor,
        launch_vx: torch.Tensor,
        launch_vy: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mode = self._randomization_mode()
        curriculum = self._scenario_curriculum()
        speed_min, speed_max = self._aircraft_speed_range(mode)
        if mode == "stress":
            pos_jitter = (26.0, 30.0, 10.0)
            amp_range = (1.10, 1.55)
            freq_range = (1.05, 1.35)
            climb_range = (1.10, 1.60)
            launch_jitter = 1.2
            heading_jitter = 0.26
        elif mode == "eval":
            pos_jitter = (18.0, 22.0, 8.0)
            amp_range = (0.75, 1.30)
            freq_range = (0.82, 1.22)
            climb_range = (0.75, 1.35)
            launch_jitter = 0.8
            heading_jitter = 0.18
        else:
            pos_jitter = (10.0, 14.0, 5.0)
            amp_range = (0.86, 1.16)
            freq_range = (0.90, 1.12)
            climb_range = (0.86, 1.18)
            launch_jitter = 0.4
            heading_jitter = 0.12

        if curriculum in {"tail4_warmup", "tail_warmup", "14-17-warmup", "tail4_warmup_residual", "tail_warmup_residual"}:
            speed_min, speed_max = 16.0, 22.0
            pos_jitter = (5.0, 7.0, 3.0)
            amp_range = (0.35, 0.70)
            freq_range = (0.65, 0.92)
            climb_range = (0.40, 0.75)
            launch_jitter = 0.15
            heading_jitter = 0.04

        target_x = target_x + (rand[:, 7] - 0.5) * 2.0 * pos_jitter[0]
        target_y = target_y + (rand[:, 8] - 0.5) * 2.0 * pos_jitter[1]
        target_z = torch.clamp(target_z + (rand[:, 9] - 0.5) * 2.0 * pos_jitter[2], 28.0, 120.0)
        self.launch_delay[env_ids] = torch.clamp(self.launch_delay[env_ids] + (rand[:, 10] - 0.5) * 2.0 * launch_jitter, min=0.0)
        if curriculum in {"tail4_warmup", "tail_warmup", "14-17-warmup", "tail4_warmup_residual", "tail_warmup_residual"}:
            self.launch_delay[env_ids] = torch.clamp(self.launch_delay[env_ids] * 0.65, 3.8, 5.2)

        basic_mask = sid < tg.LONG_FOLLOW_MIN_SID
        heading_offset = (rand[:, 12] - 0.5) * 2.0 * heading_jitter
        cos_h = torch.cos(heading_offset)
        sin_h = torch.sin(heading_offset)
        dir_x = target_vx * cos_h - target_vy * sin_h
        dir_y = target_vx * sin_h + target_vy * cos_h
        tail_speed = speed_min + rand[:, 11] * (speed_max - speed_min)
        tail_vx = dir_x / torch.linalg.norm(torch.stack((dir_x, dir_y), dim=1), dim=1).clamp_min(1.0e-6) * tail_speed
        tail_vy = dir_y / torch.linalg.norm(torch.stack((dir_x, dir_y), dim=1), dim=1).clamp_min(1.0e-6) * tail_speed
        target_vx = torch.where(basic_mask, dir_x, tail_vx)
        target_vy = torch.where(basic_mask, dir_y, tail_vy)

        self.maneuver_amp_scale[env_ids] = amp_range[0] + rand[:, 13] * (amp_range[1] - amp_range[0])
        self.maneuver_freq_scale[env_ids] = freq_range[0] + rand[:, 14] * (freq_range[1] - freq_range[0])
        self.maneuver_phase[env_ids] = rand[:, 15] * (2.0 * math.pi)
        self.heading_bias[env_ids] = (rand[:, 16] - 0.5) * 2.0 * heading_jitter
        self.climb_scale[env_ids] = climb_range[0] + rand[:, 17] * (climb_range[1] - climb_range[0])
        launch_jitter_val = 0.25 if mode == "train" else (0.5 if mode == "eval" else 0.8)
        launch_vx = launch_vx + (rand[:, 21] - 0.5) * 2.0 * launch_jitter_val
        launch_vy = launch_vy + (rand[:, 5] - 0.5) * 2.0 * launch_jitter_val
        return target_x, target_y, target_z, target_vx, target_vy, launch_vx, launch_vy

    def _randomization_mode(self) -> str:
        mode = os.environ.get("STAGE3_RANDOMIZATION_MODE", "train").strip().lower()
        if mode not in {"train", "eval", "stress"}:
            print(f"[Stage3Random] Unknown STAGE3_RANDOMIZATION_MODE={mode!r}; using train.", flush=True)
            return "train"
        return mode

    def _sample_scenario_ids(self, count: int) -> torch.Tensor:
        curriculum = self._scenario_curriculum()
        if curriculum in {"easy4", "0-3-6"}:
            ids = torch.tensor([0, 2, 3, 6], device=self.device)
            return ids[torch.randint(0, 4, (count,), device=self.device)]
        if curriculum in {"mid6", "1-4-5-8-11"}:
            ids = torch.tensor([1, 4, 5, 8, 11], device=self.device)
            return ids[torch.randint(0, 5, (count,), device=self.device)]
        if curriculum in {"hard4", "7-9-10-12-13"}:
            ids = torch.tensor([7, 9, 10, 12, 13], device=self.device)
            return ids[torch.randint(0, 5, (count,), device=self.device)]
        if curriculum in {"basic14", "0-13", "first14"}:
            max_sid = 14
        elif curriculum in {"easy8", "0-7", "first8"}:
            max_sid = 8
        elif curriculum in {"tail4_warmup", "tail_warmup", "14-17-warmup"}:
            return torch.randint(14, 18, (count,), device=self.device)
        elif curriculum in {"tail4_warmup_residual", "tail_warmup_residual"}:
            return torch.randint(14, 18, (count,), device=self.device)
        elif curriculum in {"tail4", "14-17", "tail_easy"}:
            return torch.randint(14, 18, (count,), device=self.device)
        elif curriculum in {"tail4_residual", "tail_residual"}:
            return torch.randint(14, 18, (count,), device=self.device)
        elif curriculum in {"hard6", "18-23", "tail_hard"}:
            return torch.randint(18, 24, (count,), device=self.device)
        elif curriculum in {"hard6_residual", "tail_hard_residual"}:
            return torch.randint(18, 24, (count,), device=self.device)
        elif curriculum in {"tail10", "14-23", "tail"}:
            return torch.randint(14, 24, (count,), device=self.device)
        elif curriculum in {"tail10_residual", "tail_residual10"}:
            return torch.randint(14, 24, (count,), device=self.device)
        elif curriculum in {"mix24", "mixed24", "basic_tail_mix"}:
            choose_tail = torch.rand(count, device=self.device) < 0.40
            basic_sid = torch.randint(0, 14, (count,), device=self.device)
            tail_sid = torch.randint(14, 24, (count,), device=self.device)
            return torch.where(choose_tail, tail_sid, basic_sid)
        elif curriculum in {"mix24_residual", "mixed24_residual"}:
            choose_tail = torch.rand(count, device=self.device) < 0.40
            basic_sid = torch.randint(0, 14, (count,), device=self.device)
            tail_sid = torch.randint(14, 24, (count,), device=self.device)
            return torch.where(choose_tail, tail_sid, basic_sid)
        elif curriculum in {"full24", "0-23", "all"}:
            max_sid = 24
        else:
            print(f"[Stage3Curriculum] Unknown STAGE3_SCENARIO_CURRICULUM={curriculum!r}; using basic14.", flush=True)
            max_sid = 14
        return torch.randint(0, max_sid, (count,), device=self.device)

    def _scenario_curriculum(self) -> str:
        return os.environ.get("STAGE3_SCENARIO_CURRICULUM", "basic14").strip().lower()

    def _aircraft_speed_range(self, mode: str) -> tuple[float, float]:
        if mode == "eval":
            return 18.0, 32.0
        if mode == "stress":
            return 30.0, 36.0
        return 20.0, 28.0

    def _update_aircraft(self):
        t = self._time()
        sid = self.scenario_id
        vel = self.target_base_vel.clone()
        long_mt = torch.clamp(t - self.launch_delay - 0.8, min=0.0)
        long_mask = sid >= 14
        long_xy = self._long_follow_aircraft_horizontal_velocity(long_mt, sid)
        vel[:, :2] = torch.where(long_mask.unsqueeze(1), long_xy, vel[:, :2])
        maneuver_t = torch.clamp(t - self.launch_delay - 1.4, min=0.0)

        weave = torch.zeros(self.num_envs, device=self.device)
        weave = torch.where(sid == 6, 6.5 * torch.sin(maneuver_t * 2.5), weave)
        double = (
            7.5 * torch.exp(-((maneuver_t - 1.0) / 0.75) ** 2)
            - 8.5 * torch.exp(-((maneuver_t - 2.7) / 0.85) ** 2)
            + 4.5 * torch.exp(-((maneuver_t - 4.0) / 0.95) ** 2)
        )
        weave = torch.where(sid == 7, double, weave)
        fighter = 7.0 * torch.sin(maneuver_t * 2.35) + 4.0 * torch.exp(-((maneuver_t - 1.8) / 0.55) ** 2)
        weave = torch.where(sid == 12, fighter, weave)
        follow = (
            12.0 * torch.exp(-((maneuver_t - 1.0) / 0.75) ** 2)
            - 13.0 * torch.exp(-((maneuver_t - 2.45) / 0.85) ** 2)
            + 8.0 * torch.exp(-((maneuver_t - 3.9) / 1.0) ** 2)
            + 3.0 * torch.sin(maneuver_t * 1.9)
        )
        weave = torch.where(sid == 13, follow, weave)
        evasion_mask = (sid == 6) | (sid == 7) | (sid == 12) | (sid == 13)
        turn = torch.sin(t * self.aircraft_turn_rate)
        vel[:, 1] += torch.where(evasion_mask, weave, self.aircraft_turn_amplitude * turn)

        mt_climb = torch.clamp(t - self.launch_delay - 1.1, min=0.0)
        vel[:, 1] += torch.where(sid == 4, 3.0 * torch.sin(mt_climb * 1.4), torch.zeros_like(vel[:, 1]))
        mt_dive = torch.clamp(t - self.launch_delay - 1.0, min=0.0)
        vel[:, 1] += torch.where(sid == 5, -3.0 * torch.sin(mt_dive * 1.35), torch.zeros_like(vel[:, 1]))

        extra_altitude = torch.zeros(self.num_envs, device=self.device)
        mt_climb = torch.clamp(t - self.launch_delay - 1.1, min=0.0)
        extra_altitude = torch.where(sid == 4, torch.minimum(torch.full_like(mt_climb, 42.0), mt_climb * 10.5), extra_altitude)
        mt_dive = torch.clamp(t - self.launch_delay - 1.0, min=0.0)
        extra_altitude = torch.where(sid == 5, -torch.minimum(torch.full_like(mt_dive, 44.0), mt_dive * 11.0), extra_altitude)
        extra_altitude = torch.where(sid == 12, 7.0 * torch.sin(torch.clamp(t - self.launch_delay - 1.2, min=0.0) * 1.35), extra_altitude)
        mt_follow = torch.clamp(t - self.launch_delay - 0.8, min=0.0)
        follow_alt = (
            10.0 * torch.exp(-((mt_follow - 1.2) / 0.85) ** 2)
            - 8.0 * torch.exp(-((mt_follow - 2.8) / 0.95) ** 2)
            + 4.0 * torch.exp(-((mt_follow - 4.0) / 1.1) ** 2)
        )
        extra_altitude = torch.where(sid == 13, follow_alt, extra_altitude)
        mt_double = torch.clamp(t - self.launch_delay - 3.5, min=0.0)
        extra_altitude = torch.where(sid == 7, 8.0 * torch.sin(mt_double * 1.2), extra_altitude)
        extra_altitude = self._long_follow_extra_altitude(t, sid, extra_altitude)
        altitude_error = self.aircraft_altitude + extra_altitude - self.aircraft_pos[:, 2]
        vertical_limit = torch.full((self.num_envs,), 9.0, device=self.device)
        vertical_limit = torch.where((sid >= 14) & (sid <= 17), torch.full_like(vertical_limit, 11.5), vertical_limit)
        vertical_limit = torch.where(
            (sid == 18) | (sid == 20) | (sid == 23), torch.full_like(vertical_limit, 18.0), vertical_limit
        )
        vertical_limit = torch.where((sid == 19) | (sid == 21) | (sid == 22), torch.full_like(vertical_limit, 11.5), vertical_limit)
        vel[:, 2] = torch.clamp(altitude_error * 0.9, -vertical_limit, vertical_limit)
        self.aircraft_vel = vel
        self.aircraft_pos = self.aircraft_pos + self.aircraft_vel * self.step_dt
        self._push_aircraft_history()

    def _push_aircraft_history(self) -> None:
        long_mask = self.scenario_id >= tg.LONG_FOLLOW_MIN_SID
        if not long_mask.any():
            return
        idx = self.history_len.clamp(0, self._history_steps - 1)
        batch = torch.arange(self.num_envs, device=self.device)
        self.aircraft_history[batch, idx] = self.aircraft_pos
        self.history_len = torch.where(long_mask, self.history_len + 1, self.history_len).clamp(0, self._history_steps)

    def _long_follow_aircraft_horizontal_velocity(self, maneuver_t: torch.Tensor, sid: torch.Tensor) -> torch.Tensor:
        base_speed = torch.linalg.norm(self.target_base_vel[:, :2], dim=1).clamp_min(1.0)
        heading = torch.zeros(self.num_envs, device=self.device)
        speed = base_speed.clone()
        mt = maneuver_t * self.maneuver_freq_scale + self.maneuver_phase

        heading_14 = torch.minimum(torch.full_like(maneuver_t, 5.10), maneuver_t * 0.235 * self.maneuver_freq_scale)
        speed_14 = base_speed + self.maneuver_amp_scale * 1.8 * torch.sin(mt * 0.55)
        heading = torch.where(sid == 14, heading_14, heading)
        speed = torch.where(sid == 14, speed_14, speed)

        heading_15 = (
            0.72 * torch.exp(-((mt - 2.4) / 1.4) ** 2)
            - 0.92 * torch.exp(-((mt - 6.0) / 1.7) ** 2)
            + 0.82 * torch.exp(-((mt - 9.9) / 2.0) ** 2)
            - 0.55 * torch.exp(-((mt - 14.0) / 2.4) ** 2)
            + 0.16 * torch.sin(mt * 0.70)
        ) * self.maneuver_amp_scale
        speed_15 = base_speed + self.maneuver_amp_scale * 2.3 * torch.sin(mt * 0.42)
        heading = torch.where(sid == 15, heading_15, heading)
        speed = torch.where(sid == 15, speed_15, speed)

        heading_16 = self.maneuver_amp_scale * (0.55 * torch.sin(mt * 0.42) + 0.38 * torch.sin(mt * 0.92))
        speed_16 = base_speed + self.maneuver_amp_scale * 1.9 * torch.sin(mt * 0.48)
        heading = torch.where(sid == 16, heading_16, heading)
        speed = torch.where(sid == 16, speed_16, speed)

        heading_17 = self.maneuver_amp_scale * (0.82 * torch.sin(mt * 0.31) + 0.28 * torch.sin(mt * 0.82))
        speed_17 = base_speed + self.maneuver_amp_scale * 2.6 * torch.sin(mt * 0.38)
        heading = torch.where(sid == 17, heading_17, heading)
        speed = torch.where(sid == 17, speed_17, speed)

        heading_18 = self.maneuver_amp_scale * (
            0.42 * torch.exp(-((mt - 3.0) / 1.8) ** 2) - 0.30 * torch.exp(-((mt - 7.2) / 2.0) ** 2)
        )
        speed_18 = torch.clamp(
            base_speed + self.maneuver_amp_scale * 3.0 * torch.sin(mt * 0.42) - self.maneuver_amp_scale * 7.0 * torch.exp(-((mt - 2.0) / 1.15) ** 2),
            min=18.0,
        )
        heading = torch.where(sid == 18, heading_18, heading)
        speed = torch.where(sid == 18, speed_18, speed)

        heading_19 = torch.minimum(torch.full_like(maneuver_t, 6.05), maneuver_t * 0.285 * self.maneuver_freq_scale)
        speed_19 = base_speed + self.maneuver_amp_scale * 1.6 * torch.sin(mt * 0.36)
        heading = torch.where(sid == 19, heading_19, heading)
        speed = torch.where(sid == 19, speed_19, speed)

        heading_20 = torch.minimum(torch.full_like(maneuver_t, 5.4), maneuver_t * 0.245 * self.maneuver_freq_scale) + self.maneuver_amp_scale * 0.22 * torch.sin(
            mt * 0.75
        )
        speed_20 = base_speed + self.maneuver_amp_scale * 1.7 * torch.sin(mt * 0.46)
        heading = torch.where(sid == 20, heading_20, heading)
        speed = torch.where(sid == 20, speed_20, speed)

        heading_21 = (
            1.45 * torch.exp(-((mt - 3.6) / 1.7) ** 2)
            - 1.15 * torch.exp(-((mt - 8.0) / 2.0) ** 2)
            + 0.22 * torch.sin(mt * 0.62)
        ) * self.maneuver_amp_scale
        speed_21 = base_speed + self.maneuver_amp_scale * 2.2 * torch.sin(mt * 0.45)
        heading = torch.where(sid == 21, heading_21, heading)
        speed = torch.where(sid == 21, speed_21, speed)

        heading_22 = self.maneuver_amp_scale * (1.05 * torch.sin(mt * 0.34) + 0.26 * torch.sin(mt * 0.92))
        speed_22 = base_speed + self.maneuver_amp_scale * 2.0 * torch.sin(mt * 0.34)
        heading = torch.where(sid == 22, heading_22, heading)
        speed = torch.where(sid == 22, speed_22, speed)

        heading_23 = (
            0.95 * torch.exp(-((mt - 2.8) / 1.5) ** 2)
            - 1.25 * torch.exp(-((mt - 6.4) / 1.8) ** 2)
            + 1.10 * torch.exp(-((mt - 10.5) / 2.2) ** 2)
            + 0.38 * torch.sin(mt * 0.48)
        ) * self.maneuver_amp_scale
        speed_23 = base_speed + self.maneuver_amp_scale * 2.8 * torch.sin(mt * 0.38)
        heading = torch.where(sid == 23, heading_23, heading)
        speed = torch.where(sid == 23, speed_23, speed)

        heading = heading + self.heading_bias
        return torch.stack((torch.cos(heading) * speed, torch.sin(heading) * speed), dim=1)

    def _long_follow_extra_altitude(
        self, t: torch.Tensor, sid: torch.Tensor, extra_altitude: torch.Tensor
    ) -> torch.Tensor:
        mt_base = torch.clamp(t - self.launch_delay - 0.8, min=0.0)
        mt = mt_base * self.maneuver_freq_scale + self.maneuver_phase
        alt_14 = 8.5 * torch.sin(mt * 0.86) + 5.0 * torch.sin(mt * 0.42)
        alt_15 = (
            13.0 * torch.exp(-((mt - 1.7) / 1.15) ** 2)
            - 14.0 * torch.exp(-((mt - 4.3) / 1.25) ** 2)
            + 12.0 * torch.exp(-((mt - 7.1) / 1.45) ** 2)
            - 7.0 * torch.exp(-((mt - 10.0) / 1.7) ** 2)
            + 4.5 * torch.sin(mt * 0.58)
        )
        alt_16 = (
            torch.minimum(torch.full_like(mt, 42.0), mt * 9.0)
            - torch.minimum(torch.full_like(mt, 48.0), torch.clamp(mt - 4.0, min=0.0) * 10.0)
            + torch.minimum(torch.full_like(mt, 34.0), torch.clamp(mt - 7.8, min=0.0) * 8.0)
            + 10.0 * torch.sin(mt * 0.88)
        )
        alt_17 = (
            10.0 * torch.exp(-((mt - 1.7) / 1.15) ** 2)
            - 12.0 * torch.exp(-((mt - 4.4) / 1.25) ** 2)
            + 10.0 * torch.exp(-((mt - 7.5) / 1.5) ** 2)
            - 7.0 * torch.exp(-((mt - 10.7) / 1.8) ** 2)
            + 4.5 * torch.sin(mt * 0.72)
        )
        alt_18 = 64.0 * torch.exp(-((mt - 2.4) / 1.65) ** 2) - 18.0 * torch.exp(
            -((mt - 6.2) / 2.0) ** 2
        ) + 6.0 * torch.sin(mt * 0.55)
        alt_19 = 5.5 * torch.sin(mt * 0.50)
        alt_20 = torch.minimum(torch.full_like(mt, 46.0), mt * 3.1) - 12.0 * torch.exp(
            -((mt - 13.5) / 3.0) ** 2
        ) + 6.0 * torch.sin(mt * 0.68)
        alt_21 = (
            12.0 * torch.exp(-((mt - 2.4) / 1.1) ** 2)
            - 15.0 * torch.exp(-((mt - 6.4) / 1.7) ** 2)
            + 7.0 * torch.sin(mt * 0.44)
        )
        alt_22 = 8.0 * torch.sin(mt * 0.62) + 4.0 * torch.sin(mt * 1.05)
        alt_23 = (
            28.0 * torch.exp(-((mt - 2.0) / 1.35) ** 2)
            - 22.0 * torch.exp(-((mt - 5.7) / 1.65) ** 2)
            + 18.0 * torch.exp(-((mt - 9.6) / 2.1) ** 2)
            + 7.0 * torch.sin(mt * 0.52)
        )
        for scenario_id, alt in (
            (14, alt_14),
            (15, alt_15),
            (16, alt_16),
            (17, alt_17),
            (18, alt_18),
            (19, alt_19),
            (20, alt_20),
            (21, alt_21),
            (22, alt_22),
            (23, alt_23),
        ):
            extra_altitude = torch.where(sid == scenario_id, alt * self.climb_scale, extra_altitude)
        return extra_altitude

    def _update_missile(self):
        flight_t = self._flight_t()
        prelaunch = flight_t < 0.0
        boost = (flight_t >= 0.0) & (flight_t < self.boost_duration)
        guidance = flight_t >= self.boost_duration

        rel = self.aircraft_pos - self.missile_pos
        horizontal = rel.clone()
        horizontal[:, 2] = 0.0
        vertical = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        horiz_dir = self._unit(horizontal, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        climb_weight = self.boost_climb_weight.unsqueeze(1)
        desired_boost = self._unit((1.0 - climb_weight) * horiz_dir + climb_weight * vertical, vertical)
        speed_now = torch.linalg.norm(self.missile_vel, dim=1)
        just_launched = (flight_t >= 0.0) & (flight_t < self.step_dt * 1.5) & (speed_now < 1.0)
        self.missile_vel = torch.where(just_launched.unsqueeze(1), self.launch_velocity, self.missile_vel)
        self.missile_vel = torch.where(prelaunch.unsqueeze(1), torch.zeros_like(self.missile_vel), self.missile_vel)
        self.missile_vel = torch.where(boost.unsqueeze(1), self.missile_vel + desired_boost * self.cfg.boost_accel * self.step_dt, self.missile_vel)

        forward = self._unit(self.missile_vel, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        right = self._unit(
            torch.cross(forward, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(forward), dim=1),
            torch.tensor([0.0, -1.0, 0.0], device=self.device),
        )
        up = self._unit(torch.cross(right, forward, dim=1), torch.tensor([0.0, 0.0, 1.0], device=self.device))
        speed = torch.linalg.norm(self.missile_vel, dim=1).clamp_min(1.0e-6)
        axial_accel = forward * torch.clamp((self.guidance_speed - speed) * self.cfg.speed_hold_gain, -8.0, 10.0).unsqueeze(1)

        teacher_mode = self._teacher_mode()
        guidance_t = flight_t - self.boost_duration
        long_follow = guidance & (self.scenario_id >= tg.LONG_FOLLOW_MIN_SID) & (teacher_mode in {"full", "only"})
        use_lateral = guidance & (~long_follow)

        if long_follow.any():
            closeout = self._float_env("STAGE3_BASELINE_CLOSEOUT_RANGE", 18.0)
            lf_vel = tg.long_follow_desired_velocity(
                self.scenario_id,
                guidance_t,
                self.missile_pos,
                self.missile_vel,
                self.aircraft_pos,
                self.aircraft_vel,
                self.aircraft_history,
                self.history_len,
                self.guidance_speed,
                self.max_lateral_accel,
                self.step_dt,
                closeout,
            )
            self.missile_vel = torch.where(long_follow.unsqueeze(1), lf_vel, self.missile_vel)
            self.applied_actions = torch.where(long_follow.unsqueeze(1), torch.zeros_like(self.applied_actions), self.applied_actions)

        control_actions = self._guidance_control_actions(forward, right, up)
        self.applied_actions = torch.where(use_lateral.unsqueeze(1), control_actions, self.applied_actions)
        lateral_accel = (right * control_actions[:, 0:1] + up * control_actions[:, 1:2]) * self.max_lateral_accel.unsqueeze(1)
        guided_vel = self.missile_vel + (axial_accel + lateral_accel) * self.step_dt
        guided_speed = torch.linalg.norm(guided_vel, dim=1).clamp_min(1.0e-6)
        speed_limit = self.guidance_speed * 1.15
        guided_vel = torch.where((guided_speed > speed_limit).unsqueeze(1), guided_vel / guided_speed.unsqueeze(1) * speed_limit.unsqueeze(1), guided_vel)
        self.missile_vel = torch.where(use_lateral.unsqueeze(1), guided_vel, self.missile_vel)
        self.missile_pos = self.missile_pos + self.missile_vel * self.step_dt

    def _guidance_control_actions(self, forward: torch.Tensor, right: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        teacher = self._full_teacher_action(forward, right, up)
        mode = self._teacher_mode()
        if mode == "only":
            return teacher
        if mode == "none" and not self._residual_guidance_enabled():
            return self.actions
        if self._residual_guidance_enabled() or mode == "full":
            alpha, beta = self._residual_alpha_beta()
            if mode == "full" and not self._residual_guidance_enabled():
                alpha, beta = 1.0, 0.0
            return (teacher * alpha + self.actions * beta).clamp(-1.0, 1.0)
        return self.actions

    def _full_teacher_action(self, forward: torch.Tensor, right: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        action = tg.baseline_lateral_action(
            self.scenario_id,
            self.missile_pos,
            self.missile_vel,
            self.aircraft_pos,
            self.aircraft_vel,
            forward,
            right,
            up,
            self.last_teacher_action,
            self._float_env("STAGE3_BASELINE_TAIL_GAIN", 4.8),
            self._float_env("STAGE3_BASELINE_TAIL_OFFSET", 4.5),
            self._float_env("STAGE3_BASELINE_ACTION_ALPHA", 0.42),
            self._float_env("STAGE3_BASELINE_CLOSEOUT_RANGE", 22.0),
            self._float_env("STAGE3_BASELINE_TANGENT_BLEND_MAX", 0.45),
        )
        self.last_teacher_action = action.detach()
        return action

    def _baseline_tail_guidance_action(self, forward: torch.Tensor, right: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return self._full_teacher_action(forward, right, up)

    def _teacher_mode(self) -> str:
        mode = os.environ.get("STAGE3_TEACHER_MODE", "full").strip().lower()
        if mode not in {"full", "simple", "none", "only"}:
            return "full"
        if mode == "simple":
            return "none"
        return mode

    @staticmethod
    def _obs_mode() -> str:
        mode = os.environ.get("STAGE3_OBS_MODE", "oracle").strip().lower()
        if mode not in {"oracle", "ir_track", "ir_image"}:
            return "oracle"
        return mode

    @staticmethod
    def _observation_dim(mode: str) -> int:
        return 9 if mode == "ir_track" else 18

    @staticmethod
    def _ir_enabled() -> bool:
        raw = os.environ.get("STAGE3_IR_ENABLE", "").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _update_ir_seeker(self) -> None:
        if self._ir_seeker is None:
            return
        self._ir_seeker.update(
            self.missile_pos,
            self.missile_vel,
            self.aircraft_pos,
            self.aircraft_vel,
            self.step_dt,
        )

    def get_ir_outputs(self) -> dict[str, torch.Tensor]:
        """IR frame + GT track labels (for dataset collection / debug)."""
        if self._ir_seeker is None:
            raise RuntimeError("IR seeker disabled. Set STAGE3_IR_ENABLE=1 before creating env.")
        return {
            "frame": self._ir_seeker.frame,
            "track": self._ir_seeker.track_state(),
            "locked": self._ir_seeker.locked,
        }

    def _active_reward_stage(self) -> str:
        raw = os.environ.get("STAGE3_REWARD_STAGE", "auto").strip().upper()
        if raw in {"A", "B", "C"}:
            return raw
        return tg.reward_stage(int(self.common_step_counter))

    def _residual_alpha_beta(self) -> tuple[float, float]:
        schedule = os.environ.get("STAGE3_RESIDUAL_SCHEDULE", "auto").strip().lower()
        if schedule == "auto":
            return tg.residual_schedule(int(self.common_step_counter))
        alpha = self._float_env("STAGE3_RESIDUAL_ALPHA", 1.0)
        beta = self._float_env("STAGE3_RESIDUAL_BETA", 0.25)
        return alpha, beta

    def get_teacher_action_for_bc(self) -> torch.Tensor:
        """Return current teacher lateral action (for BC data collection)."""
        forward = self._unit(self.missile_vel, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        right = self._unit(
            torch.cross(forward, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(forward), dim=1),
            torch.tensor([0.0, -1.0, 0.0], device=self.device),
        )
        up = self._unit(torch.cross(right, forward, dim=1), torch.tensor([0.0, 0.0, 1.0], device=self.device))
        return self._full_teacher_action(forward, right, up)

    def _residual_guidance_enabled(self) -> bool:
        curriculum = self._scenario_curriculum()
        if curriculum in {
            "tail4_residual",
            "tail4_warmup_residual",
            "hard6_residual",
            "tail10_residual",
            "mix24_residual",
        }:
            return True
        return os.environ.get("STAGE3_RESIDUAL_GUIDANCE", "").strip().lower() in {"1", "true", "yes", "on"}

    def _float_env(self, name: str, default: float) -> float:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            print(f"[Stage3Residual] Ignore invalid {name}={raw!r}; using {default}.", flush=True)
            return default

    def _missile_contact_point(self, pos: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
        direction = self._unit(vel, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        return pos + direction * MISSILE_NOSE_OFFSET

    def _target_contact_segment(self) -> tuple[torch.Tensor, torch.Tensor]:
        target_forward = self._unit(self.aircraft_vel, torch.tensor([1.0, 0.0, 0.0], device=self.device))
        tail_chase = tg._is_tail_chase(self.scenario_id)
        seg_start = self.aircraft_pos - target_forward * 3.6
        tail_end = self.aircraft_pos + target_forward * 0.6
        fuse_end = self.aircraft_pos + target_forward * 3.6
        seg_end = torch.where(tail_chase.unsqueeze(1), tail_end, fuse_end)
        return seg_start, seg_end

    def _contact_distance(self) -> torch.Tensor:
        prev_contact = self._missile_contact_point(self.prev_missile_pos, self.prev_missile_vel)
        curr_contact = self._missile_contact_point(self.missile_pos, self.missile_vel)
        seg_start, seg_end = self._target_contact_segment()
        return self._segment_distance(prev_contact, curr_contact, seg_start, seg_end)

    def _effective_hit_radius(self) -> torch.Tensor:
        # basic14 使用 cfg.hit_radius（与可视/训练一致）；tail 场景保留 CPU 细分半径
        radius = torch.full((self.num_envs,), self.cfg.hit_radius, device=self.device)
        radius = torch.where(self.scenario_id == 14, torch.full_like(radius, 2.70), radius)
        radius = torch.where(self.scenario_id == 15, torch.full_like(radius, 1.70), radius)
        radius = torch.where(self.scenario_id == 16, torch.full_like(radius, 1.95), radius)
        radius = torch.where(self.scenario_id == 17, torch.full_like(radius, 2.45), radius)
        radius = torch.where((self.scenario_id >= 18) & (self.scenario_id <= 23), torch.full_like(radius, 2.55), radius)
        return radius

    def _segment_distance(self, p0: torch.Tensor, p1: torch.Tensor, q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
        """Closest distance between two 3D segments (ported from intercept_core.py)."""
        d1 = p1 - p0
        d2 = q1 - q0
        r = p0 - q0
        a = torch.sum(d1 * d1, dim=1)
        e = torch.sum(d2 * d2, dim=1)
        f = torch.sum(d2 * r, dim=1)
        c = torch.sum(d1 * r, dim=1)
        b = torch.sum(d1 * d2, dim=1)

        small_a = a <= 1.0e-9
        small_e = e <= 1.0e-9

        denom = (a * e - b * b).clamp_min(1.0e-9)
        s = ((b * f - c * e) / denom).clamp(0.0, 1.0)
        t = (b * s + f) / e.clamp_min(1.0e-9)

        s_low = (-c / a.clamp_min(1.0e-9)).clamp(0.0, 1.0)
        t_low = torch.zeros_like(t)
        s_high = ((b - c) / a.clamp_min(1.0e-9)).clamp(0.0, 1.0)
        t_high = torch.ones_like(t)

        use_low = (~small_a) & (~small_e) & (t < 0.0)
        use_high = (~small_a) & (~small_e) & (t > 1.0)
        use_mid = (~small_a) & (~small_e) & (~use_low) & (~use_high)

        s = torch.where(small_a, torch.zeros_like(s), s)
        t = torch.where(small_a, (f / e.clamp_min(1.0e-9)).clamp(0.0, 1.0), t)
        s = torch.where(small_e & (~small_a), s_low, s)
        t = torch.where(small_e & (~small_a), t_low, t)
        s = torch.where(use_low, s_low, s)
        t = torch.where(use_low, t_low, t)
        s = torch.where(use_high, s_high, s)
        t = torch.where(use_high, t_high, t)
        s = torch.where(use_mid, s, s)
        t = torch.where(use_mid, t.clamp(0.0, 1.0), t)

        cp = p0 + d1 * s.unsqueeze(1)
        cq = q0 + d2 * t.unsqueeze(1)
        return torch.linalg.norm(cp - cq, dim=1)

    def _time(self) -> torch.Tensor:
        return self.episode_length_buf.float() * self.step_dt

    def _flight_t(self) -> torch.Tensor:
        return self._time() - self.launch_delay

    def _unit(self, vec: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        if vec.ndim == 1:
            vec = vec.unsqueeze(0).expand(self.num_envs, -1)
        norm = torch.linalg.norm(vec, dim=1, keepdim=True)
        fallback = fallback.unsqueeze(0).expand_as(vec)
        return torch.where(norm > 1.0e-6, vec / norm.clamp_min(1.0e-6), fallback)

    def _print_training_progress(
        self,
        reward: torch.Tensor,
        contact_distance: torch.Tensor,
        center_distance: torch.Tensor,
        hit: torch.Tensor,
        ground: torch.Tensor,
        bounds: torch.Tensor,
        timeout: torch.Tensor,
        selection_score: torch.Tensor,
        episode_hit_rate: torch.Tensor,
    ) -> None:
        step = int(self.common_step_counter)
        interval = int(self.cfg.console_log_interval_steps)
        if step <= 0 or step % interval != 0 or step == self._last_console_log_step:
            return
        self._last_console_log_step = step
        guidance = self._flight_t() >= self.boost_duration
        hit_rate = float(hit.float().mean().detach().cpu())
        in_ep_hit = float(episode_hit_rate.detach().cpu())
        completed_ep_hit_rate = (
            (self._completed_ep_hits / self._completed_ep_total) if self._completed_ep_total else 0.0
        )
        metrics = {
            "env_step": step,
            "hit_rate": hit_rate,
            "in_episode_hit_rate": in_ep_hit,
            "completed_ep_hit_rate": completed_ep_hit_rate,
            "completed_ep_hits": self._completed_ep_hits,
            "completed_ep_total": self._completed_ep_total,
            "interval_hits": self._interval_hits,
            "selection_score": float(selection_score.detach().cpu()),
            "mean_contact_distance": float(contact_distance.mean().detach().cpu()),
            "mean_center_distance": float(center_distance.mean().detach().cpu()),
            "mean_closest_distance": float(self.closest_distance.mean().detach().cpu()),
            "mean_closest_center": float(self.closest_center_distance.mean().detach().cpu()),
            "guidance_fraction": float(guidance.float().mean().detach().cpu()),
            "mean_episode_return": float(self.episode_return.mean().detach().cpu()),
            "mean_step_reward": float(reward.mean().detach().cpu()),
            "reward_stage": self._active_reward_stage(),
            "ground_rate": float(ground.float().mean().detach().cpu()),
            "bounds_rate": float(bounds.float().mean().detach().cpu()),
            "timeout_rate": float(timeout.float().mean().detach().cpu()),
        }
        print(
            "[Stage3Train] "
            f"step={metrics['env_step']} "
            f"completed_ep_hit={metrics['completed_ep_hit_rate']:.3f}"
            f"({metrics['completed_ep_hits']}/{metrics['completed_ep_total']}) "
            f"hits_{interval}={metrics['interval_hits']} in_ep_hit={metrics['in_episode_hit_rate']:.3f} "
            f"hit_step={metrics['hit_rate']:.3f} score={metrics['selection_score']:+.1f} "
            f"contact={metrics['mean_contact_distance']:.1f} center={metrics['mean_center_distance']:.1f} "
            f"closest={metrics['mean_closest_distance']:.1f} closest_c={metrics['mean_closest_center']:.1f} "
            f"guidance={metrics['guidance_fraction']:.2f} return={metrics['mean_episode_return']:+.0f} "
            f"stage={metrics['reward_stage']}",
            flush=True,
        )
        if metrics["ground_rate"] > 0.0 or metrics["bounds_rate"] > 0.0 or metrics["timeout_rate"] > 0.0:
            print(
                "[Stage3Train] "
                f"step={metrics['env_step']} terminal ground={metrics['ground_rate']:.3f} "
                f"bounds={metrics['bounds_rate']:.3f} timeout={metrics['timeout_rate']:.3f}",
                flush=True,
            )
        self._interval_hits = 0
        self._write_console_metrics(metrics)

    def _write_console_metrics(self, metrics: dict[str, float | int]) -> None:
        log_dir = getattr(self.cfg, "log_dir", None)
        if not log_dir:
            return
        output = Path(log_dir) / "console_metrics.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(metrics.keys())
        mode = "a" if output.exists() else "w"
        with output.open(mode, encoding="utf-8", newline="") as handle:
            if mode == "w":
                handle.write(",".join(fieldnames) + "\n")
            handle.write(",".join(str(metrics[name]) for name in fieldnames) + "\n")

    def _setup_policy_visuals(self) -> None:
        self._visual_xforms = {}
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        root_path = "/World/PolicyValidationVisuals"
        UsdGeom.Xform.Define(stage, root_path)
        self._visual_xforms["aircraft"] = self._make_aircraft_visual(stage, f"{root_path}/Aircraft")
        self._visual_xforms["missile"] = self._make_missile_visual(stage, f"{root_path}/Missile")
        self._visual_xforms["aircraft_trail"] = self._make_trail_curve(
            stage, f"{root_path}/AircraftTrail", (0.05, 0.25, 1.0), 0.24
        )
        self._visual_xforms["missile_trail"] = self._make_trail_curve(
            stage, f"{root_path}/MissileTrail", (1.0, 0.08, 0.02), 0.18
        )

        camera = UsdGeom.Camera.Define(stage, "/World/PolicyValidationCamera")
        camera.AddTranslateOp().Set(Gf.Vec3d(150.0, -260.0, 130.0))
        camera.AddRotateXYZOp().Set(Gf.Vec3f(62.0, 0.0, 30.0))
        camera.CreateFocalLengthAttr(22.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 3000.0))

    def _make_aircraft_visual(self, stage, prim_path: str):
        return self._make_reference_visual(
            stage,
            prim_path,
            self._project_asset("assets/converted/F22.usd"),
            scale=(1.0, 1.0, 1.0),
            color=(0.05, 0.25, 1.0),
            material_name="F22ValidationBlue",
            # Imported F22 forward axis is local -Z, up axis is local +Y.
            local_basis_columns=((0.0, -1.0, 0.0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0)),
        )

    def _make_missile_visual(self, stage, prim_path: str):
        return self._make_reference_visual(
            stage,
            prim_path,
            self._project_asset("assets/converted/HQ9DD.usd"),
            scale=(1.0, 1.0, 1.0),
            color=(1.0, 0.08, 0.02),
            material_name="HQ9DDValidationRed",
            # Imported HQ9DD forward axis is local -X, up axis is local +Z.
            local_basis_columns=((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        )

    def _make_reference_visual(
        self,
        stage,
        prim_path: str,
        asset_path: Path,
        scale: tuple[float, float, float],
        color: tuple[float, float, float],
        material_name: str,
        local_basis_columns: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    ):
        if not asset_path.exists():
            print(f"[Stage3Visual] Missing USD asset, using fallback geometry: {asset_path}", flush=True)
            if "Aircraft" in prim_path:
                return self._make_fallback_aircraft_visual(stage, prim_path)
            return self._make_fallback_missile_visual(stage, prim_path)

        xform = UsdGeom.Xform.Define(stage, prim_path)
        model_root = UsdGeom.Xform.Define(stage, f"{prim_path}/ModelRoot")
        asset_prim = stage.DefinePrim(f"{prim_path}/ModelRoot/Asset", "Xform")
        asset_prim.GetReferences().AddReference(str(asset_path).replace("\\", "/"))
        model_root.AddScaleOp().Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))
        model_root.AddOrientOp().Set(self._quat_from_basis_columns(local_basis_columns))
        self._bind_preview_material(stage, asset_prim, color, material_name)
        return (xform.AddTranslateOp(), xform.AddRotateXYZOp())

    def _make_trail_curve(
        self,
        stage,
        prim_path: str,
        color: tuple[float, float, float],
        width: float,
    ):
        curve = UsdGeom.BasisCurves.Define(stage, prim_path)
        curve.CreateTypeAttr("linear")
        curve.CreateCurveVertexCountsAttr([2])
        curve.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)])
        curve.CreateWidthsAttr([float(width)])
        curve.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        return curve

    def _make_fallback_aircraft_visual(self, stage, prim_path: str):
        xform = UsdGeom.Xform.Define(stage, prim_path)
        self._make_box(stage, f"{prim_path}/Fuselage", (7.5, 0.8, 0.8), (0.05, 0.25, 1.0), (0.0, 0.0, 0.0))
        self._make_box(stage, f"{prim_path}/Wing", (1.5, 8.0, 0.18), (0.20, 0.55, 1.0), (-0.7, 0.0, 0.0))
        self._make_box(stage, f"{prim_path}/Tail", (1.0, 3.2, 0.25), (0.20, 0.55, 1.0), (-3.1, 0.0, 0.25))
        return (xform.AddTranslateOp(), xform.AddRotateXYZOp())

    def _make_fallback_missile_visual(self, stage, prim_path: str):
        xform = UsdGeom.Xform.Define(stage, prim_path)
        self._make_box(stage, f"{prim_path}/Body", (3.4, 0.32, 0.32), (1.0, 0.08, 0.02), (0.0, 0.0, 0.0))
        self._make_box(stage, f"{prim_path}/FinH", (0.8, 1.0, 0.10), (1.0, 0.55, 0.05), (-1.2, 0.0, 0.0))
        self._make_box(stage, f"{prim_path}/FinV", (0.8, 0.10, 1.0), (1.0, 0.55, 0.05), (-1.2, 0.0, 0.0))
        return (xform.AddTranslateOp(), xform.AddRotateXYZOp())

    def _make_box(
        self,
        stage,
        prim_path: str,
        scale: tuple[float, float, float],
        color: tuple[float, float, float],
        offset: tuple[float, float, float],
    ) -> None:
        cube = UsdGeom.Cube.Define(stage, prim_path)
        cube.CreateDisplayColorAttr([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        cube.AddTranslateOp().Set(Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2])))
        cube.AddScaleOp().Set(Gf.Vec3d(float(scale[0]), float(scale[1]), float(scale[2])))

    def _project_asset(self, relative_path: str) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            candidate = parent / relative_path
            if candidate.exists():
                return candidate
        return current.parents[7] / relative_path

    def _bind_preview_material(
        self,
        stage,
        prim,
        color: tuple[float, float, float],
        material_name: str,
    ) -> None:
        material_path = f"/World/PolicyValidationMaterials/{material_name}"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.42)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.08)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)

    def _quat_from_basis_columns(
        self,
        columns: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]],
    ) -> Gf.Quatf:
        r00, r10, r20 = columns[0]
        r01, r11, r21 = columns[1]
        r02, r12, r22 = columns[2]
        trace = r00 + r11 + r22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (r21 - r12) / s
            y = (r02 - r20) / s
            z = (r10 - r01) / s
        elif r00 > r11 and r00 > r22:
            s = math.sqrt(1.0 + r00 - r11 - r22) * 2.0
            w = (r21 - r12) / s
            x = 0.25 * s
            y = (r01 + r10) / s
            z = (r02 + r20) / s
        elif r11 > r22:
            s = math.sqrt(1.0 + r11 - r00 - r22) * 2.0
            w = (r02 - r20) / s
            x = (r01 + r10) / s
            y = 0.25 * s
            z = (r12 + r21) / s
        else:
            s = math.sqrt(1.0 + r22 - r00 - r11) * 2.0
            w = (r10 - r01) / s
            x = (r02 + r20) / s
            y = (r12 + r21) / s
            z = 0.25 * s
        return Gf.Quatf(float(w), float(x), float(y), float(z))

    def _update_policy_visuals(self) -> None:
        if self.num_envs > int(self.cfg.visual_debug_max_envs) or not getattr(self, "_visual_xforms", None):
            return
        aircraft_pos = self.aircraft_pos[0].detach().cpu().tolist()
        aircraft_vel = self.aircraft_vel[0].detach().cpu().tolist()
        missile_pos = self.missile_pos[0].detach().cpu().tolist()
        missile_vel = self.missile_vel[0].detach().cpu().tolist()
        if int(self.episode_length_buf[0].detach().cpu()) <= 1:
            self._visual_aircraft_trail.clear()
            self._visual_missile_trail.clear()
        self._set_visual_pose("aircraft", aircraft_pos, aircraft_vel)
        self._set_visual_pose("missile", missile_pos, missile_vel)
        self._append_visual_trail("aircraft_trail", self._visual_aircraft_trail, aircraft_pos)
        self._append_visual_trail("missile_trail", self._visual_missile_trail, missile_pos)

    def _set_visual_pose(self, name: str, position: list[float], velocity: list[float]) -> None:
        translate_op, rotate_op = self._visual_xforms[name]
        translate_op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        speed_xy = math.hypot(float(velocity[0]), float(velocity[1]))
        yaw = math.degrees(math.atan2(float(velocity[1]), float(velocity[0])))
        pitch = -math.degrees(math.atan2(float(velocity[2]), max(speed_xy, 1.0e-6)))
        rotate_op.Set(Gf.Vec3f(0.0, float(pitch), float(yaw)))

    def _append_visual_trail(self, name: str, trail: list[tuple[float, float, float]], position: list[float]) -> None:
        trail.append((float(position[0]), float(position[1]), float(position[2])))
        max_points = int(self.cfg.visual_trail_points)
        if len(trail) > max_points:
            del trail[: len(trail) - max_points]
        if len(trail) == 1:
            points = [Gf.Vec3f(*trail[0]), Gf.Vec3f(*trail[0])]
        else:
            points = [Gf.Vec3f(x, y, z) for x, y, z in trail]
        curve = self._visual_xforms[name]
        curve.GetCurveVertexCountsAttr().Set([len(points)])
        curve.GetPointsAttr().Set(points)

    def _fixed_visual_scenario_id(self) -> int | None:
        raw = os.environ.get("STAGE3_FIXED_SCENARIO_ID", "").strip()
        if not raw:
            return None
        try:
            scenario_id = int(raw)
        except ValueError:
            print(f"[Stage3Visual] Ignore invalid STAGE3_FIXED_SCENARIO_ID={raw!r}", flush=True)
            return None
        if scenario_id < 0 or scenario_id > 23:
            print(f"[Stage3Visual] Ignore out-of-range STAGE3_FIXED_SCENARIO_ID={scenario_id}; use 0-23.", flush=True)
            return None
        return scenario_id
