from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np


DT = 1.0 / 60.0
MISSILE_VISUAL_NOSE_OFFSET = 2.35
FUSELAGE_CONTACT_RADIUS = 0.95


class FlightPhase(str, Enum):
    BOOST = "BOOST"
    GUIDANCE = "GUIDANCE"
    HIT = "HIT"
    MISS = "MISS"


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def unit(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    length = norm(vec)
    if length < 1e-9:
        if fallback is None:
            return np.array([1.0, 0.0, 0.0], dtype=float)
        return fallback.copy()
    return vec / length


def clamp_vec(vec: np.ndarray, max_length: float) -> np.ndarray:
    length = norm(vec)
    if length <= max_length or length < 1e-9:
        return vec
    return vec / length * max_length


def closest_point_on_segment(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    segment = end - start
    denom = max(float(np.dot(segment, segment)), 1e-9)
    alpha = float(np.dot(point - start, segment) / denom)
    return start + segment * float(np.clip(alpha, 0.0, 1.0))


def closest_points_between_segments(
    p1: np.ndarray,
    q1: np.ndarray,
    p2: np.ndarray,
    q2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    if a <= 1e-9 and e <= 1e-9:
        return p1.copy(), p2.copy()
    if a <= 1e-9:
        s = 0.0
        t = float(np.clip(f / max(e, 1e-9), 0.0, 1.0))
    else:
        c = float(np.dot(d1, r))
        if e <= 1e-9:
            t = 0.0
            s = float(np.clip(-c / a, 0.0, 1.0))
        else:
            b = float(np.dot(d1, d2))
            denom = a * e - b * b
            if abs(denom) > 1e-9:
                s = float(np.clip((b * f - c * e) / denom, 0.0, 1.0))
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = float(np.clip(-c / a, 0.0, 1.0))
            elif t > 1.0:
                t = 1.0
                s = float(np.clip((b - c) / a, 0.0, 1.0))

    return p1 + d1 * s, p2 + d2 * t


def quat_from_matrix(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (rotation[2, 1] - rotation[1, 2]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ],
            dtype=float,
        )
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / s,
                0.25 * s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
            ],
            dtype=float,
        )
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        quat = np.array(
            [
                (rotation[0, 2] - rotation[2, 0]) / s,
                (rotation[0, 1] + rotation[1, 0]) / s,
                0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
            ],
            dtype=float,
        )
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        quat = np.array(
            [
                (rotation[1, 0] - rotation[0, 1]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                0.25 * s,
            ],
            dtype=float,
        )

    return quat / norm(quat)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def rotate_vector(q: np.ndarray, vec: np.ndarray) -> np.ndarray:
    vq = np.array([0.0, vec[0], vec[1], vec[2]], dtype=float)
    return quat_multiply(quat_multiply(q, vq), quat_conjugate(q))[1:]


def aircraft_quat(velocity: np.ndarray, bank: float) -> np.ndarray:
    forward = unit(velocity)
    world_up = np.array([0.0, 0.0, 1.0])
    right = unit(np.cross(world_up, forward), np.array([0.0, 1.0, 0.0]))
    up = unit(np.cross(forward, right))
    up_banked = unit(up * math.cos(bank) + right * math.sin(bank))
    right_banked = unit(np.cross(up_banked, forward))
    rotation = np.array(
        [
            [forward[0], right_banked[0], up_banked[0]],
            [forward[1], right_banked[1], up_banked[1]],
            [forward[2], right_banked[2], up_banked[2]],
        ],
        dtype=float,
    )
    return quat_from_matrix(rotation)


def missile_quat(velocity: np.ndarray) -> np.ndarray:
    body_z = unit(velocity, np.array([0.0, 0.0, 1.0]))
    world_up = np.array([0.0, 0.0, 1.0])
    body_x = np.cross(body_z, world_up)
    if norm(body_x) < 1e-6:
        body_x = np.array([1.0, 0.0, 0.0])
    body_x = unit(body_x)
    body_y = unit(np.cross(body_z, body_x))
    rotation = np.array(
        [
            [body_x[0], body_y[0], body_z[0]],
            [body_x[1], body_y[1], body_z[1]],
            [body_x[2], body_y[2], body_z[2]],
        ],
        dtype=float,
    )
    return quat_from_matrix(rotation)


@dataclass
class VehicleState:
    position: np.ndarray
    velocity: np.ndarray
    bank: float = 0.0


@dataclass
class InterceptConfig:
    dt: float = DT
    max_time: float = 24.0
    hit_radius: float = 5.0
    boost_duration: float = 2.4
    boost_accel: float = 11.0
    boost_climb_weight: float = 0.48
    guidance_speed: float = 46.0
    speed_hold_gain: float = 2.4
    max_lateral_accel: float = 85.0
    max_altitude: float = 220.0
    max_range: float = 620.0
    aircraft_altitude: float = 55.0
    aircraft_turn_rate: float = 0.33
    aircraft_turn_amplitude: float = 2.2


@dataclass
class StepRecord:
    t: float
    phase: str
    distance: float
    closest_distance: float
    aircraft_x: float
    aircraft_y: float
    aircraft_z: float
    missile_x: float
    missile_y: float
    missile_z: float
    missile_speed: float
    action_yaw: float
    action_pitch: float
    reward: float


class InterceptScenario:
    """Simplified two-phase intercept simulation for RL prototyping."""

    def __init__(self, cfg: InterceptConfig | None = None, seed: int | None = None) -> None:
        self.cfg = cfg or InterceptConfig()
        self.rng = np.random.default_rng(seed)
        self.aircraft = VehicleState(np.zeros(3), np.zeros(3))
        self.missile = VehicleState(np.zeros(3), np.zeros(3))
        self.phase = FlightPhase.BOOST
        self.t = 0.0
        self.closest_distance = float("inf")
        self.last_distance = float("inf")
        self.records: list[StepRecord] = []
        self.target_speed_base = np.array([-11.5, 3.2, 0.0], dtype=float)
        self.aircraft_altitude = self.cfg.aircraft_altitude
        self.aircraft_turn_amplitude = self.cfg.aircraft_turn_amplitude
        self.aircraft_turn_rate = self.cfg.aircraft_turn_rate
        self.scenario_type = "head_on"
        self.launch_delay = 0.0
        self.launch_velocity = np.array([7.0, -1.0, 13.0], dtype=float)
        self.last_guidance_action = np.zeros(2, dtype=float)
        self.reset(randomize=False)

    def is_tail_chase_scenario(self) -> bool:
        return self.scenario_type == "overfly_tail_chase"

    def effective_hit_radius(self) -> float:
        return FUSELAGE_CONTACT_RADIUS

    def reset(self, randomize: bool = False, scenario_type: str = "nominal") -> np.ndarray:
        cfg = self.cfg
        self.aircraft_altitude = cfg.aircraft_altitude
        self.aircraft_turn_amplitude = cfg.aircraft_turn_amplitude
        self.aircraft_turn_rate = cfg.aircraft_turn_rate
        self.scenario_type = self.normalize_scenario_type(scenario_type)
        self.launch_delay = 0.0

        if randomize:
            target_x = self.rng.uniform(170.0, 230.0)
            target_y = self.rng.uniform(-80.0, 80.0)
            target_z = self.rng.uniform(42.0, 72.0)
            target_vx = self.rng.uniform(-13.0, -9.0)
            target_vy = self.rng.uniform(1.0, 5.0)
            launch_vx = self.rng.uniform(5.0, 10.0)
            launch_vy = self.rng.uniform(-3.0, 3.0)
        else:
            target_x, target_y, target_z = 205.0, -58.0, cfg.aircraft_altitude
            target_vx, target_vy = -11.5, 3.2
            launch_vx, launch_vy = 7.0, -1.0

        if self.scenario_type == "head_on":
            self.aircraft_turn_amplitude = 1.0
        elif self.scenario_type == "overfly_tail_chase":
            target_x = -50.0
            target_y = 0.0
            target_z = 58.0
            target_vx = 12.2
            target_vy = 0.0
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 0.15
            self.launch_delay = 5.4
            launch_vx, launch_vy = 8.0, 0.0
        elif self.scenario_type == "crossing_left_to_right":
            target_x = self.rng.uniform(55.0, 95.0) if randomize else 78.0
            target_y = self.rng.uniform(-180.0, -135.0) if randomize else -160.0
            target_z = self.rng.uniform(45.0, 62.0) if randomize else 54.0
            target_vx = self.rng.uniform(-2.0, 1.0) if randomize else -0.8
            target_vy = self.rng.uniform(13.0, 16.0) if randomize else 14.5
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 0.8
        elif self.scenario_type == "crossing_right_to_left":
            target_x = self.rng.uniform(55.0, 95.0) if randomize else 78.0
            target_y = self.rng.uniform(135.0, 180.0) if randomize else 160.0
            target_z = self.rng.uniform(45.0, 62.0) if randomize else 54.0
            target_vx = self.rng.uniform(-2.0, 1.0) if randomize else -0.8
            target_vy = self.rng.uniform(-16.0, -13.0) if randomize else -14.5
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 0.8
        elif self.scenario_type == "climb_escape":
            target_x = self.rng.uniform(175.0, 215.0) if randomize else 195.0
            target_y = self.rng.uniform(-70.0, -35.0) if randomize else -52.0
            target_z = self.rng.uniform(38.0, 48.0) if randomize else 42.0
            target_vx = self.rng.uniform(-15.0, -12.0) if randomize else -13.8
            target_vy = self.rng.uniform(2.0, 4.5) if randomize else 3.2
            self.aircraft_turn_amplitude = 1.6
            self.aircraft_altitude = target_z
        elif self.scenario_type == "dive_escape":
            target_x = self.rng.uniform(175.0, 215.0) if randomize else 198.0
            target_y = self.rng.uniform(35.0, 75.0) if randomize else 58.0
            target_z = self.rng.uniform(78.0, 94.0) if randomize else 86.0
            target_vx = self.rng.uniform(-15.0, -12.0) if randomize else -13.5
            target_vy = self.rng.uniform(-4.5, -2.0) if randomize else -3.4
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 1.6
        elif self.scenario_type == "s_turn_evasion":
            self.aircraft_turn_amplitude = 0.0
            self.aircraft_turn_rate = 0.65
        elif self.scenario_type == "double_evasion":
            self.aircraft_turn_amplitude = 0.0
        elif self.scenario_type == "late_launch":
            target_x = self.rng.uniform(18.0, 36.0) if randomize else 28.0
            target_y = self.rng.uniform(42.0, 76.0) if randomize else 58.0
            target_z = self.rng.uniform(50.0, 68.0) if randomize else 59.0
            target_vx = self.rng.uniform(12.0, 14.0) if randomize else 13.0
            target_vy = self.rng.uniform(-5.5, -3.0) if randomize else -4.2
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 2.2
            self.launch_delay = 3.8
            launch_vx, launch_vy = 7.0, 1.4
        elif self.scenario_type == "high_speed_pass":
            target_x = self.rng.uniform(-105.0, -82.0) if randomize else -94.0
            target_y = self.rng.uniform(-95.0, -62.0) if randomize else -78.0
            target_z = self.rng.uniform(60.0, 78.0) if randomize else 68.0
            target_vx = self.rng.uniform(20.0, 24.0) if randomize else 22.0
            target_vy = self.rng.uniform(5.0, 8.0) if randomize else 6.5
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 2.6
            self.aircraft_turn_rate = 0.42
            self.launch_delay = 4.8
            launch_vx, launch_vy = 8.5, -1.0
        elif self.scenario_type == "low_altitude_pass":
            target_x = self.rng.uniform(-82.0, -62.0) if randomize else -72.0
            target_y = self.rng.uniform(35.0, 58.0) if randomize else 46.0
            target_z = self.rng.uniform(33.0, 41.0) if randomize else 37.0
            target_vx = self.rng.uniform(13.5, 15.5) if randomize else 14.5
            target_vy = self.rng.uniform(-3.6, -2.0) if randomize else -2.8
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 0.4
            self.aircraft_turn_rate = 0.28
            self.launch_delay = 2.6
            launch_vx, launch_vy = 9.0, 0.6
        elif self.scenario_type == "far_tail_chase":
            target_x = self.rng.uniform(145.0, 185.0) if randomize else 165.0
            target_y = self.rng.uniform(72.0, 118.0) if randomize else 95.0
            target_z = self.rng.uniform(70.0, 92.0) if randomize else 82.0
            target_vx = self.rng.uniform(9.0, 11.5) if randomize else 10.2
            target_vy = self.rng.uniform(-3.2, -1.2) if randomize else -2.1
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 1.4
            self.aircraft_turn_rate = 0.24
            launch_vx, launch_vy = 9.0, 2.2
        elif self.scenario_type == "fighter_weave_chase":
            target_x = self.rng.uniform(-90.0, -70.0) if randomize else -80.0
            target_y = self.rng.uniform(-48.0, -25.0) if randomize else -36.0
            target_z = self.rng.uniform(58.0, 72.0) if randomize else 65.0
            target_vx = self.rng.uniform(19.0, 22.0) if randomize else 20.5
            target_vy = self.rng.uniform(2.0, 4.0) if randomize else 3.0
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 0.0
            self.aircraft_turn_rate = 0.75
            self.launch_delay = 1.8
            launch_vx, launch_vy = 9.5, 0.0
        elif self.scenario_type == "maneuver_follow_chase":
            target_x = self.rng.uniform(-112.0, -96.0) if randomize else -104.0
            target_y = self.rng.uniform(-22.0, 22.0) if randomize else 0.0
            target_z = self.rng.uniform(56.0, 68.0) if randomize else 62.0
            target_vx = self.rng.uniform(17.0, 19.0) if randomize else 18.0
            target_vy = self.rng.uniform(-0.8, 0.8) if randomize else 0.0
            self.aircraft_altitude = target_z
            self.aircraft_turn_amplitude = 0.0
            self.aircraft_turn_rate = 0.7
            self.launch_delay = 1.2
            launch_vx, launch_vy = 9.5, 0.0
        elif self.scenario_type == "straight":
            self.aircraft_turn_amplitude = 0.0
        elif self.scenario_type == "turning":
            self.aircraft_turn_amplitude = 3.8
            self.aircraft_turn_rate = 0.45
        elif self.scenario_type == "high_altitude":
            target_z = self.rng.uniform(70.0, 95.0) if randomize else 82.0
            self.aircraft_altitude = target_z
        elif self.scenario_type == "low_altitude":
            target_z = self.rng.uniform(32.0, 45.0) if randomize else 38.0
            self.aircraft_altitude = target_z
        elif self.scenario_type == "fast_target":
            target_vx *= 1.35
            target_vy *= 1.25
            self.aircraft_turn_amplitude = 2.8
        elif self.scenario_type == "far_target":
            target_x += self.rng.uniform(55.0, 95.0) if randomize else 75.0
            target_y += self.rng.uniform(-35.0, 35.0) if randomize else -20.0
        elif self.scenario_type == "wide_offset":
            target_y += self.rng.choice([-1.0, 1.0]) * (self.rng.uniform(85.0, 130.0) if randomize else 105.0)
            launch_vy += self.rng.uniform(-2.0, 2.0) if randomize else 1.0
        elif self.scenario_type != "nominal":
            raise ValueError(f"Unknown scenario_type: {scenario_type}")

        self.target_speed_base = np.array([target_vx, target_vy, 0.0], dtype=float)
        self.launch_velocity = np.array([launch_vx, launch_vy, 13.0], dtype=float)

        self.aircraft = VehicleState(
            position=np.array([target_x, target_y, target_z], dtype=float),
            velocity=self.target_speed_base.copy(),
        )
        self.missile = VehicleState(
            position=np.array([0.0, 0.0, 2.0], dtype=float),
            velocity=np.zeros(3, dtype=float) if self.launch_delay > 0.0 else self.launch_velocity.copy(),
        )
        self.phase = FlightPhase.BOOST
        self.t = 0.0
        self.last_guidance_action = np.zeros(2, dtype=float)
        self.closest_distance = norm(self.scoring_point() - self.missile_contact_point())
        self.last_distance = self.closest_distance
        self.records = []
        return self.observation()

    @staticmethod
    def normalize_scenario_type(scenario_type: str) -> str:
        aliases = {
            "nominal": "head_on",
            "headon": "head_on",
            "head_on_frontal": "head_on",
            "overfly": "overfly_tail_chase",
            "tail_chase": "overfly_tail_chase",
            "crossing_lr": "crossing_left_to_right",
            "crossing_rl": "crossing_right_to_left",
            "climb": "climb_escape",
            "dive": "dive_escape",
            "s_turn": "s_turn_evasion",
            "double": "double_evasion",
            "low_pass": "low_altitude_pass",
            "follow": "maneuver_follow_chase",
        }
        return aliases.get(scenario_type, scenario_type)

    def observation(self) -> np.ndarray:
        rel_pos = self.aircraft.position - self.missile.position
        rel_vel = self.aircraft.velocity - self.missile.velocity
        distance = max(norm(rel_pos), 1e-6)
        los = rel_pos / distance
        missile_dir = unit(self.missile.velocity, np.array([0.0, 0.0, 1.0]))
        phase_flag = 0.0 if self.phase == FlightPhase.BOOST else 1.0
        return np.array(
            [
                rel_pos[0] / 250.0,
                rel_pos[1] / 250.0,
                rel_pos[2] / 120.0,
                rel_vel[0] / 90.0,
                rel_vel[1] / 90.0,
                rel_vel[2] / 90.0,
                missile_dir[0],
                missile_dir[1],
                missile_dir[2],
                distance / 300.0,
                self.missile.position[2] / 120.0,
                self.t / self.cfg.max_time,
                phase_flag,
            ],
            dtype=np.float32,
        )

    def boost_action(self) -> np.ndarray:
        rel = self.aircraft.position - self.missile.position
        horizontal = np.array([rel[0], rel[1], 0.0], dtype=float)
        climb = np.array([0.0, 0.0, 1.0], dtype=float)
        climb_weight = self.scenario_boost_climb_weight()
        desired = unit((1.0 - climb_weight) * unit(horizontal) + climb_weight * climb)
        return desired

    def scenario_boost_climb_weight(self) -> float:
        if self.scenario_type == "low_altitude_pass":
            return 0.22
        if self.scenario_type == "far_tail_chase":
            return 0.38
        return self.cfg.boost_climb_weight

    def scenario_boost_duration(self) -> float:
        if self.scenario_type == "low_altitude_pass":
            return 1.55
        if self.scenario_type == "fighter_weave_chase":
            return 1.8
        if self.scenario_type == "maneuver_follow_chase":
            return 1.9
        return self.cfg.boost_duration

    def baseline_guidance_action(self) -> np.ndarray:
        forward, right, up = self.missile_frame()
        rel_pos = self.aircraft.position - self.missile.position
        lead_time = np.clip(norm(rel_pos) / max(norm(self.missile.velocity), 1.0), 0.2, 2.0)
        aim_point = self.guidance_aim_point(lead_time)
        desired_dir = unit(aim_point - self.missile.position, forward)
        correction = desired_dir - forward * np.dot(desired_dir, forward)
        gain = self.guidance_gain()
        raw = np.array(
            [
                np.clip(np.dot(correction, right) * gain, -1.0, 1.0),
                np.clip(np.dot(correction, up) * gain, -1.0, 1.0),
            ],
            dtype=float,
        )
        alpha = self.guidance_action_alpha()
        smoothed = self.last_guidance_action * (1.0 - alpha) + raw * alpha
        self.last_guidance_action = np.clip(smoothed, -1.0, 1.0)
        return self.last_guidance_action.copy()

    def guidance_gain(self) -> float:
        if self.scenario_type == "low_altitude_pass":
            return 3.2
        if self.scenario_type == "overfly_tail_chase":
            return 3.2
        if self.is_tail_chase_scenario():
            return 2.8
        if self.scenario_type in {"climb_escape", "dive_escape", "fighter_weave_chase", "maneuver_follow_chase"}:
            return 3.6
        return 4.2

    def guidance_action_alpha(self) -> float:
        if self.scenario_type == "low_altitude_pass":
            return 0.30
        if self.scenario_type == "overfly_tail_chase":
            return 0.30
        if self.is_tail_chase_scenario():
            return 0.24
        return 0.34

    def guidance_aim_point(self, lead_time: float) -> np.ndarray:
        rel_vel = self.aircraft.velocity - self.missile.velocity
        if self.scenario_type == "fighter_weave_chase":
            return self.aircraft.position + self.aircraft.velocity * lead_time * 0.12
        if self.scenario_type == "maneuver_follow_chase":
            return self.aircraft.position + self.aircraft.velocity * lead_time * 0.10
        if self.is_tail_chase_scenario():
            return self.mid_tail_point() + self.aircraft.velocity * lead_time * self.tail_chase_lead_scale()
        if self.scenario_type in {"s_turn_evasion", "double_evasion"}:
            return self.aircraft.position + rel_vel * lead_time * 0.08
        return self.aircraft.position + rel_vel * lead_time * 0.12

    def tail_chase_lead_scale(self) -> float:
        if self.scenario_type == "low_altitude_pass":
            return 0.06
        if self.scenario_type == "high_speed_pass":
            return 0.14
        if self.scenario_type == "far_tail_chase":
            return 0.10
        if self.scenario_type == "late_launch":
            return 0.08
        return 0.06

    def aircraft_forward(self) -> np.ndarray:
        return unit(self.aircraft.velocity, np.array([1.0, 0.0, 0.0]))

    def tail_point(self) -> np.ndarray:
        return self.aircraft.position - self.aircraft_forward() * 4.4

    def mid_tail_point(self) -> np.ndarray:
        return self.aircraft.position - self.aircraft_forward() * 2.2

    def mid_tail_segment_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        forward = self.aircraft_forward()
        return self.aircraft.position - forward * 3.6, self.aircraft.position + forward * 0.6

    def fuselage_segment_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        forward = self.aircraft_forward()
        return self.aircraft.position - forward * 3.6, self.aircraft.position + forward * 3.6

    def target_contact_segment_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        if self.is_tail_chase_scenario():
            return self.mid_tail_segment_endpoints()
        return self.fuselage_segment_endpoints()

    def mid_tail_segment_closest_point(self) -> np.ndarray:
        tail, mid = self.mid_tail_segment_endpoints()
        return closest_point_on_segment(self.missile_contact_point(), tail, mid)

    def missile_contact_point(self) -> np.ndarray:
        return self.missile.position + unit(self.missile.velocity, np.array([0.0, 0.0, 1.0])) * MISSILE_VISUAL_NOSE_OFFSET

    def scoring_point(self) -> np.ndarray:
        start, end = self.target_contact_segment_endpoints()
        return closest_point_on_segment(self.missile_contact_point(), start, end)

    def missile_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = unit(self.missile.velocity, np.array([1.0, 0.0, 0.0]))
        world_up = np.array([0.0, 0.0, 1.0])
        right = unit(np.cross(forward, world_up), np.array([0.0, -1.0, 0.0]))
        up = unit(np.cross(right, forward), world_up)
        return forward, right, up

    def update_aircraft(self) -> None:
        turn = math.sin(self.t * self.aircraft_turn_rate)
        self.aircraft.velocity = self.target_speed_base.copy()
        extra_altitude = 0.0

        if self.scenario_type in {"s_turn_evasion", "double_evasion", "fighter_weave_chase", "maneuver_follow_chase"}:
            self.aircraft.velocity[1] += self.evasion_lateral_velocity()
        else:
            self.aircraft.velocity[1] += self.aircraft_turn_amplitude * turn

        if self.scenario_type == "climb_escape" and self.t > self.launch_delay + 1.1:
            maneuver_t = self.t - self.launch_delay - 1.1
            extra_altitude = min(42.0, maneuver_t * 10.5)
            self.aircraft.velocity[1] += 3.0 * math.sin(maneuver_t * 1.4)
        elif self.scenario_type == "dive_escape" and self.t > self.launch_delay + 1.0:
            maneuver_t = self.t - self.launch_delay - 1.0
            extra_altitude = -min(44.0, maneuver_t * 11.0)
            self.aircraft.velocity[1] += -3.0 * math.sin(maneuver_t * 1.35)
        elif self.scenario_type == "double_evasion" and self.t > self.launch_delay + 3.5:
            extra_altitude = 8.0 * math.sin((self.t - self.launch_delay - 3.5) * 1.2)
        elif self.scenario_type == "fighter_weave_chase" and self.t > self.launch_delay + 1.2:
            maneuver_t = self.t - self.launch_delay - 1.2
            extra_altitude = 7.0 * math.sin(maneuver_t * 1.35)
        elif self.scenario_type == "maneuver_follow_chase" and self.t > self.launch_delay + 0.8:
            maneuver_t = self.t - self.launch_delay - 0.8
            climb = 10.0 * math.exp(-((maneuver_t - 1.2) / 0.85) ** 2)
            dive = -8.0 * math.exp(-((maneuver_t - 2.8) / 0.95) ** 2)
            settle = 4.0 * math.exp(-((maneuver_t - 4.0) / 1.1) ** 2)
            extra_altitude = climb + dive + settle

        altitude_error = (self.aircraft_altitude + extra_altitude) - self.aircraft.position[2]
        vertical_limit = 3.0
        if self.scenario_type in {"climb_escape", "dive_escape", "fighter_weave_chase", "maneuver_follow_chase"}:
            vertical_limit = 9.0
        self.aircraft.velocity[2] = np.clip(altitude_error * 0.9, -vertical_limit, vertical_limit)
        lateral_command = self.aircraft.velocity[1] - self.target_speed_base[1]
        self.aircraft.bank = np.clip(-0.08 * lateral_command - 0.12 * math.cos(self.t * self.aircraft_turn_rate), -0.55, 0.55)
        self.aircraft.position = self.aircraft.position + self.aircraft.velocity * self.cfg.dt

    def evasion_lateral_velocity(self) -> float:
        maneuver_t = max(0.0, self.t - self.launch_delay - 1.4)
        if self.scenario_type == "s_turn_evasion":
            return 6.5 * math.sin(maneuver_t * 2.5) if maneuver_t > 0.0 else 0.0
        if self.scenario_type == "double_evasion":
            first = 7.5 * math.exp(-((maneuver_t - 1.0) / 0.75) ** 2)
            second = -8.5 * math.exp(-((maneuver_t - 2.7) / 0.85) ** 2)
            third = 4.5 * math.exp(-((maneuver_t - 4.0) / 0.95) ** 2)
            return first + second + third
        if self.scenario_type == "fighter_weave_chase":
            if maneuver_t <= 0.0:
                return 0.0
            weave = 7.0 * math.sin(maneuver_t * 2.35)
            hard_break = 4.0 * math.exp(-((maneuver_t - 1.8) / 0.55) ** 2)
            return weave + hard_break
        if self.scenario_type == "maneuver_follow_chase":
            if maneuver_t <= 0.0:
                return 0.0
            first_break = 12.0 * math.exp(-((maneuver_t - 1.0) / 0.75) ** 2)
            reverse_break = -13.0 * math.exp(-((maneuver_t - 2.45) / 0.85) ** 2)
            final_break = 8.0 * math.exp(-((maneuver_t - 3.9) / 1.0) ** 2)
            weave = 3.0 * math.sin(maneuver_t * 1.9)
            return first_break + reverse_break + final_break + weave
        return 0.0

    def scenario_guidance_speed(self) -> float:
        if self.scenario_type == "fighter_weave_chase":
            return 68.0
        if self.scenario_type == "maneuver_follow_chase":
            return 62.0
        if self.scenario_type == "high_speed_pass":
            return 60.0
        if self.scenario_type in {"climb_escape", "dive_escape"}:
            return 56.0
        if self.scenario_type == "low_altitude_pass":
            return 58.0
        if self.scenario_type == "far_tail_chase":
            return 55.0
        if self.scenario_type == "overfly_tail_chase":
            return 54.0
        return self.cfg.guidance_speed

    def scenario_max_lateral_accel(self) -> float:
        if self.scenario_type == "low_altitude_pass":
            return 54.0
        if self.scenario_type == "fighter_weave_chase":
            return 96.0
        if self.scenario_type == "maneuver_follow_chase":
            return 88.0
        if self.scenario_type == "high_speed_pass":
            return 64.0
        if self.is_tail_chase_scenario():
            return 68.0
        if self.scenario_type in {"climb_escape", "dive_escape"}:
            return 76.0
        return self.cfg.max_lateral_accel

    def update_missile(self, action: np.ndarray | None) -> np.ndarray:
        cfg = self.cfg
        if self.t < self.launch_delay:
            self.phase = FlightPhase.BOOST
            self.missile.velocity = np.zeros(3, dtype=float)
            return np.zeros(2, dtype=float)

        if norm(self.missile.velocity) < 1e-6:
            self.missile.velocity = self.launch_velocity.copy()

        flight_t = self.t - self.launch_delay
        if flight_t < self.scenario_boost_duration():
            self.phase = FlightPhase.BOOST
            desired_dir = self.boost_action()
            accel = desired_dir * cfg.boost_accel
            self.missile.velocity = self.missile.velocity + accel * cfg.dt
        else:
            self.phase = FlightPhase.GUIDANCE
            if action is None:
                action = np.zeros(2, dtype=float)
            action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
            forward, right, up = self.missile_frame()
            speed = norm(self.missile.velocity)
            guidance_speed = self.scenario_guidance_speed()
            speed_error = guidance_speed - speed
            axial_accel = forward * np.clip(speed_error * cfg.speed_hold_gain, -8.0, 10.0)
            lateral_accel = (right * action[0] + up * action[1]) * self.scenario_max_lateral_accel()
            accel = axial_accel + lateral_accel
            self.missile.velocity = self.missile.velocity + accel * cfg.dt
            speed = norm(self.missile.velocity)
            if speed > guidance_speed * 1.15:
                self.missile.velocity = unit(self.missile.velocity) * guidance_speed * 1.15

        self.missile.position = self.missile.position + self.missile.velocity * cfg.dt
        return action if self.phase == FlightPhase.GUIDANCE else np.zeros(2, dtype=float)

    def step(self, action: np.ndarray | None = None, use_baseline: bool = False) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.cfg
        previous_distance = self.last_distance
        previous_missile_contact = self.missile_contact_point()
        self.update_aircraft()
        if use_baseline and self.t >= self.launch_delay + self.scenario_boost_duration():
            action = self.baseline_guidance_action()
        applied_action = self.update_missile(action)

        missile_contact = self.missile_contact_point()
        target_segment_start, target_segment_end = self.target_contact_segment_endpoints()
        swept_contact, target_point = closest_points_between_segments(
            previous_missile_contact,
            missile_contact,
            target_segment_start,
            target_segment_end,
        )
        distance = norm(target_point - swept_contact)
        self.closest_distance = min(self.closest_distance, distance)
        closing_reward = (previous_distance - distance) * 0.08
        heading_reward = float(np.dot(unit(target_point - self.missile.position), unit(self.missile.velocity))) * 0.02
        control_penalty = float(np.dot(applied_action, applied_action)) * 0.01
        reward = closing_reward + heading_reward - control_penalty - 0.002

        terminated = False
        truncated = False
        status = "running"
        if distance <= self.effective_hit_radius():
            self.phase = FlightPhase.HIT
            missile_dir = unit(self.missile.velocity, np.array([0.0, 0.0, 1.0]))
            self.missile.position = swept_contact - missile_dir * MISSILE_VISUAL_NOSE_OFFSET
            missile_contact = swept_contact
            distance = norm(target_point - missile_contact)
            reward += 10.0
            terminated = True
            status = "hit"
        elif self.missile.position[2] <= 0.0:
            self.phase = FlightPhase.MISS
            reward -= 5.0
            terminated = True
            status = "ground"
        elif self.missile.position[2] > cfg.max_altitude or norm(self.missile.position[:2]) > cfg.max_range:
            self.phase = FlightPhase.MISS
            reward -= 4.0
            terminated = True
            status = "bounds"
        elif self.t >= cfg.max_time:
            self.phase = FlightPhase.MISS
            reward -= 2.0
            truncated = True
            status = "timeout"

        self.records.append(
            StepRecord(
                t=self.t,
                phase=self.phase.value,
                distance=distance,
                closest_distance=self.closest_distance,
                aircraft_x=self.aircraft.position[0],
                aircraft_y=self.aircraft.position[1],
                aircraft_z=self.aircraft.position[2],
                missile_x=self.missile.position[0],
                missile_y=self.missile.position[1],
                missile_z=self.missile.position[2],
                missile_speed=norm(self.missile.velocity),
                action_yaw=float(applied_action[0]),
                action_pitch=float(applied_action[1]),
                reward=float(reward),
            )
        )
        self.last_distance = distance
        self.t += cfg.dt

        info = {
            "status": status,
            "phase": self.phase.value,
            "distance": distance,
            "closest_distance": self.closest_distance,
            "time": self.t,
            "applied_action": applied_action.copy(),
        }
        return self.observation(), float(reward), terminated, truncated, info

    def run_baseline_episode(self, randomize: bool = False, scenario_type: str = "nominal") -> dict:
        self.reset(randomize=randomize, scenario_type=scenario_type)
        done = False
        info = {}
        while not done:
            _, _, terminated, truncated, info = self.step(use_baseline=True)
            done = terminated or truncated
        return {
            "status": info.get("status", "unknown"),
            "closest_distance": self.closest_distance,
            "time": self.t,
            "records": len(self.records),
        }

    def write_records_csv(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(StepRecord.__dataclass_fields__.keys()))
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.__dict__)
