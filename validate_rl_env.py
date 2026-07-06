from __future__ import annotations

import numpy as np

from rl_env import MissileInterceptEnv


def main() -> None:
    env = MissileInterceptEnv(randomize=False)
    obs, info = env.reset(seed=3, options={"randomize": False})
    print(f"reset_obs_shape={obs.shape}")
    print(f"reset_phase={info['phase']}")

    total_reward = 0.0
    done = False
    steps = 0
    last_info = info

    while not done and steps < 2000:
        if last_info.get("phase") == "GUIDANCE":
            # Smoke-test action: mild upward turn. Training will replace this.
            action = np.array([0.0, 0.25], dtype=np.float32)
        else:
            action = np.zeros(2, dtype=np.float32)
        obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1

    print(f"steps={steps}")
    print(f"final_status={last_info.get('status')}")
    print(f"final_phase={last_info.get('phase')}")
    print(f"closest_distance={last_info.get('closest_distance'):.3f}")
    print(f"total_reward={total_reward:.3f}")
    print(f"final_obs_shape={obs.shape}")

    obs, info = env.reset(seed=3, options={"randomize": False})
    done = False
    steps = 0
    last_info = info
    while not done and steps < 2000:
        if env.scenario.phase.value == "GUIDANCE":
            action = env.scenario.baseline_guidance_action().astype(np.float32)
        else:
            action = np.zeros(2, dtype=np.float32)
        obs, _, terminated, truncated, last_info = env.step(action)
        done = terminated or truncated
        steps += 1

    print(f"baseline_policy_steps={steps}")
    print(f"baseline_policy_status={last_info.get('status')}")
    print(f"baseline_policy_closest_distance={last_info.get('closest_distance'):.3f}")


if __name__ == "__main__":
    main()
