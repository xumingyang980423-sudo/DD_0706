# Isaac Sim Two-Phase Intercept RL Prototype

Project:

```text
D:\Rocket\Missle\isaac_0703
```

Isaac Sim:

```text
E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64
```

This is an abstract simulation prototype for RL environment construction and visualization. It does not use real missile parameters.

## What Is Implemented

- Stage 2 baseline:
  - BOOST phase: rule-based launch/climb.
  - GUIDANCE phase: rule-based pursuit baseline.
  - CSV logging for hit status, closest distance, flight time, trajectory, speed, and actions.
- Stage 3 RL interface:
  - Gymnasium-style `reset/step`.
  - BOOST ignores RL action.
  - GUIDANCE uses RL action.
  - Observation: 18 dimensions.
  - Action: 2 dimensions.
  - Default Isaac Lab curriculum: scenario IDs 0-13 (`STAGE3_SCENARIO_CURRICULUM=basic14`).

## Visual Demo

Default demo:

```text
bat_scripts\visual\run_demo.bat
```

Scenario menu:

```text
bat_scripts\visual\run_visual_scenario_menu.bat
```

Single-scenario demos:

```text
bat_scripts\visual\run_demo_head_on.bat
bat_scripts\visual\run_demo_overfly_tail_chase.bat
bat_scripts\visual\run_demo_crossing_left_to_right.bat
bat_scripts\visual\run_demo_crossing_right_to_left.bat
bat_scripts\visual\run_demo_climb_escape.bat
bat_scripts\visual\run_demo_dive_escape.bat
bat_scripts\visual\run_demo_s_turn_evasion.bat
bat_scripts\visual\run_demo_double_evasion.bat
bat_scripts\visual\run_demo_late_launch.bat
bat_scripts\visual\run_demo_high_speed_pass.bat
bat_scripts\visual\run_demo_low_altitude_pass.bat
bat_scripts\visual\run_demo_far_tail_chase.bat
bat_scripts\visual\run_demo_fighter_weave_chase.bat
bat_scripts\visual\run_demo_maneuver_follow_chase.bat
bat_scripts\visual\run_demo_long_weave_tail_chase.bat
bat_scripts\visual\run_demo_extended_maneuver_follow.bat
bat_scripts\visual\run_demo_climb_dive_weave_chase.bat
bat_scripts\visual\run_demo_delayed_sustained_evasion.bat
```

Command-line example:

```powershell
cd D:\Rocket\Missle\isaac_0703
E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat .\ground_intercept_demo.py --scenario double_evasion --randomize
```

Visual demo output:

```text
logs\baseline_visual_<scenario>.csv
```

## Model Switching

The original built-in simple aircraft and missile models are still the default.

The visual demo can also use external FBX models:

```text
C:\Users\81042\Desktop\F22.fbx
C:\Users\81042\Desktop\HQ9DD.fbx
```

Run the scenario menu and answer `Y` when asked whether to use the F22/HQ9DD FBX models:

```text
bat_scripts\visual\run_visual_scenario_menu.bat
```

Direct FBX demo:

```text
bat_scripts\visual\run_demo_with_fbx_models.bat
```

The FBX files are converted and cached as USD assets under:

```text
assets\converted\F22.usd
assets\converted\HQ9DD.usd
```

If the imported model size or orientation is not visually correct, adjust:

```text
--aircraft-model-scale
--missile-model-scale
--aircraft-model-rotation
--missile-model-rotation
--aircraft-model-forward
--aircraft-model-up
--missile-model-forward
--missile-model-up
--aircraft-model-color
--missile-model-color
```

Current F22/HQ9DD defaults:

```text
aircraft rotation: 0,0,0
missile rotation: 0,0,0
aircraft forward axis: 0,0,-1
aircraft up axis: 0,1,0
missile forward axis: -1,0,0
missile up axis: 0,0,1
aircraft color: 0.05,0.25,1.0
missile color: 1.0,0.08,0.02
```

## Stage 2 Scenario Suite

The current Stage 2 suite contains 24 scenarios:

```text
head_on              Head-on frontal intercept
overfly_tail_chase   Aircraft overflies launcher, then missile tail-chases
crossing_left_to_right
crossing_right_to_left
climb_escape         Target climbs to evade, missile follows upward
dive_escape          Target dives to evade, missile follows downward
s_turn_evasion       Target performs S-turn evasion
double_evasion       Target performs two evasive turns before hit
late_launch          Delayed crossing shot after target has passed the launcher
high_speed_pass      Fast diagonal pass, high-speed pursuit
low_altitude_pass    Low-altitude terrain-hugging pass, damped pursuit
far_tail_chase       Far, high-altitude oblique chase
fighter_weave_chase  Fast fighter weave, missile follows target motion
maneuver_follow_chase Target evades, missile follows the maneuver path and catches it
long_weave_tail_chase Overfly first, aircraft makes a wide turn, missile follows the turn
extended_maneuver_follow Overfly first, fast multi-break evasion, missile follows the trail
climb_dive_weave_chase Overfly first, climb/dive/loop-like evasion, missile follows
delayed_sustained_evasion Overfly first, delayed launch, sustained high-speed evasion
cobra_pop_up_chase Overfly first, sudden nose-up pop-up climb, missile follows
circle_turn_chase Overfly first, aircraft flies a wide circle, missile follows
spiral_climb_chase Overfly first, spiral climbing evasion, missile follows
hard_reversal_chase Overfly first, hard reversal turn, missile follows
wide_snake_chase Overfly first, large-amplitude snake evasion
super_combo_chase Overfly first, combined pop-up, break turn, and sustained evasion
```

## Stage 2 Evaluation

Quick smoke test:

```text
bat_scripts\evaluation\evaluate_baseline_suite_quick.bat
```

Full evaluation:

```text
bat_scripts\evaluation\evaluate_baseline_suite.bat
```

Outputs:

```text
logs\baseline_suite_summary.csv
logs\baseline_suite_episodes.csv
logs\baseline_suite_failures.csv
```

Latest full evaluation result:

```text
episodes=1460
hits=1460
hit_rate=1.000
```

The 15-18 extension scenarios emphasize aircraft-driven, overfly-first pursuit.
The aircraft passes the ground launcher before missile launch, then accelerates
into richer maneuvers. The missile now follows a delayed aircraft trail using
trail-point correction plus trail-tangent velocity, so the missile path is
driven by the aircraft path instead of orbiting on its own. These cases include
a wide aircraft turn, multi-break evasions, climb/dive combinations, loop-like
altitude arcs, and longer delayed-launch pursuit before the rear-fuselage hit.
Scenarios 19-24 extend this into a higher-maneuver set: sudden nose-up pop-up,
wide circle, spiral climb, hard reversal, large snake, and combined super
maneuver pursuit.

Latest average intercept times for scenarios 15-18:

```text
long_weave_tail_chase       23.31s
extended_maneuver_follow    31.23s
climb_dive_weave_chase      32.42s
delayed_sustained_evasion   30.50s
```

Latest average intercept times for scenarios 19-24:

```text
cobra_pop_up_chase          30.52s
circle_turn_chase           37.13s
spiral_climb_chase          38.48s
hard_reversal_chase         32.73s
wide_snake_chase            37.07s
super_combo_chase           41.07s
```

## Stage 3 RL Environment Check

```text
bat_scripts\training\validate_rl_env.bat
```

Main class:

```python
MissileInterceptEnv
```

Files:

```text
intercept_core.py          Core dynamics, scenarios, baseline, reward, CSV logging
ground_intercept_demo.py   Isaac Sim visualization
evaluate_baseline_suite.py Stage 2 multi-scenario evaluation
rl_env.py                  Gymnasium-style RL environment
validate_rl_env.py         RL environment smoke test
```

## Batch Script Layout

All Windows launcher scripts are grouped under:

```text
bat_scripts
```

Categories:

```text
bat_scripts\visual          Isaac Sim visual scenario demos
bat_scripts\evaluation      Stage-2 baseline and legacy policy evaluation
bat_scripts\training        Isaac Lab GPU training launchers and smoke tests
bat_scripts\legacy_sb3_cpu  Older SB3 CPU baseline launchers
```

For Stage-3 training, prefer the explicit Isaac Lab GPU command in
`sandbox\isaaclab_gpu\README.md` when changing parameters such as `num_envs`,
`device`, or `max_iterations`.
