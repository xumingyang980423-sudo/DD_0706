# Batch Script Index

All Windows launcher scripts are grouped here. The Python source files,
training configs, assets, logs, and Isaac Lab sandbox stay in their original
locations.

## visual

Isaac Sim stage-2 visual demos.

Main entry:

```text
visual\run_visual_scenario_menu.bat
```

Direct scenario launchers:

```text
visual\run_demo_head_on.bat
visual\run_demo_overfly_tail_chase.bat
visual\run_demo_maneuver_follow_chase.bat
visual\run_demo_long_weave_tail_chase.bat
visual\run_demo_extended_maneuver_follow.bat
visual\run_demo_climb_dive_weave_chase.bat
visual\run_demo_delayed_sustained_evasion.bat
```

## evaluation

Stage-2 baseline evaluation and older policy evaluation wrappers.

```text
evaluation\evaluate_baseline_suite_quick.bat
evaluation\evaluate_baseline_suite.bat
evaluation\evaluate_rl_policy.bat
```

## training

Stage-3 Isaac Lab GPU training launchers.

```text
training\run_isaaclab_gpu_smoke.bat
training\run_isaaclab_gpu_validate.bat
training\run_isaaclab_gpu_train.bat
training\train_parallel_rl_smoke.bat
training\train_parallel_rl.bat
```

For parameter changes, prefer running the explicit command from:

```text
sandbox\isaaclab_gpu\README.md
```

## legacy_sb3_cpu

Older Stable-Baselines3 CPU baseline launchers. These are kept only for
comparison and are not the main Isaac Lab GPU training path.
