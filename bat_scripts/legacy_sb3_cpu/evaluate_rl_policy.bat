@echo off
cd /d "%~dp0..\..\sandbox\sb3_cpu_baseline"
"E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat" ".\evaluate_rl_policy.py" --model ".\logs\rl\ppo_stage3_parallel\models\ppo_final.zip" --episodes-per-scenario 20 --deterministic
pause
