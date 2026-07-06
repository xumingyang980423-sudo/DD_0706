@echo off
cd /d "%~dp0..\..\sandbox\sb3_cpu_baseline"
"E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat" ".\train_parallel_rl.py" --num-envs 4 --total-timesteps 300000 --run-name ppo_stage3_parallel
pause
