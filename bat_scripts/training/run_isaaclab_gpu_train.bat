@echo off
setlocal
cd /d "%~dp0..\..\sandbox\isaaclab_gpu"
set TORCH_COMPILE_DISABLE=1
set PYTHONPATH=%CD%\source\isaaclab_stage3;%PYTHONPATH%
set STAGE3_SCENARIO_CURRICULUM=basic14
set STAGE3_RANDOMIZATION_MODE=train
set STAGE3_FIXED_SCENARIO_ID=
"E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat" "%CD%\scripts\train_isaaclab_intercept_rlgames.py" --task Isaac-Stage3-Intercept-Direct-v0 --num_envs 4096 --device cuda:0 --headless --max_iterations 300
endlocal
pause
