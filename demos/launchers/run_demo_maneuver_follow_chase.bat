@echo off
cd /d "%~dp0..\.."
"E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat" ".\demos\ground_intercept_demo.py" --scenario maneuver_follow_chase --randomize --hold-frames 7200

