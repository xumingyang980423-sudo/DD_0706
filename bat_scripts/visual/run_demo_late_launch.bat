@echo off
cd /d "%~dp0..\.."
"E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat" ".\ground_intercept_demo.py" --scenario late_launch --randomize --hold-frames 7200
pause


