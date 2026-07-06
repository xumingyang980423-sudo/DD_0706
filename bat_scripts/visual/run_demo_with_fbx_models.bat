@echo off
cd /d "%~dp0..\.."
"E:\Issac_sim\isaac-sim-standalone-5.1.0-windows-x86_64\python.bat" ".\ground_intercept_demo.py" --scenario maneuver_follow_chase --randomize --hold-frames 7200 --aircraft-model "C:\Users\81042\Desktop\F22.fbx" --missile-model "C:\Users\81042\Desktop\HQ9DD.fbx" --aircraft-model-scale=1,1,1 --missile-model-scale=1,1,1 --aircraft-model-rotation=0,0,0 --missile-model-rotation=0,0,0 --aircraft-model-forward=0,0,-1 --aircraft-model-up=0,1,0 --missile-model-forward=-1,0,0 --missile-model-up=0,0,1 --aircraft-model-color=0.05,0.25,1.0 --missile-model-color=1.0,0.08,0.02

