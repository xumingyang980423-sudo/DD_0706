@echo off
cd /d "%~dp0..\.."
powershell -ExecutionPolicy Bypass -File "%~dp0run_visual_scenario_menu.ps1"
pause
