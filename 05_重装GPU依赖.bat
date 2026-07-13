@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title 智慧交通视觉感知系统 - 重装GPU依赖

for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"
if not defined PYEXE for /f "usebackq delims=" %%P in (`py -3.13 -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"

if not defined PYEXE (
    echo 找不到系统 Python 3.12/3.13。
    pause
    exit /b 1
)

"%PYEXE%" tools\install_environment.py --mode gpu
set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%
