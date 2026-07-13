@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title 智慧交通视觉感知系统 - CPU环境安装

set "PYEXE="
for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"
if not defined PYEXE for /f "usebackq delims=" %%P in (`py -3.13 -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"

if not defined PYEXE (
    echo 请先运行 01_一键安装环境_自动GPU.bat，让它安装 Python 3.12。
    pause
    exit /b 1
)

"%PYEXE%" tools\install_environment.py --mode cpu
set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%
