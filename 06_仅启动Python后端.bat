@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title 智慧交通 Python 后端

if not exist ".venv\Scripts\python.exe" (
    echo 请先运行 01_一键安装环境_自动GPU.bat。
    pause
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
".venv\Scripts\python.exe" backend\plate_runtime_backend.py
pause
