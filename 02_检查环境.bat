@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title 智慧交通视觉感知系统 - 环境检查

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 尚未创建 .venv，请先运行 01_一键安装环境_自动GPU.bat。
    pause
    exit /b 1
)

".venv\Scripts\python.exe" tools\verify_environment.py
set "CODE=%ERRORLEVEL%"
echo.
pause
exit /b %CODE%
