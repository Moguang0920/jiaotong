@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title 智慧交通视觉感知系统

if not exist ".venv\Scripts\python.exe" (
    echo [错误] 尚未安装 Python 环境，请先运行 01_一键安装环境_自动GPU.bat。
    pause
    exit /b 1
)

if not exist "node_modules\electron" (
    echo [错误] 尚未安装 Electron 依赖，请先运行 01_一键安装环境_自动GPU.bat。
    pause
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
set "TRAFFIC_BACKEND_PORT=8765"

call npm start
set "CODE=%ERRORLEVEL%"
if not "%CODE%"=="0" (
    echo.
    echo 项目启动失败，错误码：%CODE%
    pause
)
exit /b %CODE%
