@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title PaddleOCR 模型预下载

if not exist ".venv\Scripts\python.exe" (
    echo 请先运行 01_一键安装环境_自动GPU.bat。
    pause
    exit /b 1
)

set "PADDLE_PDX_MODEL_SOURCE=BOS"
".venv\Scripts\python.exe" tools\preload_ocr_model.py
set "CODE=%ERRORLEVEL%"
pause
exit /b %CODE%
