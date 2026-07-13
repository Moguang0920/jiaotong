@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title 智慧交通视觉感知系统 - BOS模型源

set "PADDLE_PDX_MODEL_SOURCE=BOS"
call "03_一键启动项目.bat"
exit /b %ERRORLEVEL%
