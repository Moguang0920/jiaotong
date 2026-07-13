@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title 智慧交通视觉感知系统 - 一键安装环境

echo ================================================================
echo 智慧交通视觉感知系统 - 一键安装环境
echo 将使用 Python 3.12/3.13 独立虚拟环境，不使用 Python 3.14
echo ================================================================
echo.

set "PYEXE="

for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"
if not defined PYEXE (
    for /f "usebackq delims=" %%P in (`py -3.13 -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"
)
if not defined PYEXE (
    for /f "usebackq delims=" %%P in (`python -c "import sys;print(sys.executable if sys.version_info[:2] in [(3,12),(3,13)] else '')" 2^>nul`) do set "PYEXE=%%P"
)

if not defined PYEXE (
    echo 没有检测到 Python 3.12/3.13，准备使用 winget 安装 Python 3.12 x64...
    where winget >nul 2>nul
    if errorlevel 1 (
        echo.
        echo [失败] 当前系统没有 winget，无法自动安装 Python 3.12。
        echo 请先安装 64 位 Python 3.12，并勾选 Python Launcher。
        pause
        exit /b 1
    )
    winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
    if errorlevel 1 (
        echo [失败] Python 3.12 自动安装失败。
        pause
        exit /b 1
    )

    for /f "usebackq delims=" %%P in (`py -3.12 -c "import sys;print(sys.executable)" 2^>nul`) do set "PYEXE=%%P"
    if not defined PYEXE if exist "%LocalAppData%\Programs\Python\Python312\python.exe" set "PYEXE=%LocalAppData%\Programs\Python\Python312\python.exe"
)

if not defined PYEXE (
    echo [失败] 安装后仍然无法找到 Python 3.12。
    pause
    exit /b 1
)

echo 使用 Python：
"%PYEXE%" --version
echo.

where npm >nul 2>nul
if errorlevel 1 (
    echo 没有检测到 Node.js/npm，准备使用 winget 安装 Node.js LTS...
    where winget >nul 2>nul
    if errorlevel 1 (
        echo [失败] 当前系统没有 winget，请先安装 Node.js LTS。
        pause
        exit /b 1
    )
    winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
    set "PATH=%ProgramFiles%\nodejs;%PATH%"
)

if not exist "%ProgramFiles%\nodejs\npm.cmd" (
    where npm >nul 2>nul
    if errorlevel 1 (
        echo [失败] Node.js 安装后仍然找不到 npm，请注销或重启电脑后重新运行本 BAT。
        pause
        exit /b 1
    )
)

"%PYEXE%" tools\install_environment.py --mode auto
set "INSTALL_CODE=%ERRORLEVEL%"

echo.
if "%INSTALL_CODE%"=="0" (
    echo [完成] 环境安装和检查已完成。
    echo 请把 ONNX 模型复制到 models 文件夹，然后双击 03_一键启动项目.bat。
) else (
    echo [失败] 安装器返回错误码 %INSTALL_CODE%。
    echo 请查看 install_logs 文件夹中的最新日志。
)
echo.
pause
exit /b %INSTALL_CODE%
