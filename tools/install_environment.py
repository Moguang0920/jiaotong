# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = ROOT / ".venv"
OCR_VENV_DIR = ROOT / ".venv_ocr"
LOG_DIR = ROOT / "install_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"install_{dt.datetime.now():%Y%m%d_%H%M%S}.log"

ORT_VERSION = "1.27.0"
PADDLE_VERSION = "3.3.0"


class InstallError(RuntimeError):
    pass


def log(message: str) -> None:
    text = str(message)
    print(text, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def run(
    command: Iterable[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    cmd = [str(item) for item in command]
    log("\n>>> " + subprocess.list2cmdline(cmd))
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd or ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        line = line.rstrip("\r\n")
        output_lines.append(line)
        log(line)
    code = process.wait()
    result = subprocess.CompletedProcess(cmd, code, "\n".join(output_lines), "")
    if code != 0 and check:
        raise InstallError(f"命令执行失败，退出码 {code}: {subprocess.list2cmdline(cmd)}")
    return result


def require_supported_python() -> None:
    version = sys.version_info[:2]
    if version not in {(3, 12), (3, 13)}:
        raise InstallError(
            f"当前安装器由 Python {sys.version.split()[0]} 启动。"
            "本项目要求 Python 3.12 或 3.13，请运行根目录的 01_一键安装环境_自动GPU.bat。"
        )
    if sys.maxsize <= 2**32:
        raise InstallError("检测到 32 位 Python，本项目必须使用 64 位 Python。")


def venv_python() -> Path:
    return VENV_DIR / "Scripts" / "python.exe"


def create_venv() -> Path:
    target = venv_python()
    if not target.exists():
        log(f"创建独立虚拟环境：{VENV_DIR}")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    else:
        log(f"复用现有虚拟环境：{VENV_DIR}")
    return target


def pip(python_exe: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run([str(python_exe), "-m", "pip", *args], check=check)


def detect_nvidia() -> tuple[bool, Optional[float], str]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        common = (
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation"
            / "NVSMI"
            / "nvidia-smi.exe"
        )
        if common.exists():
            nvidia_smi = str(common)
    if not nvidia_smi:
        return False, None, "未找到 nvidia-smi"

    result = run([nvidia_smi], check=False)
    if result.returncode != 0:
        return False, None, "nvidia-smi 执行失败"

    match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", result.stdout or "")
    cuda_version = float(match.group(1)) if match else None
    return True, cuda_version, result.stdout or ""


def choose_paddle_index(cuda_version: Optional[float]) -> str:
    if cuda_version is None:
        return "cu126"
    if cuda_version >= 13.0:
        return "cu130"
    if cuda_version >= 12.9:
        return "cu129"
    if cuda_version >= 12.6:
        return "cu126"
    return "cu118"


def uninstall_conflicting_engines(python_exe: Path) -> None:
    pip(
        python_exe,
        "uninstall",
        "-y",
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-directml",
        "paddlepaddle",
        "paddlepaddle-gpu",
        check=False,
    )


def install_onnx_gpu(python_exe: Path) -> bool:
    log("安装 ONNX Runtime GPU + CUDA/cuDNN Python 运行库。")
    first = pip(
        python_exe,
        "install",
        "--upgrade",
        "--no-cache-dir",
        f"onnxruntime-gpu[cuda,cudnn]=={ORT_VERSION}",
        check=False,
    )
    if first.returncode == 0:
        return True

    log("ONNX GPU 首次安装失败，清理 pip 缓存后重试。")
    pip(python_exe, "cache", "purge", check=False)
    second = pip(
        python_exe,
        "install",
        "--upgrade",
        "--no-cache-dir",
        f"onnxruntime-gpu[cuda,cudnn]=={ORT_VERSION}",
        check=False,
    )
    if second.returncode == 0:
        return True

    log("附带 CUDA/cuDNN 的安装仍失败，尝试普通 onnxruntime-gpu。")
    third = pip(
        python_exe,
        "install",
        "--upgrade",
        "--no-cache-dir",
        f"onnxruntime-gpu=={ORT_VERSION}",
        check=False,
    )
    return third.returncode == 0


def install_onnx_cpu(python_exe: Path) -> None:
    pip(
        python_exe,
        "install",
        "--upgrade",
        "--no-cache-dir",
        f"onnxruntime=={ORT_VERSION}",
    )


def install_paddle_gpu(
    python_exe: Path,
    cuda_version: Optional[float],
) -> tuple[bool, str]:
    preferred = choose_paddle_index(cuda_version)
    candidates = [preferred]
    if preferred != "cu126":
        candidates.append("cu126")
    if "cu118" not in candidates:
        candidates.append("cu118")

    for cuda_tag in candidates:
        index_url = f"https://www.paddlepaddle.org.cn/packages/stable/{cuda_tag}/"
        log(f"尝试安装 PaddlePaddle GPU {PADDLE_VERSION}：{cuda_tag}")
        result = pip(
            python_exe,
            "install",
            "--upgrade",
            "--no-cache-dir",
            f"paddlepaddle-gpu=={PADDLE_VERSION}",
            "-i",
            index_url,
            check=False,
        )
        if result.returncode == 0:
            return True, cuda_tag
    return False, ""


def install_paddle_cpu(python_exe: Path) -> None:
    pip(
        python_exe,
        "install",
        "--upgrade",
        "--no-cache-dir",
        f"paddlepaddle=={PADDLE_VERSION}",
        "-i",
        "https://www.paddlepaddle.org.cn/packages/stable/cpu/",
    )



def install_ocr_gpu_subprocess_env(
    main_python: Path,
    cuda_version: Optional[float],
) -> tuple[bool, str]:
    """创建只包含 PaddleOCR GPU 的独立虚拟环境。"""
    log(f"创建 PaddleOCR GPU 独立环境：{OCR_VENV_DIR}")
    if not OCR_VENV_DIR.exists():
        run([str(main_python), "-m", "venv", str(OCR_VENV_DIR)])

    ocr_python = OCR_VENV_DIR / "Scripts" / "python.exe"
    if not ocr_python.exists():
        return False, ""

    pip(
        ocr_python,
        "install",
        "--upgrade",
        "pip",
        "wheel",
        "setuptools<82",
    )
    pip(
        ocr_python,
        "uninstall",
        "-y",
        "paddlepaddle",
        "paddlepaddle-gpu",
        check=False,
    )

    paddle_ok, paddle_tag = install_paddle_gpu(
        ocr_python,
        cuda_version,
    )
    if not paddle_ok:
        return False, ""

    pip(
        ocr_python,
        "install",
        "--upgrade",
        "--no-cache-dir",
        "numpy==1.26.4",
        "opencv-python-headless==4.10.0.84",
        "paddleocr>=3.3.0,<4",
    )

    test = run(
        [
            str(ocr_python),
            "-c",
            (
                "import paddle;"
                "print('paddle=',paddle.__version__);"
                "print('cuda=',paddle.device.is_compiled_with_cuda());"
                "print('count=',paddle.device.cuda.device_count());"
                "paddle.set_device('gpu:0');"
                "print('device=',paddle.get_device())"
            ),
        ],
        check=False,
    )
    return test.returncode == 0, paddle_tag



def find_npm() -> Optional[str]:
    found = shutil.which("npm")
    if found:
        return found
    common = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npm.cmd"
    return str(common) if common.exists() else None


def install_node_dependencies() -> None:
    npm = find_npm()
    if not npm:
        raise InstallError(
            "未找到 npm。请重新运行 01_一键安装环境_自动GPU.bat，"
            "或先安装 Node.js LTS。"
        )
    result = run([npm, "ci"], cwd=ROOT, check=False)
    if result.returncode != 0:
        log("npm ci 失败，改用 npm install。")
        run([npm, "install"], cwd=ROOT)


def install_requirements(python_exe: Path) -> None:
    pip(
        python_exe,
        "install",
        "--upgrade",
        "--no-cache-dir",
        "-r",
        str(ROOT / "requirements.txt"),
    )


def write_install_state(
    mode: str,
    nvidia: bool,
    cuda: Optional[float],
    paddle_tag: str,
) -> None:
    state = {
        "installed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "requested_mode": mode,
        "nvidia_detected": nvidia,
        "driver_reported_cuda": cuda,
        "paddle_cuda_tag": paddle_tag,
        "python": sys.version,
        "installer_log": str(LOG_PATH),
    }
    runtime_dir = ROOT / "runtime_data"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "install_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--skip-node", action="store_true")
    args = parser.parse_args()

    require_supported_python()
    log("=" * 72)
    log("智慧交通视觉感知系统：环境安装器")
    log(f"系统 Python：{sys.executable}")
    log(f"Python 版本：{sys.version}")
    log(f"安装日志：{LOG_PATH}")
    log("=" * 72)

    python_exe = create_venv()
    pip(python_exe, "install", "--upgrade", "pip", "setuptools", "wheel")
    uninstall_conflicting_engines(python_exe)

    nvidia, cuda_version, _ = detect_nvidia()
    use_gpu = args.mode == "gpu" or (args.mode == "auto" and nvidia)
    paddle_tag = ""

    if use_gpu and not nvidia:
        log("要求 GPU 模式，但没有检测到 NVIDIA 驱动；自动切换 CPU。")
        use_gpu = False

    if use_gpu:
        log(f"检测到 NVIDIA 显卡，驱动报告 CUDA：{cuda_version or '未知'}")
        if not install_onnx_gpu(python_exe):
            log("ONNX Runtime GPU 安装失败，自动回退 CPU 版。")
            uninstall_conflicting_engines(python_exe)
            install_onnx_cpu(python_exe)
            install_paddle_cpu(python_exe)
            use_gpu = False
            paddle_tag = "cpu"
        else:
            log(
                "ONNX Runtime 在主 .venv 使用 GPU；"
                "PaddleOCR 在 .venv_ocr 独立进程使用 GPU。"
            )
            ocr_ok, paddle_tag = install_ocr_gpu_subprocess_env(
                python_exe,
                cuda_version,
            )
            if not ocr_ok:
                raise RuntimeError(
                    "PaddleOCR GPU 独立环境安装或验证失败。"
                )
    else:
        log("使用 CPU 安装模式。")
        install_onnx_cpu(python_exe)
        install_paddle_cpu(python_exe)
        paddle_tag = "cpu"

    install_requirements(python_exe)

    if not args.skip_node:
        install_node_dependencies()

    for folder in [
        ROOT / "runtime_data",
        ROOT / "cache" / "trt_engine",
        ROOT / "models",
    ]:
        folder.mkdir(parents=True, exist_ok=True)

    write_install_state(args.mode, nvidia, cuda_version, paddle_tag)

    log("\n开始执行环境检查。")
    verify = run(
        [str(python_exe), str(ROOT / "tools" / "verify_environment.py")],
        check=False,
    )

    log("\n" + "=" * 72)
    if verify.returncode == 0:
        log("安装完成。接下来双击 03_一键启动项目.bat。")
    else:
        log("依赖已安装，但环境检查发现问题，请查看检查结果和日志。")
    log("=" * 72)
    return verify.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("用户取消安装。")
        raise SystemExit(130)
    except Exception as exc:
        log(f"\n[安装失败] {type(exc).__name__}: {exc}")
        log(f"详细日志：{LOG_PATH}")
        raise SystemExit(1)
