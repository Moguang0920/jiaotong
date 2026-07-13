智慧交通视觉感知系统：新电脑迁移说明
====================================

一、为什么不用电脑里的 Python 3.14
当前 PaddlePaddle/PaddleOCR 的正式支持范围尚未包含 Python 3.14。
本包会自动安装 Python 3.12，并创建项目自己的 .venv。
电脑原来的 Python 3.14 不会被删除，也不会被修改。

二、迁移步骤
1. 解压整个 Jiaotong-gpt 文件夹。
2. 从旧电脑复制 ONNX 模型到 models 文件夹：
   - best(1).onnx 或 best.onnx
   - hearmap.onnx 或 heatmap.onnx
   - stop.onnx
   - normal.onnx
3. 双击：01_一键安装环境_自动GPU.bat
4. 双击：02_检查环境.bat
5. 双击：04_预下载OCR模型.bat
6. 双击：03_一键启动项目.bat

三、安装器自动执行
- 自动寻找 Python 3.12/3.13。
- 如果只有 Python 3.14，通过 winget 安装 Python 3.12。
- 创建 .venv 独立环境。
- 安装 OpenCV、FastAPI、PaddleOCR、Pillow、NumPy 等依赖。
- 检测 NVIDIA 显卡和驱动报告的 CUDA 版本。
- 安装 ONNX Runtime GPU 和 CUDA/cuDNN Python 运行库。
- 根据 CUDA 版本选择 PaddlePaddle GPU 安装源。
- GPU 安装失败时自动回退 CPU。
- 缺少 Node.js 时自动安装 Node.js LTS。
- 通过 npm ci 安装 Electron。
- 自动生成环境检查报告和安装日志。

四、首次 OCR
PaddleOCR 第一次运行会联网下载 PP-OCRv6_tiny_rec。
建议先运行 04_预下载OCR模型.bat。
若默认模型源失败，可运行 03B_一键启动项目_BOS模型源.bat。

五、判断 GPU 是否生效
运行 02_检查环境.bat：
- ONNX Runtime Providers 应包含 CUDAExecutionProvider。
- Paddle CUDA 编译为 True 时，OCR 也可使用 GPU。
即使 Paddle OCR 回退 CPU，只要 ONNX 有 CUDAExecutionProvider，
YOLO 检测仍然运行在 GPU 上。

六、迁移包功能版本
- 包含 OCR 高频识别和三票稳定逻辑。
- 包含队列释放、防串牌和候选记录去重修复。
- 车牌识别模式只显示车牌号，不显示车辆/车牌跟踪码。
- 车辆检测、热力图、禁停模式的车辆跟踪逻辑保留。
