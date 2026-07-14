PaddleOCR GPU 独立子进程 V6

架构
----
主后端 .venv：
- Electron/FastAPI
- PyTorch CUDA DLL preload
- ONNX Runtime CUDA
- best(1).onnx / hearmap.onnx / stop.onnx / normal.onnx
- 视频读取、跟踪、投票、白名单、道路异常

OCR 子进程 .venv_ocr：
- paddlepaddle-gpu
- PaddleOCR PP-OCRv6_tiny_rec
- device=gpu:0

主后端绝不 import paddle/paddleocr。车牌裁剪图通过 stdin/stdout JSON
协议发送给 OCR 子进程，OCR 文字和置信度返回主后端继续执行原来的
多帧投票、格式过滤和白名单逻辑。

替换和运行
----------
1. 完全关闭 Electron 与所有 Python：
   taskkill /F /IM electron.exe
   taskkill /F /IM python.exe

2. 解压本包到项目根目录并覆盖同名文件。

3. 运行：
   setup_paddleocr_gpu_subprocess.bat

4. 必须看到：
   [ONNX 主进程 OK]
   [PaddleOCR 子进程 OK]
   [SUCCESS] ONNX CUDA 与 PaddleOCR GPU 已在不同进程同时通过。

5. 启动：
   npm start

正常后端日志
------------
Starting PaddleOCR GPU subprocess...
OCR subprocess stderr: Loading PP-OCRv6_tiny_rec on gpu:0
ocr_ready=true
ocr_device=gpu:0
ocr_process_pid=...

说明
----
第一次执行 setup 脚本时会创建 .venv_ocr 并下载 PaddleOCR 模型。
以后 Electron 切换模型不会重新连接视频。OCR 子进程保持存活，只有
后端退出时才关闭。

后端启动时 ONNX 主模型和 OCR GPU 子进程会并行预加载；模型管理页无需先启动视频即可显示 OCR 已加载。
