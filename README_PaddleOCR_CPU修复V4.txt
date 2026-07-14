PaddleOCR CPU 修复 V4

问题原因
--------
项目主后端已经加载 PyTorch CUDA 和 ONNX Runtime CUDA。
当前安装的 paddlepaddle-gpu 又尝试加载另一套 cuDNN 9，
导致 cudnn_cnn64_9.dll / WinError 127。

修复策略
--------
1. best(1).onnx、normal.onnx、hearmap.onnx、stop.onnx 继续使用 CUDA。
2. PP-OCRv6_tiny_rec 改用 paddlepaddle CPU。
3. OCR 本身仍然是异步线程，并受到每秒次数限制，不阻塞视频显示。
4. 环境安装器以后也固定安装 Paddle CPU，避免问题再次出现。

替换与执行
----------
1. 完全关闭 Electron 和后端。
2. 将压缩包内容覆盖到项目根目录。
3. 双击 fix_paddleocr_cpu.bat。
4. 脚本最后看到 [SUCCESS] PaddleOCR model loaded on CPU。
5. 执行 npm start。
6. 切换到车牌识别模式，模型管理页应显示 PaddleOCR 已加载 / cpu。

说明
----
第一次加载 PP-OCRv6_tiny_rec 可能需要下载约数 MB 模型文件。
脚本已将模型源默认设置为 BOS。
