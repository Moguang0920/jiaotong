Jiaotong-gpt 道路异常实时集成替换包
版本：2026-07-14 LIVE-BASELINE-INTEGRATION-V1

一、替换方法
1. 关闭 Electron 和 Python 后端。
2. 备份原项目。
3. 将本压缩包直接解压到 Jiaotong-gpt 项目根目录。
4. 出现同名文件时选择“全部覆盖”。
5. 新增文件 backend/runtime_road_anomaly_detector.py 必须保留。

二、集成后的 normal.onnx 模式流程
1. 选择“道路异常 · normal.onnx”。
2. 选择本地视频或手机视频流作为当前来源。
3. 点击“选择四点道路区域”，严格点击四个点并确认。
4. 点击启动本地视频或连接手机视频。
5. 后端不重新连接视频，而是从同一个最新帧缓存同时运行：
   - 原连续车道线检测；
   - normal.onnx 车辆检测与车辆 Mask；
   - 当前视频源正常道路多基准采集；
   - 多基准 ORB/Homography 道路异常检测。
6. 前端会显示“正常基准采集中 0/12”。此阶段道路中不要放异常物品。
7. 采集达到 12 张后自动进入异常检测，不需要重新启动。
8. 摄像头位置明显变化时，点击“重新采集正常基准”；视频连接不会断开。

三、基准数据位置
runtime_data/road_anomaly/live_baseline/

该目录中的图片全部来自当前实际运行的视频源，不使用测试阶段的
road_anomaly_data/camera_01/reference_bank 作为答辩现场基准。

四、同一 ROI
车道线和道路异常检测严格共用前端确认的同一组四点 ROI。
前端 Canvas 会同时绘制：
- 黄色四点道路区域；
- 原连续车道线；
- 橙色异常候选框；
- 红色确认异常框；
- normal.onnx 识别出的道路车辆框。

五、自检
在项目根目录执行：
python verify_road_anomaly_integration.py

六、启动
保持原项目启动方式不变。也可以直接运行：
python backend/plate_runtime_backend.py
npm start


【2026-07-14 启动修复 V2】
1. 修复 Electron 后端先加载 ONNX CUDA DLL、随后 PaddleOCR 间接导入 Torch 时出现 WinError 127。
2. 正确顺序调整为：PyTorch CUDA -> ONNX Runtime -> PaddleOCR。
3. PaddleOCR 仅在车牌识别模式初始化一次；道路异常/热力/禁停模式不再重复初始化 PaddleX。
4. 修复 PDX has already been initialized 重复初始化错误。
