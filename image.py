import datetime
import os
import time
import cv2

# 1. 填入你朋友手机的 Tailscale 虚拟 IP
video_stream_url = "http://100.70.11.30:8080/video"

print("正在试图连接朋友手机的画面...")
cap = cv2.VideoCapture(video_stream_url)

# 适度调节 OpenCV 底层接收缓冲，确保实时画面不积压
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("连接失败！请确认你的电脑是否已经运行 Tailscale 并登录了朋友的账号！")
    exit()

print("连接成功！开始初始化时间戳存储系统...")

# ==================== 【自动命名与目录防覆盖配置】 ====================
# 自动在当前运行目录下创建 dataset 文件夹
save_dir = "dataset"
os.makedirs(save_dir, exist_ok=True)

# 生成精确到秒的时间戳，例如：2026-07-06_16-25-27
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# 最终文件名，确保每次运行绝对唯一且绝不覆盖旧数据
output_filename = os.path.join(save_dir, f"traffic_train_{timestamp}.mp4")

# 动态获取手机画面的原始宽度和高度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 使用适用于 AI 训练集最通用的 mp4v 编码
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 视频标称帧率（建议和 App 中的推流设置保持一致，建议 15 或 30）
save_fps = 15.0

# 创建写入器
out = cv2.VideoWriter(
    output_filename, fourcc, save_fps, (frame_width, frame_height)
)
# ====================================================================

print(f"当前视频录制中！任务文件独立保存至: {output_filename}")
print("提示：按键盘 'q' 键即可随时停止并生成当前的训练数据包。")

frame_count = 0
start_record_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("网络画面传输中断或已关闭")
        break

    # 1. 立即写入当前帧，完成精准保存
    out.write(frame)
    frame_count += 1

    # 2. 在这里可以接入你的离线或轻量化实时模型展示
    # results = your_model(frame)

    # 3. 实时可视化，并在顶部渲染当前正在记录的数据集编号
    cv2.putText(
        frame,
        f"REC: {timestamp}.mp4 | Frames: {frame_count}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )
    cv2.imshow("Smart Traffic Dataset Recorder", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("手动停止采集，准备封装并保存视频...")
        break

# 安全释放资源，防止文件结尾损坏
out.release()
cap.release()
cv2.destroyAllWindows()

# 计算实际收流指标
elapsed_time = time.time() - start_record_time
actual_fps = frame_count / elapsed_time if elapsed_time > 0 else 0

print("\n================ 数据采集报告 ================")
print(f"数据包成功保存至 : {output_filename}")
print(f"采集总时长       : {elapsed_time:.2f} 秒")
print(f"收集总有效帧数   : {frame_count} 帧")
print(f"平均实际网络帧率 : {actual_fps:.2f} FPS")
print("==============================================")

