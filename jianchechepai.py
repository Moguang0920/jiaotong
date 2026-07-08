import os
import re
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from paddleocr import TextRecognition


def cv_imread_chinese_path(img_path):
    """
    解决 Windows 下 cv2.imread 读取中文路径失败的问题。
    """
    try:
        img_bytes = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def preprocess_plate(img):
    """
    简单预处理车牌图。
    这里不要做太重的处理，先保证能跑通。
    """
    if img is None:
        return None

    h, w = img.shape[:2]

    # 如果 YOLO 裁剪出来的车牌太小，适当放大，方便识别
    if w < 160:
        scale = 160 / max(w, 1)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return img


def extract_text_recognition_result(output):
    """
    兼容 PaddleOCR 3.x TextRecognition 的返回格式。
    官方返回的 Result 对象中通常包含：
    {
        "res": {
            "rec_text": "...",
            "rec_score": ...
        }
    }
    """
    if not output:
        return "", 0.0

    res_obj = output[0]

    # PaddleOCR 3.x Result 对象一般有 json 属性
    data = getattr(res_obj, "json", None)

    if callable(data):
        data = data()

    if isinstance(data, dict):
        inner = data.get("res", data)

        # TextRecognition 模块：rec_text / rec_score
        if "rec_text" in inner:
            return inner.get("rec_text", ""), float(inner.get("rec_score", 0.0))

        # 完整 OCR 流程：rec_texts / rec_scores
        if "rec_texts" in inner:
            texts = inner.get("rec_texts", [])
            scores = inner.get("rec_scores", [])
            if texts:
                text = "".join([str(t) for t in texts if t])
                score = float(max(scores)) if len(scores) else 0.0
                return text, score

    # 兜底：直接转字符串，方便你看到原始结构
    return str(res_obj), 0.0


def clean_plate_text(text):
    """
    简单清洗车牌字符：
    保留中文、省份简称、英文字母、数字。
    """
    if not text:
        return ""

    text = text.upper()
    text = re.sub(r"[^0-9A-Z\u4e00-\u9fa5]", "", text)
    return text


def main():
    root = tk.Tk()
    root.withdraw()

    print("等待选择文件夹...")
    folder_path = filedialog.askdirectory(title="请选择存放【YOLO裁剪后车牌图】的文件夹")

    if not folder_path:
        print("未选择任何文件夹，程序退出。")
        return

    print(f"已选择文件夹: {folder_path}")

    print("正在加载车牌文字识别模型，请稍候...")

    # 这是纯文字识别模块，不会再加载 det 检测模型
    # 默认模型是 PP-OCRv6_medium_rec，准确率较高但稍重
    # 如果你想更轻，可以后面尝试 model_name="PP-OCRv6_tiny_rec"
    recognizer = TextRecognition()

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(valid_extensions)
    ]

    if not image_files:
        print("❌ 文件夹中没有找到图片文件！")
        return

    print(f"✅ 共找到 {len(image_files)} 张图片，开始识别：")
    print("=" * 70)

    for img_name in image_files:
        img_path = os.path.join(folder_path, img_name)

        if not os.path.exists(img_path):
            print(f"📄 文件: {img_name:<36} | ❌ 文件不存在")
            continue

        if os.path.getsize(img_path) == 0:
            print(f"📄 文件: {img_name:<36} | ❌ 文件大小为 0，可能是坏图")
            continue

        img = cv_imread_chinese_path(img_path)

        if img is None:
            print(f"📄 文件: {img_name:<36} | ❌ 图片读取失败")
            continue

        img = preprocess_plate(img)

        try:
            output = recognizer.predict(input=img, batch_size=1)
            plate_text, confidence = extract_text_recognition_result(output)
            plate_text = clean_plate_text(plate_text)

            if plate_text:
                print(
                    f"📄 文件: {img_name:<36} | 🚘 车牌号: {plate_text:<12} | 🎯 置信度: {confidence:.4f}"
                )
            else:
                print(f"📄 文件: {img_name:<36} | ⚠️ 未识别出有效字符")

        except Exception as e:
            print(f"📄 文件: {img_name:<36} | ❌ 识别报错: {e}")

    print("=" * 70)
    print("🎉 全部遍历识别完成！")


if __name__ == "__main__":
    main()
