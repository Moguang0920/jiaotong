# -*- coding: utf-8 -*-
"""
将 reference_01.png 上已经画好的道路 ROI，
自动映射到 reference_bank 中的其他正常道路基准图。

输入：
road_anomaly_data/camera_01/
├── reference_bank/
│   ├── reference_01.png
│   ├── reference_02.png
│   └── ...
└── road_roi/
    ├── road_mask.png
    └── road_roi.json

输出：
road_anomaly_data/camera_01/
├── road_masks/
│   ├── reference_01_road_mask.png
│   ├── reference_02_road_mask.png
│   └── ...
├── road_mask_previews/
│   ├── reference_01_preview.jpg
│   ├── reference_02_preview.jpg
│   └── ...
└── road_mask_mapping.json

运行：
    python map_road_roi_to_references.py

说明：
- 使用 ORB 特征匹配 + RANSAC 仿射变换。
- 只对道路 Mask 做几何映射，不会模糊或合成原始图片。
- 映射失败的图片会标记为 failed，不会偷偷保存错误 Mask。
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


CAMERA_ROOT = Path("road_anomaly_data") / "camera_01"
REFERENCE_DIR = CAMERA_ROOT / "reference_bank"
ANCHOR_IMAGE = REFERENCE_DIR / "reference_01.png"
ANCHOR_MASK = CAMERA_ROOT / "road_roi" / "road_mask.png"

OUTPUT_MASK_DIR = CAMERA_ROOT / "road_masks"
OUTPUT_PREVIEW_DIR = CAMERA_ROOT / "road_mask_previews"
OUTPUT_JSON = CAMERA_ROOT / "road_mask_mapping.json"

# ORB 参数
ORB_FEATURES = 7000
LOWE_RATIO = 0.76
MIN_GOOD_MATCHES = 24
MIN_INLIERS = 16
MIN_INLIER_RATIO = 0.28

# 合理晃动范围保护，避免错误匹配产生离谱 Mask
MAX_TRANSLATION_X_RATIO = 0.20
MAX_TRANSLATION_Y_RATIO = 0.20
MIN_SCALE = 0.82
MAX_SCALE = 1.18
MAX_ROTATION_DEGREES = 12.0


def imread_unicode(path: Path, flags: int) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(
    path: Path,
    image: np.ndarray,
    params: Optional[List[int]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".png"

    ok, encoded = cv2.imencode(
        suffix,
        image,
        params or [],
    )

    if not ok:
        raise RuntimeError(f"图片编码失败：{path}")

    encoded.tofile(str(path))


def normalize_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    )

    return clahe.apply(gray)


def create_orb():
    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=21,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=12,
    )


def estimate_anchor_to_target(
    anchor_image: np.ndarray,
    target_image: np.ndarray,
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    计算从 anchor 坐标映射到 target 坐标的 2x3 仿射矩阵。
    """
    if anchor_image.shape[:2] != target_image.shape[:2]:
        return None, {
            "status": "failed",
            "reason": "image_size_mismatch",
        }

    anchor_gray = normalize_gray(anchor_image)
    target_gray = normalize_gray(target_image)

    orb = create_orb()

    keypoints_anchor, descriptors_anchor = orb.detectAndCompute(
        anchor_gray,
        None,
    )

    keypoints_target, descriptors_target = orb.detectAndCompute(
        target_gray,
        None,
    )

    if (
        descriptors_anchor is None
        or descriptors_target is None
        or len(keypoints_anchor) < 10
        or len(keypoints_target) < 10
    ):
        return None, {
            "status": "failed",
            "reason": "not_enough_keypoints",
            "anchor_keypoints": len(keypoints_anchor or []),
            "target_keypoints": len(keypoints_target or []),
        }

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False,
    )

    knn_matches = matcher.knnMatch(
        descriptors_anchor,
        descriptors_target,
        k=2,
    )

    good_matches = []

    for pair in knn_matches:
        if len(pair) != 2:
            continue

        first, second = pair

        if first.distance < LOWE_RATIO * second.distance:
            good_matches.append(first)

    if len(good_matches) < MIN_GOOD_MATCHES:
        return None, {
            "status": "failed",
            "reason": "not_enough_good_matches",
            "anchor_keypoints": len(keypoints_anchor),
            "target_keypoints": len(keypoints_target),
            "good_matches": len(good_matches),
        }

    anchor_points = np.float32([
        keypoints_anchor[match.queryIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    target_points = np.float32([
        keypoints_target[match.trainIdx].pt
        for match in good_matches
    ]).reshape(-1, 1, 2)

    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        anchor_points,
        target_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.5,
        maxIters=4000,
        confidence=0.995,
        refineIters=30,
    )

    if matrix is None or inlier_mask is None:
        return None, {
            "status": "failed",
            "reason": "affine_estimation_failed",
            "good_matches": len(good_matches),
        }

    inliers = int(inlier_mask.reshape(-1).sum())
    inlier_ratio = inliers / max(len(good_matches), 1)

    if (
        inliers < MIN_INLIERS
        or inlier_ratio < MIN_INLIER_RATIO
    ):
        return None, {
            "status": "failed",
            "reason": "weak_geometry_consistency",
            "good_matches": len(good_matches),
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 4),
        }

    a, b, tx = matrix[0]
    c, d, ty = matrix[1]

    scale_x = math.sqrt(a * a + c * c)
    scale_y = math.sqrt(b * b + d * d)
    average_scale = (scale_x + scale_y) / 2.0

    rotation_degrees = math.degrees(
        math.atan2(c, a)
    )

    height, width = anchor_image.shape[:2]

    transform_ok = (
        abs(tx) <= width * MAX_TRANSLATION_X_RATIO
        and abs(ty) <= height * MAX_TRANSLATION_Y_RATIO
        and MIN_SCALE <= average_scale <= MAX_SCALE
        and abs(rotation_degrees) <= MAX_ROTATION_DEGREES
    )

    diagnostics = {
        "status": "success" if transform_ok else "failed",
        "reason": None if transform_ok else "transform_out_of_range",
        "anchor_keypoints": len(keypoints_anchor),
        "target_keypoints": len(keypoints_target),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "inlier_ratio": round(inlier_ratio, 4),
        "translation_x": round(float(tx), 4),
        "translation_y": round(float(ty), 4),
        "scale": round(float(average_scale), 6),
        "rotation_degrees": round(float(rotation_degrees), 5),
        "matrix": [
            [round(float(value), 8) for value in matrix[0]],
            [round(float(value), 8) for value in matrix[1]],
        ],
    }

    if not transform_ok:
        return None, diagnostics

    return matrix.astype(np.float32), diagnostics


def build_preview(
    image: np.ndarray,
    mask: np.ndarray,
    title: str,
) -> np.ndarray:
    preview = image.copy()

    layer = preview.copy()
    layer[mask > 0] = (40, 210, 130)

    preview = cv2.addWeighted(
        layer,
        0.28,
        preview,
        0.72,
        0,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        preview,
        contours,
        -1,
        (0, 225, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        preview,
        (0, 0),
        (preview.shape[1], 48),
        (10, 17, 27),
        -1,
    )

    cv2.putText(
        preview,
        title,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )

    return preview


def save_failed_preview(
    target_image: np.ndarray,
    target_name: str,
    reason: str,
    output_path: Path,
) -> None:
    preview = target_image.copy()

    cv2.rectangle(
        preview,
        (0, 0),
        (preview.shape[1], 86),
        (25, 25, 160),
        -1,
    )

    cv2.putText(
        preview,
        f"{target_name}: MAPPING FAILED",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        preview,
        reason[:80],
        (16, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (230, 230, 255),
        2,
        cv2.LINE_AA,
    )

    imwrite_unicode(
        output_path,
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), 92],
    )


def main() -> None:
    project_root = Path(__file__).resolve().parent

    reference_dir = project_root / REFERENCE_DIR
    anchor_image_path = project_root / ANCHOR_IMAGE
    anchor_mask_path = project_root / ANCHOR_MASK

    output_mask_dir = project_root / OUTPUT_MASK_DIR
    output_preview_dir = project_root / OUTPUT_PREVIEW_DIR
    output_json_path = project_root / OUTPUT_JSON

    if not anchor_image_path.exists():
        raise FileNotFoundError(
            f"没有找到主基准图：{anchor_image_path}"
        )

    if not anchor_mask_path.exists():
        raise FileNotFoundError(
            f"没有找到你刚才画好的道路 Mask：{anchor_mask_path}"
        )

    reference_paths = sorted(
        reference_dir.glob("reference_*.png")
    )

    if not reference_paths:
        raise FileNotFoundError(
            f"基准图目录中没有图片：{reference_dir}"
        )

    anchor_image = imread_unicode(
        anchor_image_path,
        cv2.IMREAD_COLOR,
    )

    anchor_mask = imread_unicode(
        anchor_mask_path,
        cv2.IMREAD_GRAYSCALE,
    )

    if anchor_image is None:
        raise RuntimeError(
            f"无法读取主基准图：{anchor_image_path}"
        )

    if anchor_mask is None:
        raise RuntimeError(
            f"无法读取道路 Mask：{anchor_mask_path}"
        )

    if anchor_image.shape[:2] != anchor_mask.shape[:2]:
        raise RuntimeError(
            "主基准图和 road_mask.png 的尺寸不一致。"
        )

    # 规范化为纯黑白 Mask
    _, anchor_mask = cv2.threshold(
        anchor_mask,
        127,
        255,
        cv2.THRESH_BINARY,
    )

    if output_mask_dir.exists():
        shutil.rmtree(output_mask_dir)

    if output_preview_dir.exists():
        shutil.rmtree(output_preview_dir)

    output_mask_dir.mkdir(parents=True, exist_ok=True)
    output_preview_dir.mkdir(parents=True, exist_ok=True)

    records = []
    success_count = 0
    failed_count = 0

    print("=" * 78)
    print("正在把道路 ROI 映射到所有正常道路基准图")
    print(f"主基准图：{anchor_image_path}")
    print(f"基准图数量：{len(reference_paths)}")
    print("=" * 78)

    for index, target_path in enumerate(
        reference_paths,
        start=1,
    ):
        target_image = imread_unicode(
            target_path,
            cv2.IMREAD_COLOR,
        )

        if target_image is None:
            failed_count += 1
            records.append({
                "reference": target_path.name,
                "status": "failed",
                "reason": "image_read_failed",
            })
            continue

        mask_filename = (
            f"{target_path.stem}_road_mask.png"
        )

        preview_filename = (
            f"{target_path.stem}_preview.jpg"
        )

        mask_output_path = (
            output_mask_dir / mask_filename
        )

        preview_output_path = (
            output_preview_dir / preview_filename
        )

        if target_path.name == anchor_image_path.name:
            mapped_mask = anchor_mask.copy()

            diagnostics = {
                "status": "success",
                "reason": "anchor_copy",
                "matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
            }
        else:
            matrix, diagnostics = estimate_anchor_to_target(
                anchor_image,
                target_image,
            )

            if matrix is None:
                failed_count += 1

                save_failed_preview(
                    target_image=target_image,
                    target_name=target_path.name,
                    reason=str(
                        diagnostics.get(
                            "reason",
                            "unknown",
                        )
                    ),
                    output_path=preview_output_path,
                )

                records.append({
                    "reference": target_path.name,
                    "status": "failed",
                    "mask_file": None,
                    "preview_file": str(
                        Path("road_mask_previews")
                        / preview_filename
                    ).replace("\\", "/"),
                    **diagnostics,
                })

                print(
                    f"[{index}/{len(reference_paths)}] "
                    f"{target_path.name}: 映射失败 "
                    f"({diagnostics.get('reason')})"
                )
                continue

            height, width = target_image.shape[:2]

            mapped_mask = cv2.warpAffine(
                anchor_mask,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

            _, mapped_mask = cv2.threshold(
                mapped_mask,
                127,
                255,
                cv2.THRESH_BINARY,
            )

            # 清除小型锯齿孔洞，不改变整体 ROI 位置。
            kernel = np.ones((3, 3), np.uint8)

            mapped_mask = cv2.morphologyEx(
                mapped_mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=1,
            )

        imwrite_unicode(
            mask_output_path,
            mapped_mask,
        )

        preview = build_preview(
            target_image,
            mapped_mask,
            (
                f"{target_path.name} | "
                f"road ROI mapped"
            ),
        )

        imwrite_unicode(
            preview_output_path,
            preview,
            [int(cv2.IMWRITE_JPEG_QUALITY), 94],
        )

        success_count += 1

        records.append({
            "reference": target_path.name,
            "status": "success",
            "mask_file": str(
                Path("road_masks")
                / mask_filename
            ).replace("\\", "/"),
            "preview_file": str(
                Path("road_mask_previews")
                / preview_filename
            ).replace("\\", "/"),
            **diagnostics,
        })

        print(
            f"[{index}/{len(reference_paths)}] "
            f"{target_path.name}: 成功"
        )

    mapping_data = {
        "camera_id": "camera_01",
        "anchor_reference": anchor_image_path.name,
        "anchor_mask": str(
            Path("road_roi") / "road_mask.png"
        ).replace("\\", "/"),
        "reference_count": len(reference_paths),
        "success_count": success_count,
        "failed_count": failed_count,
        "method": "ORB + RANSAC affine mapping",
        "important": (
            "映射失败的基准图片没有生成道路 Mask，"
            "需要之后手动补画或删除该基准图。"
        ),
        "records": records,
    }

    output_json_path.write_text(
        json.dumps(
            mapping_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("道路 ROI 批量映射完成")
    print(f"成功：{success_count}")
    print(f"失败：{failed_count}")
    print(f"Mask 目录：{output_mask_dir}")
    print(f"预览目录：{output_preview_dir}")
    print(f"映射记录：{output_json_path}")
    print("=" * 78)
    print(
        "下一步先检查 road_mask_previews 文件夹，"
        "确认每张绿色区域都准确覆盖道路。"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        cv2.destroyAllWindows()
        print(
            f"[失败] {type(exc).__name__}: {exc}"
        )
        sys.exit(1)
