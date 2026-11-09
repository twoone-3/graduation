# -*- coding: utf-8 -*-
"""常见图像处理攻击，用于鲁棒性测试。

对应任务书要求的五类处理：JPEG 压缩、加性噪声、轻度滤波、亮度调整、小范围裁剪。
所有函数都保持图像尺寸不变（裁剪类攻击是"挖掉一块再填色"），便于与水印图像逐像素比较。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "jpeg",
    "gaussian_noise",
    "gaussian_blur",
    "median_blur",
    "brightness",
    "contrast",
    "crop_fill",
    "crop_resize",
    "rescale",
]


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(img), 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """JPEG 有损压缩/解压。"""
    if not 1 <= quality <= 100:
        raise ValueError("quality 应在 1..100")
    data = _ensure_uint8(img)
    flag = cv2.IMWRITE_JPEG_QUALITY
    ok, buf = cv2.imencode(".jpg", data, [flag, int(quality)])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return out


def gaussian_noise(img: np.ndarray, sigma: float = 3.0, seed: int = 0) -> np.ndarray:
    """零均值高斯加性噪声（标准差 sigma，灰度级）。"""
    rng = np.random.default_rng(seed)
    data = np.asarray(img, dtype=np.float64)
    noisy = data + rng.normal(0.0, sigma, size=data.shape)
    return np.rint(noisy).clip(0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, ksize: int = 3, sigma: float = 0.0) -> np.ndarray:
    """高斯低通滤波（轻度模糊）。"""
    out = cv2.GaussianBlur(_ensure_uint8(img), (ksize, ksize), sigma)
    return out


def median_blur(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    """中值滤波（去噪类处理）。"""
    return cv2.medianBlur(_ensure_uint8(img), ksize)


def brightness(img: np.ndarray, delta: float = 10.0) -> np.ndarray:
    """亮度偏移：所有像素加常数（可为负）。"""
    out = np.asarray(img, dtype=np.float64) + float(delta)
    return np.rint(out).clip(0, 255).astype(np.uint8)


def contrast(img: np.ndarray, factor: float = 1.05) -> np.ndarray:
    """对比度缩放：围绕 128 灰度级线性放大。"""
    data = np.asarray(img, dtype=np.float64)
    out = (data - 128.0) * float(factor) + 128.0
    return np.rint(out).clip(0, 255).astype(np.uint8)


def crop_fill(img: np.ndarray, ratio: float = 0.1, fill: int = 0, where: str = "corner") -> np.ndarray:
    """小范围裁剪：挖掉一块并以常数填充，尺寸保持不变。

    ratio 为被裁剪区域的边长占短边的比例；where 可选 corner / center。
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio 应在 [0, 1)")
    out = _ensure_uint8(img).copy()
    h, w = out.shape[:2]
    side = max(1, int(round(ratio * min(h, w))))

    if where == "corner":
        y0, x0 = 0, 0
    elif where == "center":
        y0, x0 = (h - side) // 2, (w - side) // 2
    else:
        raise ValueError("where 只支持 corner / center")

    out[y0 : y0 + side, x0 : x0 + side] = fill
    return out


def crop_resize(img: np.ndarray, ratio: float = 0.1) -> np.ndarray:
    """裁剪后再缩放回原尺寸（模拟裁掉边缘又拉伸的情况）。"""
    if not 0.0 <= ratio < 1.0:
        raise ValueError("ratio 应在 [0, 1)")
    data = _ensure_uint8(img)
    h, w = data.shape[:2]
    dy = int(round(h * ratio / 2))
    dx = int(round(w * ratio / 2))
    cropped = data[dy : h - dy, dx : w - dx]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


def rescale(img: np.ndarray, scale: float = 0.75) -> np.ndarray:
    """先缩小再放大回原尺寸（有损重采样）。"""
    if scale <= 0:
        raise ValueError("scale 必须为正")
    data = _ensure_uint8(img)
    h, w = data.shape[:2]
    small = cv2.resize(data, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
