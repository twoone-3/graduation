# -*- coding: utf-8 -*-
"""评价指标：PSNR、SSIM（不可见性），BER、NC（水印恢复效果）。"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["psnr", "ssim", "ber", "nc", "to_gray"]


def to_gray(img: np.ndarray) -> np.ndarray:
    """转灰度 float64。彩色图按 OpenCV 的 BGR 顺序处理。"""
    arr = np.asarray(img)
    if arr.ndim == 2:
        return arr.astype(np.float64)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)


def psnr(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """峰值信噪比（dB）。完全相同返回 inf。"""
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"尺寸不一致: {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def _gaussian(size: int = 11, sigma: float = 1.5) -> np.ndarray:
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(x**2) / (2.0 * sigma**2))
    g /= g.sum()
    return g


def ssim(reference: np.ndarray, test: np.ndarray, data_range: float = 255.0) -> float:
    """结构相似度（11x11 高斯窗，sigma=1.5），彩色图取灰度后再算。"""
    x = to_gray(reference)
    y = to_gray(test)
    if x.shape != y.shape:
        raise ValueError(f"尺寸不一致: {x.shape} vs {y.shape}")

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    kernel = _gaussian(11, 1.5)

    def blur(v: np.ndarray) -> np.ndarray:
        return cv2.sepFilter2D(v, cv2.CV_64F, kernel, kernel)

    mu_x, mu_y = blur(x), blur(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y

    sigma_xx = blur(x * x) - mu_xx
    sigma_yy = blur(y * y) - mu_yy
    sigma_xy = blur(x * y) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_xx + mu_yy + c1) * (sigma_xx + sigma_yy + c2)
    return float(np.mean(numerator / denominator))


def ber(reference_bits: np.ndarray, test_bits: np.ndarray) -> float:
    """误码率：错误比特数 / 总比特数。"""
    a = np.asarray(reference_bits, dtype=np.uint8).ravel()
    b = np.asarray(test_bits, dtype=np.uint8).ravel()
    if a.size != b.size:
        raise ValueError(f"比特数不一致: {a.size} vs {b.size}")
    if a.size == 0:
        return 0.0
    return float(np.mean(a != b))


def nc(reference_bits: np.ndarray, test_bits: np.ndarray) -> float:
    """归一化相关系数：把 0/1 映射到 -1/+1 后求相关系数，1 表示完全一致。"""
    a = np.asarray(reference_bits, dtype=np.float64).ravel() * 2.0 - 1.0
    b = np.asarray(test_bits, dtype=np.float64).ravel() * 2.0 - 1.0
    if a.size != b.size:
        raise ValueError(f"比特数不一致: {a.size} vs {b.size}")
    denom = float(np.sqrt(np.sum(a**2) * np.sum(b**2)))
    if denom <= 0:
        return 0.0
    return float(np.sum(a * b) / denom)
