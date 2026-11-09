# -*- coding: utf-8 -*-
"""水印载荷编解码：短文本 / 二值图像 <-> 比特序列。

文本按 UTF-8 编码后定长补零，二值图像按行优先展开；两者都以 uint8 的 0/1 数组表示，
方便直接送入纠错码与嵌入流程。
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "bytes_to_bits",
    "bits_to_bytes",
    "text_to_bits",
    "bits_to_text",
    "image_to_bits",
    "bits_to_image",
    "text_capacity",
]


def bytes_to_bits(buf: bytes) -> np.ndarray:
    """字节 -> 比特（高位在前）。"""
    return np.unpackbits(np.frombuffer(buf, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """比特 -> 字节（高位在前），长度必须是 8 的倍数。"""
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if arr.size % 8:
        raise ValueError("比特数必须是 8 的倍数")
    return np.packbits(arr).tobytes()


def text_to_bits(text: str, n_bytes: int) -> np.ndarray:
    """定长文本载荷：UTF-8 编码 + 补 0x00 至 n_bytes。"""
    if n_bytes <= 0:
        raise ValueError("n_bytes 必须为正")
    raw = text.encode("utf-8")
    if len(raw) > n_bytes:
        raise ValueError(f"文本过长：{len(raw)} 字节，上限 {n_bytes} 字节")
    return bytes_to_bits(raw + b"\x00" * (n_bytes - len(raw)))


def bits_to_text(bits: np.ndarray) -> str:
    """还原文本，去掉补位产生的空字符。"""
    raw = bits_to_bytes(bits)
    cut = raw.find(b"\x00")
    if cut >= 0:
        raw = raw[:cut]
    return raw.decode("utf-8", errors="replace")


def text_capacity(n_bits: int) -> int:
    """给定可用比特数，能装多少个字节的文本（向下取整）。"""
    return int(n_bits) // 8


def image_to_bits(img: np.ndarray, size: tuple[int, int] = (32, 32), threshold: int = 128) -> np.ndarray:
    """二值水印图像 -> 比特。图像会被缩放到 size 再二值化。"""
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray.astype(np.uint8), (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return (resized > threshold).astype(np.uint8).ravel()


def bits_to_image(bits: np.ndarray, size: tuple[int, int] = (32, 32)) -> np.ndarray:
    """比特 -> 二值水印图像（0/255）。"""
    h, w = size
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if arr.size != h * w:
        raise ValueError(f"比特数 {arr.size} 与 size={size} 不匹配")
    return (arr.reshape(h, w) * 255).astype(np.uint8)
