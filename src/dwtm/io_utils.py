# -*- coding: utf-8 -*-
"""图像读写（兼容中文/非 ASCII 路径）。

OpenCV 的 imread/imwrite 在 Windows 上对非 ASCII 路径会静默失败，
这里统一走 imdecode/imencode + numpy 文件读写。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

__all__ = ["imread", "imwrite"]


def imread(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray | None:
    """读取图像，失败返回 None。"""
    p = Path(path)
    try:
        buf = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    img = cv2.imdecode(buf, flags)
    return img


def imwrite(path: str | Path, img: np.ndarray) -> bool:
    """写出图像，按扩展名选择编码格式。"""
    p = Path(path)
    suffix = p.suffix.lower() or ".png"
    if suffix in (".jpg", ".jpeg"):
        params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    elif suffix == ".bmp":
        params = []
    else:
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    ok, buf = cv2.imencode(suffix, img, params)
    if not ok:
        return False
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(p))
    except OSError:
        return False
    return True
