# -*- coding: utf-8 -*-
"""端到端流程：彩色/灰度图像 <-> 亮度平面，比特分散，纠错码接入。

对外只暴露两个函数：

    embed(image, info_bits, codec=...)      -> (含水印图像, EmbedInfo)
    extract(image, n_info_bits, codec=...)  -> ExtractionResult

水印只嵌入亮度（Y）通道：人眼对亮度细节的容忍度与色度不同，嵌 Y 通道在同等
PSNR 下主观质量更好，也让灰度图与彩色图共用同一套流程。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .codes import Codec, NoCodec
from .dft_watermark import (
    BLOCK_SIZE,
    DEFAULT_MAX_REPEAT,
    DEFAULT_PAIRS,
    DEFAULT_SEED,
    DEFAULT_STRENGTH,
    count_blocks,
    embed_block_bits,
    extract_block_soft,
    fit_to_blocks,
    flatten_pair_differences,
    gather_soft,
    plan_layout,
)


@dataclass(frozen=True)
class EmbedInfo:
    shape: tuple[int, ...]
    n_blocks: int
    n_info_bits: int
    n_coded_bits: int
    repeat: int
    used_blocks: int
    strength: float
    codec: str


@dataclass(frozen=True)
class ExtractionResult:
    info_bits: np.ndarray
    soft: np.ndarray
    repeat: int
    n_blocks: int
    codec: str

    @property
    def confidence(self) -> float:
        """软信息平均绝对幅度，可作为提取可信度的粗略指标。"""
        return float(np.mean(np.abs(self.soft))) if self.soft.size else 0.0


# --------------------------------------------------------------------------- #
# 颜色空间搬运
# --------------------------------------------------------------------------- #
def _to_plane(img: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """取出待嵌入的亮度平面（float64, 0..255）以及用于回填的颜色载体。"""
    if img.ndim == 2:
        return img.astype(np.float64), None

    if img.shape[2] == 4:
        img = img[:, :, :3]

    bgr = img.astype(np.float32) / 255.0
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycc[:, :, 0].astype(np.float64) * 255.0, ycc


def _from_plane(plane: np.ndarray, carrier: np.ndarray | None, like: np.ndarray) -> np.ndarray:
    plane = np.clip(plane, 0.0, 255.0)

    if carrier is None:
        return np.rint(plane).clip(0, 255).astype(np.uint8)

    carrier = carrier.copy()
    carrier[:, :, 0] = (plane / 255.0).astype(np.float32)
    bgr = cv2.cvtColor(carrier, cv2.COLOR_YCrCb2BGR)
    out = np.rint(bgr.astype(np.float64) * 255.0).clip(0, 255).astype(np.uint8)

    if like.ndim == 3 and like.shape[2] == 4:
        out = np.dstack([out, like[:, :, 3]])
    return out


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def embed(
    image: np.ndarray,
    info_bits: np.ndarray,
    *,
    codec: Codec | None = None,
    strength: float = DEFAULT_STRENGTH,
    pairs=DEFAULT_PAIRS,
    block_size: int = BLOCK_SIZE,
    scale_mode: str = "global",
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    seed: int = DEFAULT_SEED,
) -> tuple[np.ndarray, EmbedInfo]:
    """把信息比特嵌入图像。

    返回的图像尺寸会被边缘复制填充到 block_size 的整数倍（最多补 7 像素），
    这样嵌入端与提取端看到的分块网格完全一致。
    """
    codec = codec or NoCodec()
    info_bits = np.asarray(info_bits, dtype=np.uint8).ravel()
    coded = np.asarray(codec.encode(info_bits), dtype=np.uint8).ravel()

    img = fit_to_blocks(np.asarray(image), block_size)
    plane, carrier = _to_plane(img)

    n_blocks = count_blocks(plane.shape, block_size)
    positions, repeat = plan_layout(
        coded.size, n_blocks, max_repeat=max_repeat, seed=seed
    )

    block_bits = np.zeros(n_blocks, dtype=np.uint8)
    block_bits[positions] = coded[np.arange(positions.size) % coded.size]

    strength_map = np.zeros(n_blocks, dtype=np.float64)
    strength_map[positions] = strength

    wm_plane = embed_block_bits(
        plane,
        block_bits,
        strength=strength_map,
        pairs=pairs,
        block_size=block_size,
        scale_mode=scale_mode,
    )
    out = _from_plane(wm_plane, carrier, img)

    info = EmbedInfo(
        shape=out.shape,
        n_blocks=n_blocks,
        n_info_bits=int(info_bits.size),
        n_coded_bits=int(coded.size),
        repeat=int(repeat),
        used_blocks=int(positions.size),
        strength=float(np.mean(strength)),
        codec=codec.name,
    )
    return out, info


def extract(
    image: np.ndarray,
    n_info_bits: int,
    *,
    codec: Codec | None = None,
    pairs=DEFAULT_PAIRS,
    block_size: int = BLOCK_SIZE,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    seed: int = DEFAULT_SEED,
) -> ExtractionResult:
    """从图像中提取信息比特（软信息经重复合并后送入译码器）。"""
    codec = codec or NoCodec()
    n_coded = codec.n_coded(n_info_bits)

    img = fit_to_blocks(np.asarray(image), block_size)
    plane, _ = _to_plane(img)

    n_blocks = count_blocks(plane.shape, block_size)
    positions, repeat = plan_layout(n_coded, n_blocks, max_repeat=max_repeat, seed=seed)

    soft_blocks = extract_block_soft(plane, pairs=pairs, block_size=block_size)
    soft = gather_soft(soft_blocks[positions], n_coded, repeat)
    info_bits = np.asarray(codec.decode(soft), dtype=np.uint8)

    return ExtractionResult(
        info_bits=info_bits,
        soft=soft,
        repeat=int(repeat),
        n_blocks=n_blocks,
        codec=codec.name,
    )


def remove(
    image: np.ndarray,
    n_info_bits: int,
    *,
    codec: Codec | None = None,
    pairs=DEFAULT_PAIRS,
    block_size: int = BLOCK_SIZE,
    max_repeat: int | None = DEFAULT_MAX_REPEAT,
    seed: int = DEFAULT_SEED,
) -> np.ndarray:
    """去水印：把嵌入时用过的块里的系数对幅值拉平，抹掉相对幅度承载的比特。

    这是"已知密钥 + 盲目抑制"的去除方式，不需要原图：只需要知道载荷长度、
    重复次数和选块种子（与提取端相同）。去除后再次提取，BER 应接近 0.5（等于瞎猜），
    同时图像回到嵌入前的能量分布，PSNR 相对原图通常优于含水印图。
    """
    codec = codec or NoCodec()
    n_coded = codec.n_coded(n_info_bits)

    img = fit_to_blocks(np.asarray(image), block_size)
    plane, carrier = _to_plane(img)

    n_blocks = count_blocks(plane.shape, block_size)
    positions, _ = plan_layout(n_coded, n_blocks, max_repeat=max_repeat, seed=seed)

    clean_plane = flatten_pair_differences(
        plane, positions, pairs=pairs, block_size=block_size
    )
    return _from_plane(clean_plane, carrier, img)


def capacity_bits(image_shape: tuple[int, ...], block_size: int = BLOCK_SIZE) -> int:
    """图像在"每块 1 bit"下的原始容量（不含纠错码开销）。"""
    h, w = image_shape[:2]
    h, w = h + (-h) % block_size, w + (-w) % block_size
    return count_blocks((h, w), block_size)


def prepare_reference(image: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """返回与 embed 输出同尺寸的参考图（仅做边缘填充），用于 PSNR/SSIM 比较。"""
    return fit_to_blocks(np.asarray(image), block_size)
