# -*- coding: utf-8 -*-
"""分块 DFT 域数字图像水印（中频系数对相对幅度调制）。

算法要点
--------
1. 8x8 非重叠分块：与 JPEG 的变换块网格对齐，局部篡改/压缩的影响被限制在少数块内。
2. 每块在若干"中频系数对"上嵌入 1 bit：以 |C(u1,v1)| - |C(u2,v2)| 的符号表示比特，
   幅度差由该块的局部能量乘以嵌入强度决定，因此对整体亮度/增益变化不敏感。
3. 只改幅度、保留相位，并同步修改共轭位置 C(-u,-v)，保证 IDFT 结果仍是实图像。
4. 比特按"块序号 k --> 比特序号 k % n_bits"交织分散到全图，使得局部裁剪只损伤
   每个比特的一部分副本，为后续的多数表决与 Polar 纠错码留出修复空间。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

BLOCK_SIZE = 8

#: 中频系数对。取值偏中低频，兼顾 JPEG 存活率与不可见性。
DEFAULT_PAIRS: tuple[tuple[tuple[int, int], tuple[int, int]], ...] = (
    ((1, 2), (2, 1)),
    ((2, 3), (3, 2)),
    ((1, 3), (3, 1)),
)

#: 默认嵌入强度（幅度差 = strength * 全图平均幅值），由 grid_search.py 标定：
#: strength=1.2、max_repeat=8 时，5 幅测试图 PSNR 均值 44.0 dB、SSIM 0.995，
#: JPEG Q60 下误码率约 7.8%，为后续 Polar 纠错留出发挥空间。
DEFAULT_STRENGTH = 1.2

#: 默认重复次数：每个比特被写入 8 个随机块。None 表示用满全部块。
DEFAULT_MAX_REPEAT: int | None = 8

#: 默认选块种子（相当于水印密钥的一部分，提取端必须一致）。
DEFAULT_SEED = 2027

_EPS = 1e-9


def validate_pairs(
    pairs: Sequence[tuple[tuple[int, int], tuple[int, int]]],
    block_size: int = BLOCK_SIZE,
) -> None:
    """检查系数对是否合法：不重复、不落在 DC 与二维 Nyquist 上。"""
    seen: set[tuple[int, int]] = set()
    for p1, p2 in pairs:
        for pos in (p1, p2):
            u, v = pos
            if not (0 <= u < block_size and 0 <= v < block_size):
                raise ValueError(f"系数位置越界: {pos}")
            mirrors = {(u, v), ((block_size - u) % block_size, (block_size - v) % block_size)}
            if (u, v) == (0, 0) or (u, v) == (block_size // 2, block_size // 2):
                raise ValueError(f"不能使用 DC 或二维 Nyquist 位置: {pos}")
            for m in mirrors:
                if m in seen:
                    raise ValueError(f"系数位置冲突: {m}")
                seen.add(m)


validate_pairs(DEFAULT_PAIRS)


# --------------------------------------------------------------------------- #
# 分块工具
# --------------------------------------------------------------------------- #
def fit_to_blocks(plane: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """边缘复制填充，使高、宽都是 block_size 的整数倍。"""
    h, w = plane.shape[:2]
    ph, pw = (-h) % block_size, (-w) % block_size
    if ph == 0 and pw == 0:
        return plane
    pad = ((0, ph), (0, pw)) + (((0, 0),) * (plane.ndim - 2))
    return np.pad(plane, pad, mode="edge")


def to_blocks(plane: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """(H, W) -> (H/bs, W/bs, bs, bs)，块按行优先排列。"""
    h, w = plane.shape
    if h % block_size or w % block_size:
        raise ValueError("图像尺寸必须是 block_size 的整数倍")
    return plane.reshape(h // block_size, block_size, w // block_size, block_size).transpose(0, 2, 1, 3)


def from_blocks(blocks: np.ndarray, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """(H/bs, W/bs, bs, bs) -> (H, W)，与 to_blocks 互逆。"""
    bh, bw = blocks.shape[:2]
    return blocks.transpose(0, 2, 1, 3).reshape(bh * block_size, bw * block_size)


def count_blocks(shape: Iterable[int], block_size: int = BLOCK_SIZE) -> int:
    """给定 (H, W) 能容纳的水印块数（即每块 1 bit 时的容量）。"""
    h, w = list(shape)[:2]
    return (h // block_size) * (w // block_size)


# --------------------------------------------------------------------------- #
# 嵌入 / 提取
# --------------------------------------------------------------------------- #
def _pair_coords(pairs, block_size=BLOCK_SIZE) -> list[tuple[int, int]]:
    coords = []
    for p1, p2 in pairs:
        coords.extend([p1, p2])
    return coords


def embed_block_bits(
    plane: np.ndarray,
    block_bits: np.ndarray,
    *,
    strength: float = DEFAULT_STRENGTH,
    pairs=DEFAULT_PAIRS,
    block_size: int = BLOCK_SIZE,
    scale_mode: str = "local",
) -> np.ndarray:
    """逐块嵌入 1 bit，返回与输入同尺寸（已填充）的实值图像。

    block_bits 的长度必须等于块数，顺序与 to_blocks 的行优先顺序一致。

    scale_mode 决定幅度差的尺度基准：
      * ``local``     —— 块内所选系数的平均幅值，纹理区嵌得更"重"，主观掩蔽好但 PSNR 偏低；
      * ``global``    —— 全图平均幅值，各块强度一致，PSNR 稳定但在平滑区偏"重"；
      * ``geometric`` —— 两者的几何平均，兼顾。
    """
    plane = fit_to_blocks(np.asarray(plane, dtype=np.float64), block_size)
    blocks = to_blocks(plane, block_size)
    nb = blocks.shape[0] * blocks.shape[1]

    validate_pairs(pairs, block_size)

    bits = np.asarray(block_bits, dtype=np.uint8).ravel()
    if bits.size != nb:
        raise ValueError(f"block_bits 长度 {bits.size} 与块数 {nb} 不一致")
    bit_map = bits.reshape(blocks.shape[0], blocks.shape[1])

    spec = np.fft.fft2(blocks, axes=(2, 3))

    # 局部能量：所选系数幅值的平均值，用作幅度差的尺度基准
    coords = _pair_coords(pairs, block_size)
    local = np.mean([np.abs(spec[..., u, v]) for u, v in coords], axis=0)
    if scale_mode == "local":
        base = local
    elif scale_mode == "global":
        base = np.full_like(local, float(local.mean()))
    elif scale_mode == "geometric":
        base = np.sqrt(local * float(local.mean()))
    else:
        raise ValueError(f"未知的 scale_mode: {scale_mode}")

    target = np.asarray(strength, dtype=np.float64)
    if target.ndim == 1:
        target = target.reshape(bit_map.shape)
    target = target * (base + _EPS)
    strength_arr = target

    sign = np.where(bit_map > 0, 1.0, -1.0)

    for p1, p2 in pairs:
        u1, v1 = p1
        u2, v2 = p2
        c1 = spec[..., u1, v1]
        c2 = spec[..., u2, v2]
        m1, m2 = np.abs(c1), np.abs(c2)
        m = 0.5 * (m1 + m2)
        # 目标幅度差：2*delta，符号由比特决定；上限避免负幅度或过强失真
        delta = np.minimum(0.5 * strength_arr, 0.9 * (m + _EPS))
        want = sign * 2.0 * delta
        shift = 0.5 * (want - (m1 - m2))
        active = delta > 0.0

        n1 = np.where(active, np.maximum(m1 + shift, 0.0), m1)
        n2 = np.where(active, np.maximum(m2 - shift, 0.0), m2)
        # 若一侧被截断在 0，重新拉开以维持幅度差
        deficit = np.where(active, want - (n1 - n2), 0.0)
        n1 = np.maximum(n1 + np.maximum(deficit, 0.0), 0.0)
        n2 = np.maximum(n2 + np.maximum(-deficit, 0.0), 0.0)

        new1 = n1 * np.exp(1j * np.angle(c1))
        new2 = n2 * np.exp(1j * np.angle(c2))
        spec[..., u1, v1] = new1
        spec[..., u2, v2] = new2

        # 共轭对称位置，保证 ifft2 为实信号
        cu1, cv1 = (block_size - u1) % block_size, (block_size - v1) % block_size
        cu2, cv2 = (block_size - u2) % block_size, (block_size - v2) % block_size
        spec[..., cu1, cv1] = np.conj(new1)
        spec[..., cu2, cv2] = np.conj(new2)

    out = np.real(np.fft.ifft2(spec, axes=(2, 3)))
    return from_blocks(out, block_size)


def extract_block_soft(
    plane: np.ndarray,
    *,
    pairs=DEFAULT_PAIRS,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """逐块输出软判决值（正数判 1，负数判 0），形状 (块数,)。

    软值定义为各系数对幅值差之均值，再除以局部能量做归一化，
    便于在不同亮度/强度的图像之间比较置信度。
    """
    plane = fit_to_blocks(np.asarray(plane, dtype=np.float64), block_size)
    blocks = to_blocks(plane, block_size)
    validate_pairs(pairs, block_size)
    spec = np.fft.fft2(blocks, axes=(2, 3))

    coords = _pair_coords(pairs, block_size)
    local = np.mean([np.abs(spec[..., u, v]) for u, v in coords], axis=0) + _EPS

    diffs = []
    for p1, p2 in pairs:
        u1, v1 = p1
        u2, v2 = p2
        diffs.append(np.abs(spec[..., u1, v1]) - np.abs(spec[..., u2, v2]))

    soft = np.mean(diffs, axis=0) / local
    return soft.reshape(-1)


def flatten_pair_differences(
    plane: np.ndarray,
    positions: np.ndarray | None = None,
    *,
    pairs=DEFAULT_PAIRS,
    block_size: int = BLOCK_SIZE,
) -> np.ndarray:
    """把指定块内各系数对的幅值拉平到"两者均值"，用于去除相对幅度水印。

    嵌入时的信息载体是 |C_a| - |C_b| 的符号；把这一差值压成 0，符号信息随之消失，
    而两个系数回到原本的能量水平。相位保持不变，共轭位置同步更新，保证 IDFT 为实。

    positions 为要处理的块序号（一般取自 plan_layout，即嵌入时用过的块）；
    传 None 表示处理全部块。
    """
    plane = fit_to_blocks(np.asarray(plane, dtype=np.float64), block_size)
    blocks = to_blocks(plane, block_size)
    validate_pairs(pairs, block_size)
    spec = np.fft.fft2(blocks, axes=(2, 3))

    bh, bw = blocks.shape[:2]
    if positions is None:
        mask = np.ones((bh, bw), dtype=bool)
    else:
        mask = np.zeros(bh * bw, dtype=bool)
        mask[np.asarray(positions, dtype=np.int64)] = True
        mask = mask.reshape(bh, bw)

    for p1, p2 in pairs:
        u1, v1 = p1
        u2, v2 = p2
        c1 = spec[..., u1, v1]
        c2 = spec[..., u2, v2]
        m1, m2 = np.abs(c1), np.abs(c2)
        mean = 0.5 * (m1 + m2)
        n1 = np.where(mask, mean, m1)
        n2 = np.where(mask, mean, m2)

        new1 = n1 * np.exp(1j * np.angle(c1))
        new2 = n2 * np.exp(1j * np.angle(c2))
        spec[..., u1, v1] = new1
        spec[..., u2, v2] = new2

        cu1, cv1 = (block_size - u1) % block_size, (block_size - v1) % block_size
        cu2, cv2 = (block_size - u2) % block_size, (block_size - v2) % block_size
        spec[..., cu1, cv1] = np.conj(new1)
        spec[..., cu2, cv2] = np.conj(new2)

    out = np.real(np.fft.ifft2(spec, axes=(2, 3)))
    return from_blocks(out, block_size)


def plan_layout(
    n_bits: int,
    n_blocks: int,
    *,
    max_repeat: int | None = None,
    seed: int = 2027,
) -> tuple[np.ndarray, int]:
    """规划"比特 -> 块"的映射。

    返回 (positions, repeat)：positions 是承载水印的块序号（升序），块 positions[k]
    承载第 k % n_bits 个比特。

    * ``max_repeat=None`` —— 用满全部块（repeat = n_blocks // n_bits），重复次数最大；
    * 指定 max_repeat 时 —— 只随机挑选 repeat * n_bits 个块，其余块保持原样。
      这种"稀疏嵌入"按种子选块，既降低了失真，也让未授权者难以定位水印，
      代价是每个比特的副本数变少。
    """
    if n_bits <= 0:
        raise ValueError("比特数必须为正")
    full_repeat = n_blocks // n_bits
    if full_repeat < 1:
        raise ValueError(
            f"容量不足：{n_blocks} 个块无法承载 {n_bits} bit（建议减少载荷或增大图像）"
        )
    repeat = full_repeat if max_repeat is None else min(full_repeat, int(max_repeat))
    if repeat < 1:
        raise ValueError("max_repeat 至少为 1")

    used = repeat * n_bits
    if max_repeat is None:
        positions = np.arange(used, dtype=np.int64)
    else:
        rng = np.random.default_rng(seed)
        positions = np.sort(rng.permutation(n_blocks)[:used])
    return positions, repeat


def gather_soft(soft_at_positions: np.ndarray, n_bits: int, repeat: int) -> np.ndarray:
    """把块级软值按比特做均值合并（软信息的重复合并）。

    与 plan_layout 的"第 k 个块承载第 k % n_bits 个比特"对应：按 (repeat, n_bits)
    重排后第 j 列恰好是第 j 个比特的全部副本。
    """
    used = repeat * n_bits
    values = np.asarray(soft_at_positions, dtype=np.float64)[:used].reshape(repeat, n_bits)
    return values.mean(axis=0)
