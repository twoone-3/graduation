# -*- coding: utf-8 -*-
"""纠错码接口层。

系统需要对比"无信道编码"与"Polar 编码"两种方案，因此把编码器抽象成统一接口：

    encode(信息比特) -> 编码比特
    decode(软信息)   -> 信息比特

接口按"软信息"（正值倾向于 1）设计，而不是硬判决比特，这样后续接入 Polar 码的
连续消除（SC）译码时可以原样使用 DFT 提取端输出的软值，不需要改上层流程。
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Codec(Protocol):
    """纠错码统一接口。"""

    name: str

    def n_coded(self, n_info: int) -> int:
        """给定信息比特数，返回编码后的比特数。"""

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """信息比特 (uint8, 0/1) -> 编码比特 (uint8, 0/1)。"""

    def decode(self, soft: np.ndarray) -> np.ndarray:
        """软信息 (float，正值判 1) -> 信息比特 (uint8, 0/1)。"""


class NoCodec:
    """无信道编码：直通，仅做硬判决。作为对比基线。"""

    name = "none"

    def n_coded(self, n_info: int) -> int:
        return n_info

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        return np.asarray(info_bits, dtype=np.uint8).ravel().copy()

    def decode(self, soft: np.ndarray) -> np.ndarray:
        return (np.asarray(soft, dtype=np.float64).ravel() > 0).astype(np.uint8)


class PolarCode:
    """固定码长 Polar 码（SC 译码）。

    TODO(下一阶段)：
      * 码长 N = 2^n，信息位 K 固定（如 (N, K) = (256, 128)，码率 1/2）；
      * 构造：Bhattacharyya 上界或高斯近似，离线算好冻结位集合后写成常量；
      * 编码：u -> x = u * F^{⊗n} * B（比特反序置换）；
      * 译码：SC，逐位计算 LLR，用提取端软值作为信道 LLR。
    """

    name = "polar"

    def __init__(self, n: int = 256, k: int = 128, design_snr_db: float = 2.0) -> None:
        if n & (n - 1):
            raise ValueError("码长 N 必须是 2 的幂")
        if not 0 < k <= n:
            raise ValueError("信息位 K 必须满足 0 < K <= N")
        self.n = n
        self.k = k
        self.design_snr_db = design_snr_db

    def n_coded(self, n_info: int) -> int:
        if n_info != self.k:
            raise ValueError(
                f"当前 Polar 码固定 K={self.k}，收到 {n_info} bit；请调整载荷长度或换一组参数"
            )
        return self.n

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Polar 编码将在下一阶段实现")

    def decode(self, soft: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Polar SC 译码将在下一阶段实现")


def make_codec(name: str = "none") -> Codec:
    """按名字创建编码器，便于命令行/界面切换方案。"""
    key = name.strip().lower()
    if key in ("none", "nocode", "raw"):
        return NoCodec()
    if key == "polar":
        return PolarCode()
    raise ValueError(f"未知的编码方案: {name}")
