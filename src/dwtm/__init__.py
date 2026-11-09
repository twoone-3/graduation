# -*- coding: utf-8 -*-
"""基于离散傅里叶变换的数字图像水印系统。

模块划分
--------
* ``dft_watermark`` —— 分块 DFT 中频系数对嵌入/提取（算法核心）
* ``pipeline``      —— 端到端流程：颜色空间、比特分散、纠错码接入
* ``codes``         —— 纠错码接口（无编码基线 / Polar 码）
* ``metrics``       —— PSNR、SSIM、BER、NC
* ``attacks``       —— JPEG、噪声、滤波、亮度、裁剪等攻击
* ``payload``       —— 短文本 / 二值图像 <-> 比特
"""

from . import attacks, metrics, payload
from .codes import Codec, NoCodec, PolarCode, make_codec
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
from .pipeline import (
    EmbedInfo,
    ExtractionResult,
    capacity_bits,
    embed,
    extract,
    prepare_reference,
    remove,
)

__all__ = [
    "attacks",
    "metrics",
    "payload",
    "Codec",
    "NoCodec",
    "PolarCode",
    "make_codec",
    "BLOCK_SIZE",
    "DEFAULT_PAIRS",
    "DEFAULT_STRENGTH",
    "DEFAULT_MAX_REPEAT",
    "DEFAULT_SEED",
    "count_blocks",
    "embed_block_bits",
    "extract_block_soft",
    "fit_to_blocks",
    "flatten_pair_differences",
    "gather_soft",
    "plan_layout",
    "EmbedInfo",
    "ExtractionResult",
    "capacity_bits",
    "embed",
    "extract",
    "prepare_reference",
    "remove",
]

__version__ = "0.1.0"
