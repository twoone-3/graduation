# -*- coding: utf-8 -*-
"""核心功能自检。运行：

    .venv\\Scripts\\python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dwtm import attacks, payload  # noqa: E402
from dwtm.dft_watermark import (  # noqa: E402
    count_blocks,
    embed_block_bits,
    extract_block_soft,
    fit_to_blocks,
    from_blocks,
    plan_layout,
    to_blocks,
    validate_pairs,
)
from dwtm.metrics import ber, nc, psnr, ssim  # noqa: E402
from dwtm.pipeline import capacity_bits, embed, extract, prepare_reference, remove  # noqa: E402


def sample_image(h: int = 256, w: int = 256, seed: int = 0) -> np.ndarray:
    """构造一幅平滑的合成图（渐变 + 低频纹理），近似自然图像的频谱分布。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, 3)
    gray = (
        128.0
        + 40.0 * np.sin(2 * np.pi * xx / 61.0 + phase[0])
        + 30.0 * np.cos(2 * np.pi * yy / 47.0 + phase[1])
        + 20.0 * np.sin(2 * np.pi * (xx + yy) / 113.0 + phase[2])
        + rng.normal(0.0, 8.0, size=(h, w))  # 少量纹理噪声，贴近自然图像的宽带频谱
    )
    gray = np.clip(gray, 0, 255)
    return np.dstack([gray] * 3).astype(np.uint8)


class TestBlocks(unittest.TestCase):
    def test_roundtrip(self):
        img = np.arange(64, dtype=np.float64).reshape(8, 8)
        self.assertTrue(np.allclose(from_blocks(to_blocks(img)), img))

    def test_fit_to_blocks(self):
        img = np.zeros((13, 21), dtype=np.uint8)
        out = fit_to_blocks(img, 8)
        self.assertEqual(out.shape, (16, 24))
        self.assertTrue(np.array_equal(out[:13, :21], img))

    def test_count_blocks(self):
        self.assertEqual(count_blocks((256, 512)), 32 * 64)

    def test_validate_pairs_rejects_duplicate(self):
        with self.assertRaises(ValueError):
            validate_pairs((((1, 2), (2, 1)), ((2, 1), (3, 3))))

    def test_validate_pairs_rejects_dc(self):
        with self.assertRaises(ValueError):
            validate_pairs((((0, 0), (1, 2)),))


class TestSingleBlock(unittest.TestCase):
    def test_bit_is_recoverable_before_rounding(self):
        plane = np.random.default_rng(1).uniform(0, 255, size=(32, 32))
        for bit in (0, 1):
            bits = np.full(16, bit, dtype=np.uint8)
            wm = embed_block_bits(plane, bits, strength=1.0)
            soft = extract_block_soft(wm)
            signs = soft > 0
            self.assertTrue(np.all(signs == bool(bit)), msg=f"bit={bit} 未正确嵌入")

    def test_idempotent_when_strength_zero(self):
        plane = np.random.default_rng(2).uniform(0, 255, size=(32, 32))
        bits = np.zeros(16, dtype=np.uint8)
        wm = embed_block_bits(plane, bits, strength=0.0)
        self.assertTrue(np.allclose(wm, plane, atol=1e-9))


class TestLayout(unittest.TestCase):
    def test_plan_layout_full(self):
        pos, repeat = plan_layout(128, 4096, max_repeat=None)
        self.assertEqual(repeat, 32)
        self.assertEqual(pos.size, 4096)
        self.assertTrue(np.array_equal(pos, np.arange(4096)))

    def test_plan_layout_sparse(self):
        pos, repeat = plan_layout(128, 4096, max_repeat=8, seed=1)
        self.assertEqual(repeat, 8)
        self.assertEqual(pos.size, 1024)
        self.assertEqual(len(set(pos.tolist())), 1024)
        self.assertTrue(np.all(np.diff(pos) > 0))

    def test_capacity_error(self):
        with self.assertRaises(ValueError):
            plan_layout(100, 10)


class TestPayload(unittest.TestCase):
    def test_text_roundtrip(self):
        bits = payload.text_to_bits("水印 test", 16)
        self.assertEqual(bits.size, 128)
        self.assertEqual(payload.bits_to_text(bits), "水印 test")

    def test_text_too_long(self):
        with self.assertRaises(ValueError):
            payload.text_to_bits("x" * 100, 16)

    def test_image_roundtrip(self):
        img = (np.arange(32 * 32).reshape(32, 32) % 2 * 255).astype(np.uint8)
        bits = payload.image_to_bits(img, (32, 32))
        back = payload.bits_to_image(bits, (32, 32))
        self.assertTrue(np.array_equal(back, img))


class TestMetrics(unittest.TestCase):
    def test_psnr_identical(self):
        a = np.zeros((8, 8), dtype=np.uint8)
        self.assertEqual(psnr(a, a), float("inf"))

    def test_psnr_known(self):
        a = np.zeros((8, 8), dtype=np.uint8)
        b = np.full((8, 8), 1, dtype=np.uint8)
        self.assertAlmostEqual(psnr(a, b), 48.13, places=2)

    def test_ssim_identical(self):
        a = np.random.default_rng(0).integers(0, 256, (64, 64)).astype(np.uint8)
        self.assertAlmostEqual(ssim(a, a), 1.0, places=6)

    def test_ber_nc(self):
        a = np.array([0, 1, 1, 0], dtype=np.uint8)
        b = np.array([0, 1, 0, 0], dtype=np.uint8)
        self.assertAlmostEqual(ber(a, b), 0.25)
        self.assertAlmostEqual(nc(a, b), 0.5, places=6)


class TestPipeline(unittest.TestCase):
    def test_embed_extract_clean(self):
        img = sample_image()
        bits = payload.text_to_bits("hello 水印", 12)
        wm, info = embed(img, bits)
        self.assertEqual(info.n_info_bits, bits.size)
        self.assertGreater(psnr(prepare_reference(img), wm), 35.0)
        res = extract(wm, bits.size)
        self.assertEqual(ber(bits, res.info_bits), 0.0)

    def test_survives_brightness_and_crop(self):
        img = sample_image()
        bits = payload.text_to_bits("robust", 8)
        wm, _ = embed(img, bits)
        for fn in (lambda im: attacks.brightness(im, 20), lambda im: attacks.crop_fill(im, 0.05)):
            res = extract(fn(wm), bits.size)
            self.assertEqual(ber(bits, res.info_bits), 0.0)

    def test_noise_increases_ber(self):
        img = sample_image(seed=3)
        bits = payload.text_to_bits("noise test", 16)
        wm, _ = embed(img, bits, strength=0.2, max_repeat=2)
        clean = ber(bits, extract(wm, bits.size, max_repeat=2).info_bits)
        noisy = ber(bits, extract(attacks.jpeg(wm, 20), bits.size, max_repeat=2).info_bits)
        self.assertGreaterEqual(noisy, clean)

    def test_capacity(self):
        self.assertEqual(capacity_bits((512, 512)), 4096)

    def test_remove_destroys_watermark(self):
        img = sample_image(seed=5)
        bits = payload.text_to_bits("remove me", 12)
        wm, _ = embed(img, bits)
        ref = prepare_reference(img)
        clean = remove(wm, bits.size)

        # 去除后仍接近原图，且比含水印图更接近
        self.assertEqual(clean.shape, wm.shape)
        self.assertGreater(psnr(ref, clean), 30.0)
        self.assertGreater(psnr(ref, clean), psnr(ref, wm))

        # 水印已不可读：误码率显著上升
        self.assertGreater(ber(bits, extract(clean, bits.size).info_bits), 0.15)

    def test_wrong_seed_breaks_extraction(self):
        img = sample_image()
        bits = payload.text_to_bits("key", 4)
        wm, _ = embed(img, bits, seed=1)
        wrong = extract(wm, bits.size, seed=999).info_bits
        self.assertAlmostEqual(ber(bits, wrong), 0.5, delta=0.2)

    def test_grayscale_supported(self):
        img = sample_image()[:, :, 0]
        bits = payload.text_to_bits("gray", 4)
        wm, _ = embed(img, bits)
        self.assertEqual(wm.ndim, 2)
        self.assertEqual(ber(bits, extract(wm, bits.size).info_bits), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
