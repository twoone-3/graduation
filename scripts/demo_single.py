# -*- coding: utf-8 -*-
"""单图演示：嵌入 -> 质量评估 -> 攻击 -> 提取，并把中间结果存成图片。

用法（在项目根目录）：
    .venv\\Scripts\\python.exe scripts\\demo_single.py
    .venv\\Scripts\\python.exe scripts\\demo_single.py --image assets/images/baboon.jpg --text "hello"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from dwtm import attacks, payload  # noqa: E402
from dwtm.metrics import ber, nc, psnr, ssim  # noqa: E402
from dwtm.codes import make_codec  # noqa: E402
from dwtm.pipeline import capacity_bits, embed, extract, prepare_reference  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="DFT 水印单图演示")
    ap.add_argument("--image", default=str(ROOT / "assets/images/lena.jpg"))
    ap.add_argument("--text", default="DEMO 水印 2027")
    ap.add_argument("--bytes", type=int, default=16, help="文本载荷字节数（8 bit/字节）")
    ap.add_argument("--strength", type=float, default=1.2, help="嵌入强度")
    ap.add_argument("--codec", default="none", choices=["none", "polar"])
    ap.add_argument("--out", default=str(ROOT / "results/demo"))
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise SystemExit(f"读不到图片: {args.image}")

    bits = payload.text_to_bits(args.text, args.bytes)
    codec = make_codec(args.codec)

    print(f"图像      : {Path(args.image).name}  {img.shape}")
    print(f"载荷      : {bits.size} bit ({args.bytes} 字节)  文本: {args.text!r}")
    print(f"裸容量    : {capacity_bits(img.shape)} bit   编解码: {codec.name}")

    wm, info = embed(img, bits, codec=codec, strength=args.strength)
    ref = prepare_reference(img)

    print(f"实际嵌入  : {info.n_coded_bits} bit, 每比特重复 {info.repeat} 次, 用块 {info.used_blocks}/{info.n_blocks}")
    print(f"不可见性  : PSNR = {psnr(ref, wm):6.2f} dB   SSIM = {ssim(ref, wm):.4f}")

    res = extract(wm, bits.size, codec=codec)
    print(f"无攻击提取: BER = {ber(bits, res.info_bits):.4f}   NC = {nc(bits, res.info_bits):.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "original.png"), ref)
    cv2.imwrite(str(out_dir / "watermarked.png"), wm)

    diff = np.abs(wm.astype(np.int16) - ref.astype(np.int16)).astype(np.float64)
    amp = np.clip(diff * 10.0, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "difference_x10.png"), amp)

    cases = [
        ("原图", lambda im: im),
        ("JPEG Q=80", lambda im: attacks.jpeg(im, 80)),
        ("JPEG Q=60", lambda im: attacks.jpeg(im, 60)),
        ("高斯噪声 sigma=3", lambda im: attacks.gaussian_noise(im, 3.0)),
        ("高斯模糊 3x3", lambda im: attacks.gaussian_blur(im, 3)),
        ("亮度 +20", lambda im: attacks.brightness(im, 20)),
        ("裁剪 10%", lambda im: attacks.crop_fill(im, 0.10)),
    ]

    print("\n攻击后提取:")
    print(f"{'攻击':<18}{'PSNR':>8}{'BER':>9}{'NC':>8}   结果")
    for name, fn in cases:
        attacked = fn(wm)
        r = extract(attacked, bits.size, codec=codec)
        text = payload.bits_to_text(r.info_bits)
        print(
            f"{name:<18}{psnr(wm, attacked):8.2f}{ber(bits, r.info_bits):9.4f}"
            f"{nc(bits, r.info_bits):8.4f}   {text!r}"
        )
        safe = name.replace(" ", "_").replace("=", "")
        cv2.imwrite(str(out_dir / f"attacked_{safe}.png"), attacked)

    print(f"\n结果已保存到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
