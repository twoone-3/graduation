# -*- coding: utf-8 -*-
"""参数标定：扫描嵌入强度与系数对组合，观察不可见性与抗 JPEG 能力的折中。

用法：
    .venv\\Scripts\\python.exe scripts\\calibrate_strength.py
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
from dwtm.dft_watermark import DEFAULT_PAIRS  # noqa: E402
from dwtm.metrics import ber, psnr, ssim  # noqa: E402
from dwtm.pipeline import embed, extract, prepare_reference  # noqa: E402

PAIR_SETS = {
    "默认(1,2)(2,3)(1,3)": DEFAULT_PAIRS,
    "低频(1,1)(2,2)(1,3)": (((1, 1), (2, 2)), ((1, 3), (3, 1)), ((1, 2), (2, 1))),
    "偏低频(1,2)(2,3)(2,4)": (((1, 2), (2, 1)), ((2, 3), (3, 2)), ((2, 4), (4, 2))),
    "偏高(2,3)(3,4)(3,3)": (((2, 3), (3, 2)), ((3, 4), (4, 3)), ((2, 4), (4, 2))),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=str(ROOT / "assets/images/lena.jpg"))
    ap.add_argument("--bits", type=int, default=128)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    ref = prepare_reference(img)
    bits = payload.text_to_bits("DEMO 水印 2027", 16)[: args.bits]
    bits = np.resize(bits, args.bits)

    print(f"{'系数对':<24}{'强度':>6}{'PSNR':>8}{'SSIM':>8}{'Q90':>8}{'Q80':>8}{'Q70':>8}{'模糊':>8}")
    for pair_name, pairs in PAIR_SETS.items():
        for strength in (0.2, 0.35, 0.5, 0.7, 1.0, 1.4):
            wm, info = embed(img, bits, strength=strength, pairs=pairs)
            row = [psnr(ref, wm), ssim(ref, wm)]
            for q in (90, 80, 70):
                got = extract(attacks.jpeg(wm, q), bits.size, pairs=pairs).info_bits
                row.append(ber(bits, got))
            got = extract(attacks.gaussian_blur(wm, 3), bits.size, pairs=pairs).info_bits
            row.append(ber(bits, got))
            print(
                f"{pair_name:<24}{strength:>6.2f}{row[0]:>8.2f}{row[1]:>8.4f}"
                f"{row[2]:>8.3f}{row[3]:>8.3f}{row[4]:>8.3f}{row[5]:>8.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
