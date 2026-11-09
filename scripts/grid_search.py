# -*- coding: utf-8 -*-
"""参数网格搜索：嵌入强度 x 重复次数，输出现象表，用于确定默认参数。

用法：
    .venv\\Scripts\\python.exe scripts\\grid_search.py
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
from dwtm.metrics import ber, psnr  # noqa: E402
from dwtm.pipeline import embed, extract, prepare_reference  # noqa: E402

IMAGES = ["lena", "baboon", "fruits", "board", "butterfly"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.8, 1.2, 1.6, 2.0])
    ap.add_argument("--repeats", nargs="+", type=int, default=[4, 8, 16])
    args = ap.parse_args()

    bits = payload.text_to_bits("DEMO 水印 2027", 16)
    cases = [
        ("Q70", lambda im: attacks.jpeg(im, 70)),
        ("Q60", lambda im: attacks.jpeg(im, 60)),
        ("Q50", lambda im: attacks.jpeg(im, 50)),
        ("blur5", lambda im: attacks.gaussian_blur(im, 5)),
    ]

    print(f"{'强度':>6}{'重复':>6}{'PSNR':>8}{'最低':>7}{'SSIM':>8}" + "".join(f"{n:>9}" for n, _ in cases) + f"{'平均':>8}")
    for strength in args.strengths:
        for repeat in args.repeats:
            psnrs, ssims, bers = [], [], [[] for _ in cases]
            for name in IMAGES:
                img = cv2.imread(str(ROOT / f"assets/images/{name}.jpg"))
                ref = prepare_reference(img)
                wm, _ = embed(img, bits, strength=strength, max_repeat=repeat)
                psnrs.append(psnr(ref, wm))
                ssims.append(__import__("dwtm").metrics.ssim(ref, wm))
                for i, (_, fn) in enumerate(cases):
                    got = extract(fn(wm), bits.size, max_repeat=repeat).info_bits
                    bers[i].append(ber(bits, got))
            means = [float(np.mean(b)) for b in bers]
            print(
                f"{strength:>6.2f}{repeat:>6}{np.mean(psnrs):>8.2f}{min(psnrs):>7.2f}"
                f"{np.mean(ssims):>8.4f}" + "".join(f"{m:>9.3f}" for m in means) + f"{np.mean(means):>8.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
