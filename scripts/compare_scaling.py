# -*- coding: utf-8 -*-
"""尺度基准对照：local / global / geometric 三种强度归一化方式的折中。

用法：
    .venv\\Scripts\\python.exe scripts\\compare_scaling.py
"""

from __future__ import annotations

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
STRENGTHS = (0.5, 0.8, 1.1, 1.5, 2.0)


def main() -> int:
    bits = payload.text_to_bits("DEMO 水印 2027", 16)
    attacks_to_test = [
        ("Q80", lambda im: attacks.jpeg(im, 80)),
        ("Q70", lambda im: attacks.jpeg(im, 70)),
        ("Q60", lambda im: attacks.jpeg(im, 60)),
        ("gauss5", lambda im: attacks.gaussian_blur(im, 5)),
    ]

    print(f"{'模式':<11}{'强度':>6}{'PSNR均值':>10}{'最低PSNR':>10}" + "".join(f"{n:>9}" for n, _ in attacks_to_test))
    for mode in ("local", "geometric", "global"):
        for strength in STRENGTHS:
            psnrs, bers = [], [[] for _ in attacks_to_test]
            for name in IMAGES:
                img = cv2.imread(str(ROOT / f"assets/images/{name}.jpg"))
                ref = prepare_reference(img)
                wm, _ = embed(img, bits, strength=strength, scale_mode=mode)
                psnrs.append(psnr(ref, wm))
                for i, (_, fn) in enumerate(attacks_to_test):
                    got = extract(fn(wm), bits.size).info_bits
                    bers[i].append(ber(bits, got))
            means = "".join(f"{np.mean(b):>9.3f}" for b in bers)
            print(f"{mode:<11}{strength:>6.2f}{np.mean(psnrs):>10.2f}{min(psnrs):>10.2f}{means}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
