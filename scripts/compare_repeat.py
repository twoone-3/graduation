# -*- coding: utf-8 -*-
"""重复次数（嵌入块密度）对不可见性与鲁棒性的影响。

稀疏嵌入：只随机挑选 repeat * n_bits 个块承载水印，其余块保持原样。
重复次数越多 -> 每个比特的副本越多、抗噪越好，但改动块也越多、PSNR 越低。

用法：
    .venv\\Scripts\\python.exe scripts\\compare_repeat.py
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
from dwtm.metrics import ber, psnr, ssim  # noqa: E402
from dwtm.pipeline import embed, extract, prepare_reference  # noqa: E402

IMAGES = ["lena", "baboon", "fruits", "board", "butterfly"]


def main() -> int:
    bits = payload.text_to_bits("DEMO 水印 2027", 16)
    cases = [
        ("Q80", lambda im: attacks.jpeg(im, 80)),
        ("Q60", lambda im: attacks.jpeg(im, 60)),
        ("Q50", lambda im: attacks.jpeg(im, 50)),
        ("gauss5", lambda im: attacks.gaussian_blur(im, 5)),
    ]

    print(f"{'重复':>6}{'改动块':>8}{'PSNR':>8}{'SSIM':>8}" + "".join(f"{n:>9}" for n, _ in cases))
    for max_repeat in (2, 4, 8, 16, 32, None):
        psnrs, ssims, bers = [], [], [[] for _ in cases]
        for name in IMAGES:
            img = cv2.imread(str(ROOT / f"assets/images/{name}.jpg"))
            ref = prepare_reference(img)
            wm, info = embed(img, bits, max_repeat=max_repeat)
            psnrs.append(psnr(ref, wm))
            ssims.append(ssim(ref, wm))
            for i, (_, fn) in enumerate(cases):
                got = extract(fn(wm), bits.size, max_repeat=max_repeat).info_bits
                bers[i].append(ber(bits, got))
        means = "".join(f"{np.mean(b):>9.3f}" for b in bers)
        label = "全部" if max_repeat is None else str(max_repeat)
        print(f"{label:>6}{info.used_blocks:>8}{np.mean(psnrs):>8.2f}{np.mean(ssims):>8.4f}{means}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
