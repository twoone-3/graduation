# -*- coding: utf-8 -*-
"""批量鲁棒性实验：多幅图像 x 多类攻击 x 多种编码方案 -> CSV。

用法（项目根目录）：
    .venv\\Scripts\\python.exe scripts\\run_experiments.py
    .venv\\Scripts\\python.exe scripts\\run_experiments.py --codec none polar --strength 0.8 1.0
    .venv\\Scripts\\python.exe scripts\\run_experiments.py --images assets/images/*.jpg --limit 5

输出：
    results/experiments_<时间戳>.csv    每次"图像 x 攻击 x 参数"一行的明细
    results/summary_<时间戳>.csv        按攻击类型聚合的均值，可直接进论文表格
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from dwtm import attacks, payload  # noqa: E402
from dwtm.codes import make_codec  # noqa: E402
from dwtm.metrics import ber, nc, psnr, ssim  # noqa: E402
from dwtm.pipeline import embed, extract, prepare_reference  # noqa: E402

Attack = tuple[str, str, Callable[[np.ndarray], np.ndarray]]


def attack_suite() -> list[Attack]:
    """任务书要求的五类攻击 + 参照组。参数档位由轻到重。"""
    suite: list[Attack] = [("none", "-", lambda im: im)]

    for q in (90, 80, 70, 60, 50):
        suite.append(("jpeg", f"Q{q}", lambda im, q=q: attacks.jpeg(im, q)))

    for sigma in (1.0, 2.0, 3.0, 5.0):
        suite.append(
            ("noise", f"sigma{sigma:g}", lambda im, s=sigma: attacks.gaussian_noise(im, s, seed=2026))
        )

    suite.append(("blur", "gauss3", lambda im: attacks.gaussian_blur(im, 3)))
    suite.append(("blur", "gauss5", lambda im: attacks.gaussian_blur(im, 5)))
    suite.append(("median", "med3", lambda im: attacks.median_blur(im, 3)))

    for d in (10, 20, -10, -20):
        suite.append(("brightness", f"{d:+d}", lambda im, d=d: attacks.brightness(im, d)))

    for r in (0.05, 0.10, 0.20):
        suite.append(("crop", f"{int(r * 100)}%", lambda im, r=r: attacks.crop_fill(im, r)))

    suite.append(("rescale", "0.75x", lambda im: attacks.rescale(im, 0.75)))
    return suite


def load_images(patterns: Iterable[str], limit: int | None) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        p = Path(pattern)
        if not p.is_absolute():
            p = ROOT / pattern
        matches = sorted(Path(m) for m in glob.glob(str(p))) if any(ch in str(p) for ch in "*?[") else [p]
        paths.extend(m for m in matches if m.is_file())
    unique = [p for i, p in enumerate(paths) if p not in paths[:i]]
    return unique[:limit] if limit else unique


def main() -> int:
    ap = argparse.ArgumentParser(description="DFT 水印批量鲁棒性实验")
    ap.add_argument(
        "--images",
        nargs="+",
        default=[
            "assets/images/lena.jpg",
            "assets/images/baboon.jpg",
            "assets/images/fruits.jpg",
            "assets/images/board.jpg",
            "assets/images/butterfly.jpg",
        ],
    )
    ap.add_argument("--limit", type=int, default=5, help="最多使用几幅图像（默认 5）")
    ap.add_argument("--codec", nargs="+", default=["none"])
    ap.add_argument("--strength", nargs="+", type=float, default=[1.2])
    ap.add_argument("--text", default="DEMO 水印 2027")
    ap.add_argument("--bytes", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=768, help="超过该边长的图先缩放，控制耗时")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    args = ap.parse_args()

    paths = load_images(args.images, args.limit)
    if not paths:
        raise SystemExit("没有找到测试图像")

    bits = payload.text_to_bits(args.text, args.bytes)
    suite = attack_suite()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"experiments_{stamp}.csv"
    summary_path = out_dir / f"summary_{stamp}.csv"

    rows: list[dict[str, object]] = []
    started = time.time()

    for strength in args.strength:
        for codec_name in args.codec:
            codec = make_codec(codec_name)
            for path in paths:
                img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"跳过（无法读取）: {path}")
                    continue
                if max(img.shape[:2]) > args.max_side:
                    scale = args.max_side / max(img.shape[:2])
                    img = cv2.resize(
                        img,
                        (int(round(img.shape[1] * scale)), int(round(img.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA,
                    )

                wm, info = embed(img, bits, codec=codec, strength=strength)
                ref = prepare_reference(img)
                cover_psnr, cover_ssim = psnr(ref, wm), ssim(ref, wm)

                for attack_name, param, fn in suite:
                    attacked = fn(wm)
                    res = extract(attacked, bits.size, codec=codec)
                    rows.append(
                        {
                            "image": path.stem,
                            "codec": codec.name,
                            "strength": strength,
                            "attack": attack_name,
                            "param": param,
                            "psnr": round(cover_psnr, 3),
                            "ssim": round(cover_ssim, 4),
                            "attack_psnr": round(psnr(wm, attacked), 3),
                            "ber": round(ber(bits, res.info_bits), 5),
                            "nc": round(nc(bits, res.info_bits), 5),
                            "repeat": info.repeat,
                            "n_blocks": info.n_blocks,
                            "n_info_bits": info.n_info_bits,
                            "n_coded_bits": info.n_coded_bits,
                        }
                    )

    fieldnames = list(rows[0].keys())
    with detail_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 按 (codec, strength, attack, param) 聚合
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        key = (r["codec"], r["strength"], r["attack"], r["param"])
        groups.setdefault(key, []).append(r)

    summary = []
    for (codec_name, strength, attack_name, param), items in groups.items():
        summary.append(
            {
                "codec": codec_name,
                "strength": strength,
                "attack": attack_name,
                "param": param,
                "psnr": round(float(np.mean([i["psnr"] for i in items])), 3),
                "ssim": round(float(np.mean([i["ssim"] for i in items])), 4),
                "attack_psnr": round(float(np.mean([i["attack_psnr"] for i in items])), 3),
                "ber": round(float(np.mean([i["ber"] for i in items])), 5),
                "nc": round(float(np.mean([i["nc"] for i in items])), 5),
                "n_images": len(items),
            }
        )

    with summary_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print(f"图像: {', '.join(p.stem for p in paths)}")
    print(f"共 {len(rows)} 条明细, 用时 {time.time() - started:.1f}s\n")

    for codec_name, strength in {(r['codec'], r['strength']) for r in rows}:
        sub = [s for s in summary if s["codec"] == codec_name and s["strength"] == strength]
        print(f"== codec={codec_name}  strength={strength} ==")
        print(f"{'攻击':<12}{'参数':>9}{'PSNR':>8}{'SSIM':>8}{'攻击PSNR':>10}{'BER':>9}{'NC':>9}")
        for s in sub:
            print(
                f"{s['attack']:<12}{s['param']:>9}{s['psnr']:>8.2f}{s['ssim']:>8.4f}"
                f"{s['attack_psnr']:>10.2f}{s['ber']:>9.4f}{s['nc']:>9.4f}"
            )
        print()

    print(f"明细: {detail_path}")
    print(f"汇总: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
