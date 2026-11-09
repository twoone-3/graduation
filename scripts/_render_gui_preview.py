# -*- coding: utf-8 -*-
"""渲染主窗口截图，用于检查排版（不在正式流程中使用）。

    .venv\\Scripts\\python.exe scripts\\_render_gui_preview.py [text|binary] [宽] [高]

默认按用户机器的逻辑屏幕（1280x764）渲染；传 980 600 可检查最小尺寸。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from PyQt5.QtGui import QFont  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from dwtm.gui import STYLE, MainWindow  # noqa: E402

mode = sys.argv[1] if len(sys.argv) > 1 else "text"
width = int(sys.argv[2]) if len(sys.argv) > 2 else 1280
height = int(sys.argv[3]) if len(sys.argv) > 3 else 764

app = QApplication([])
app.setFont(QFont("Microsoft YaHei", 9))
app.setStyleSheet(STYLE)

win = MainWindow()
win.resize(width, height)
win.show()
app.processEvents()

win._load_image(ROOT / "assets/images/lena.jpg")

if mode == "blank":
    app.processEvents()
    out = ROOT / "results" / "gui_preview_blank.png"
    win.grab().save(str(out))
    print(out)
    raise SystemExit(0)

if mode == "advanced":
    win.btn_advanced.setChecked(True)

if mode == "binary":
    win.combo_mode.setCurrentIndex(1)
    logo = np.zeros((32, 32), dtype=np.uint8)
    logo[4:28, 10:22] = 255
    logo[10:22, 4:28] = 255
    logo[13:19, 13:19] = 0
    win.payload_image = logo
    win._on_payload_changed()

win.on_embed()
win.combo_attack.setCurrentIndex(2)  # noise
win.spin_attack.setValue(2.0)
win.on_attack()
win.on_extract()
if mode == "remove":
    win.on_remove()
app.processEvents()

suffix = "" if (width, height) == (1280, 764) else f"_{width}x{height}"
out = ROOT / "results" / f"gui_preview_{mode}{suffix}.png"
win.grab().save(str(out))
print(out)
