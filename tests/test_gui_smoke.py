# -*- coding: utf-8 -*-
"""界面冒烟测试：无显示环境下构造窗口并跑通「载入 -> 嵌入 -> 攻击 -> 提取」。

运行：.venv\\Scripts\\python.exe -m unittest tests.test_gui_smoke -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

try:
    from PyQt5.QtWidgets import QApplication

    from dwtm.gui import MainWindow
    from dwtm.payload import bits_to_image
    from dwtm.pipeline import prepare_reference

    HAS_QT = True
except Exception:  # noqa: BLE001
    HAS_QT = False

IMAGE = ROOT / "assets/images/lena.jpg"


@unittest.skipUnless(HAS_QT, "PyQt5 不可用")
class TestGuiSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.win = MainWindow()

    def tearDown(self):
        self.win.close()
        self.win.deleteLater()

    def test_load_embed_attack_extract(self):
        self.assertTrue(self.win._load_image(IMAGE))
        self.assertIsNotNone(self.win.orig)
        self.assertIn("4096", self.win.lbl_capacity.text())

        self.win.on_embed()
        self.assertIsNotNone(self.win.wm)
        self.assertNotEqual(self.win.metric_labels["psnr"].text(), "—")

        # 无攻击时提取正确
        self.win.on_extract()
        self.assertEqual(self.win.metric_labels["ber_count"].text(), "0 / 128")
        self.assertIn("DEMO", self.win.txt_result.toPlainText())

        # JPEG 攻击后仍能提取出文本
        self.win.combo_attack.setCurrentIndex(1)  # jpeg
        self.win.spin_attack.setValue(80)
        self.win.on_attack()
        self.assertIsNotNone(self.win.attacked)
        self.win.on_extract()
        self.assertLess(float(self.win.metric_labels["ber"].text()), 0.1)

    def test_binary_watermark_mode(self):
        self.assertTrue(self.win._load_image(IMAGE))
        self.win.combo_mode.setCurrentIndex(1)

        logo = np.zeros((32, 32), dtype=np.uint8)
        logo[6:26, 6:26] = 255
        logo[12:20, 12:20] = 0
        self.win.payload_image = logo
        self.win._on_payload_changed()
        self.assertIn("1024 bit", self.win.lbl_bits.text())

        self.win.on_embed()
        self.win.on_extract()
        self.assertEqual(self.win.metric_labels["ber_count"].text(), "0 / 1024")

        extracted = bits_to_image(self.win._last_extracted, (32, 32))
        self.assertTrue(np.array_equal(extracted, (logo > 128).astype(np.uint8) * 255))

    def test_remove_after_embed(self):
        self.assertTrue(self.win._load_image(IMAGE))
        self.win.on_embed()
        self.win.on_extract()
        self.assertEqual(self.win.metric_labels["ber_count"].text(), "0 / 128")

        self.win.on_remove()
        self.assertIsNotNone(self.win.removed)
        self.assertGreater(float(self.win.metric_labels["ber"].text()), 0.1)

    def test_reset_clears_state(self):
        self.assertTrue(self.win._load_image(IMAGE))
        self.win.on_embed()
        self.win.on_reset()
        self.assertIsNone(self.win.orig)
        self.assertIsNone(self.win.wm)
        self.assertEqual(self.win.lbl_file.text(), "未载入")

    def test_difference_view_shape(self):
        self.assertTrue(self.win._load_image(IMAGE))
        self.win.on_embed()
        ref = prepare_reference(self.win.orig)
        self.assertEqual(ref.shape, self.win.wm.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
