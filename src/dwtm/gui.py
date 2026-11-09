# -*- coding: utf-8 -*-
"""PyQt5 图形界面：加水印 / 去水印演示。

主流程只有四步：打开图片 -> 输入水印 -> 加水印 -> 去水印（提取）。
嵌入强度、重复次数、选块种子、攻击模拟等放在"高级设置"里，默认折叠，
演示时界面上只留最必要的元素。

窗口尺寸按屏幕可用区域自适应（本机高 DPI 下逻辑分辨率 1280x764），
不再使用写死的像素尺寸。

启动：
    .venv\\Scripts\\python.exe scripts\\run_gui.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import attacks, payload
from .codes import make_codec
from .io_utils import imread, imwrite
from .metrics import ber, nc, psnr, ssim
from .pipeline import capacity_bits, embed, extract, prepare_reference, remove

APP_TITLE = "数字图像水印演示"

STYLE = """
QWidget { font-family: "Microsoft YaHei", "Segoe UI"; font-size: 12px; color: #1f2328; }
QMainWindow, QWidget#root { background: #f4f6f9; }
QLabel#appTitle { font-size: 16px; font-weight: 700; color: #111827; }
QLabel#appSubtitle { font-size: 11px; color: #7b8494; }
QGroupBox {
    background: #ffffff; border: 1px solid #e3e7ee; border-radius: 8px;
    margin-top: 8px; padding: 6px 9px 6px 9px; font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #4b5563; }
QPushButton {
    background: #ffffff; border: 1px solid #d6dbe3; border-radius: 6px;
    padding: 5px 12px; color: #374151;
}
QPushButton:hover { border-color: #a9b4c4; background: #fafbfc; }
QPushButton:pressed { background: #eef1f5; }
QPushButton:disabled { color: #b3bac5; border-color: #eceff3; background: #fafbfc; }
QPushButton#primary { background: #1f6feb; border: none; color: #ffffff; font-weight: 600; }
QPushButton#primary:hover { background: #1a5fd0; }
QPushButton#primary:disabled { background: #b8c9e6; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
    background: #ffffff; border: 1px solid #d6dbe3; border-radius: 6px; padding: 3px 7px;
    selection-background-color: #1f6feb;
}
QComboBox::drop-down { border: none; width: 16px; }
QToolButton { border: none; color: #1f6feb; font-weight: 600; padding: 3px 2px; text-align: left; }
QToolButton:hover { color: #1a5fd0; }
QLabel#metricName { color: #7b8494; }
QLabel#metricValue { font-family: Consolas, "Courier New"; font-weight: 600; color: #111827; }
QLabel#chipName { color: #8b94a3; font-size: 10px; }
QLabel#chipValue { font-family: Consolas, "Courier New"; font-weight: 700; font-size: 13px; color: #111827; }
QLabel#hint { color: #8b94a3; font-size: 11px; }
QLabel#status { color: #5b6472; font-size: 11px; }
QLabel#view { background: #ffffff; border: 1px solid #e3e7ee; border-radius: 8px; color: #a7b0bd; }
"""

# 攻击类型 -> 参数控件配置
ATTACK_SPECS: dict[str, dict] = {
    "none": dict(label="无攻击", lo=0.0, hi=0.0, step=1.0, dec=0, suffix="", default=0.0),
    "jpeg": dict(label="JPEG 压缩", lo=1, hi=100, step=1.0, dec=0, suffix=" 质量", default=70),
    "noise": dict(label="高斯噪声", lo=0.5, hi=50.0, step=0.5, dec=1, suffix=" σ", default=3.0),
    "blur": dict(label="高斯模糊", lo=3, hi=15, step=2.0, dec=0, suffix=" 核", default=3.0),
    "median": dict(label="中值滤波", lo=3, hi=9, step=2.0, dec=0, suffix=" 核", default=3.0),
    "brightness": dict(label="亮度偏移", lo=-100, hi=100, step=5.0, dec=0, suffix=" 级", default=20.0),
    "crop": dict(label="裁剪填充", lo=0.01, hi=0.9, step=0.01, dec=2, suffix=" 比例", default=0.10),
    "rescale": dict(label="缩放往返", lo=0.1, hi=0.99, step=0.05, dec=2, suffix=" ×", default=0.75),
}


# --------------------------------------------------------------------------- #
# 工具函数
# --------------------------------------------------------------------------- #
def to_qpixmap(img: np.ndarray) -> QPixmap:
    """numpy(BGR / BGRA / gray) -> QPixmap。"""
    arr = np.ascontiguousarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data, w, h, w, QImage.Format_Grayscale8)
    else:
        h, w = arr.shape[:2]
        if arr.shape[2] == 4:
            qimg = QImage(arr.data, w, h, 4 * w, QImage.Format_ARGB32)
        else:
            qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_BGR888)
    return QPixmap.fromImage(qimg.copy())


def difference_view(reference: np.ndarray, watermarked: np.ndarray, gain: float = 16.0) -> np.ndarray:
    """差分图（放大并着色），用于展示改动分布。"""
    diff = np.abs(watermarked.astype(np.int16) - reference.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.max(axis=2)
    amp = np.clip(diff.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(amp, cv2.COLORMAP_INFERNO)


def apply_attack(img: np.ndarray, key: str, value: float) -> np.ndarray:
    if key == "none":
        return img.copy()
    if key == "jpeg":
        return attacks.jpeg(img, int(round(value)))
    if key == "noise":
        return attacks.gaussian_noise(img, float(value), seed=2026)
    if key == "blur":
        return attacks.gaussian_blur(img, int(round(value)))
    if key == "median":
        return attacks.median_blur(img, int(round(value)))
    if key == "brightness":
        return attacks.brightness(img, float(value))
    if key == "crop":
        return attacks.crop_fill(img, float(value))
    if key == "rescale":
        return attacks.rescale(img, float(value))
    raise ValueError(f"未知攻击: {key}")


class ImagePanel(QWidget):
    """带标题的等比缩放图像显示区。"""

    def __init__(self, title: str, placeholder: str = "（未载入）", parent: QWidget | None = None):
        super().__init__(parent)
        self.placeholder = placeholder
        self._pixmap: QPixmap | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("metricName")

        self.view = QLabel(placeholder)
        self.view.setObjectName("view")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(180, 120)

        layout.addWidget(self.title_label)
        layout.addWidget(self.view, 1)

    def set_image(self, img: np.ndarray | None) -> None:
        if img is None:
            self._pixmap = None
            self.view.clear()
            self.view.setText(self.placeholder)
            return
        self._pixmap = to_qpixmap(img)
        self._render()

    def _render(self) -> None:
        if self._pixmap is None:
            return
        self.view.setPixmap(
            self._pixmap.scaled(self.view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt 命名)
        super().resizeEvent(event)
        self._render()


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_TITLE)

        self.orig: np.ndarray | None = None
        self.wm: np.ndarray | None = None
        self.attacked: np.ndarray | None = None
        self.removed: np.ndarray | None = None
        self.reference: np.ndarray | None = None
        self.bits: np.ndarray | None = None
        self.payload_image: np.ndarray | None = None
        self.embed_params: dict | None = None
        self._last_extracted: np.ndarray | None = None
        self.metric_labels: dict[str, QLabel] = {}

        self._build_ui()
        self._fit_to_screen()
        self._on_mode_changed()
        self._on_attack_changed()
        self._on_payload_changed()
        self._update_buttons()
        self.log("就绪：先「打开图片」，输入水印内容，然后「加水印」。")

    # ---------------- 尺寸自适应 ----------------
    def _fit_to_screen(self) -> None:
        screen = QApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else None
        if area is None:
            self.resize(1180, 720)
            return
        width = min(1240, area.width() - 16)
        height = min(760, area.height() - 16)
        self.resize(width, height)
        self.setMinimumSize(min(980, width), min(600, height))
        x = area.x() + max(0, (area.width() - width) // 2)
        y = area.y() + max(0, (area.height() - height) // 2)
        self.move(x, y)

    def _ensure_on_screen(self) -> None:
        """show() 之后再按窗口外框（含标题栏/边框）校正一次。

        resize() 只设置客户区大小，Windows 的标题栏和边框会额外占用高度，
        只按 availableGeometry 算客户区仍可能顶出屏幕。
        """
        screen = QApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        frame = self.frameGeometry()
        if frame.width() > area.width() or frame.height() > area.height():
            self.resize(
                min(self.width(), area.width() - 32),
                min(self.height(), area.height() - 56),
            )
            frame = self.frameGeometry()
        self.move(
            area.x() + max(0, (area.width() - frame.width()) // 2),
            area.y() + max(0, (area.height() - frame.height()) // 2),
        )

    # ---------------- 界面搭建 ----------------
    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        outer.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_preview_area())
        splitter.addWidget(self._build_control_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([860, 330])
        outer.addWidget(splitter, 1)

        outer.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(8)

        titles = QVBoxLayout()
        titles.setSpacing(0)
        title = QLabel(APP_TITLE)
        title.setObjectName("appTitle")
        subtitle = QLabel("基于离散傅里叶变换（DFT）的分块中频系数嵌入与提取")
        subtitle.setObjectName("appSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        h.addLayout(titles)
        h.addStretch(1)

        self.btn_open = QPushButton("打开图片")
        self.btn_save_wm = QPushButton("保存图片")
        self.btn_save_ext = QPushButton("保存结果")
        self.btn_reset = QPushButton("重置")
        self.btn_embed = QPushButton("加水印")
        self.btn_embed.setObjectName("primary")
        self.btn_embed.setMinimumWidth(96)
        self.btn_extract = QPushButton("提取信息")
        self.btn_extract.setMinimumWidth(96)
        self.btn_remove = QPushButton("去水印")
        self.btn_remove.setMinimumWidth(96)

        for btn in (self.btn_open, self.btn_save_wm, self.btn_save_ext):
            h.addWidget(btn)
        h.addSpacing(6)
        h.addWidget(self.btn_embed)
        h.addWidget(self.btn_extract)
        h.addWidget(self.btn_remove)
        h.addSpacing(6)
        h.addWidget(self.btn_reset)

        self.btn_open.clicked.connect(self.on_open_image)
        self.btn_save_wm.clicked.connect(self.on_save_watermarked)
        self.btn_save_ext.clicked.connect(self.on_save_extracted)
        self.btn_reset.clicked.connect(self.on_reset)
        self.btn_embed.clicked.connect(self.on_embed)
        self.btn_extract.clicked.connect(self.on_extract)
        self.btn_remove.clicked.connect(self.on_remove)
        return box

    def _build_preview_area(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)

        row = QWidget()
        grid = QHBoxLayout(row)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(8)

        self.panel_orig = ImagePanel("原图", "打开一张图片开始")
        self.panel_wm = ImagePanel("含水印图像", "点击「加水印」后显示")
        self.panel_processed = ImagePanel("处理结果", "（去水印 / 攻击后显示）")
        self.panel_processed.setVisible(False)
        self.panel_diff = ImagePanel("差异图（×16）", "点击「加水印」后显示改动分布")
        self.panel_diff.setVisible(False)

        grid.addWidget(self.panel_orig, 1)
        grid.addWidget(self.panel_wm, 1)
        grid.addWidget(self.panel_processed, 1)
        grid.addWidget(self.panel_diff, 1)
        v.addWidget(row, 1)

        card = self._build_result_card()
        card.setFixedHeight(132)
        v.addWidget(card)
        return box

    def _build_result_card(self) -> QGroupBox:
        group = QGroupBox("提取结果")
        h = QHBoxLayout(group)
        h.setContentsMargins(8, 4, 8, 6)
        h.setSpacing(8)

        info = QVBoxLayout()
        info.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.lbl_result_hint = QLabel("点击「去水印」后显示提取出的水印")
        self.lbl_result_hint.setObjectName("hint")
        head.addWidget(self.lbl_result_hint, 1)

        self.lbl_ber_count = QLabel("—")
        self.lbl_ber_count.setObjectName("hint")
        self.metric_labels["ber_count"] = self.lbl_ber_count
        head.addWidget(self.lbl_ber_count)
        info.addLayout(head)

        self.txt_result = QPlainTextEdit()
        self.txt_result.setReadOnly(True)
        self.txt_result.setMinimumHeight(52)
        info.addWidget(self.txt_result, 1)
        h.addLayout(info, 1)

        self.lbl_result_img = QLabel()
        self.lbl_result_img.setObjectName("view")
        self.lbl_result_img.setAlignment(Qt.AlignCenter)
        self.lbl_result_img.setFixedSize(96, 96)
        self.lbl_result_img.setVisible(False)
        h.addWidget(self.lbl_result_img)
        return group

    def _build_control_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(360)

        inner = QWidget()
        v = QVBoxLayout(inner)
        v.setContentsMargins(2, 0, 6, 0)
        v.setSpacing(6)

        v.addWidget(self._build_image_group())
        v.addWidget(self._build_payload_group())

        self.btn_advanced = QToolButton()
        self.btn_advanced.setText("高级设置")
        self.btn_advanced.setCheckable(True)
        self.btn_advanced.setChecked(False)
        self.btn_advanced.setArrowType(Qt.RightArrow)
        self.btn_advanced.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.btn_advanced.toggled.connect(self._toggle_advanced)

        self.advanced = QWidget()
        adv = QVBoxLayout(self.advanced)
        adv.setContentsMargins(0, 0, 0, 0)
        adv.setSpacing(4)
        adv.addWidget(self._build_param_group())
        adv.addWidget(self._build_attack_group())
        self.advanced.setVisible(False)

        v.addWidget(self.btn_advanced)
        v.addWidget(self.advanced)
        v.addStretch(1)

        scroll.setWidget(inner)
        return scroll

    def _build_image_group(self) -> QGroupBox:
        group = QGroupBox("图片")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.lbl_file = QLabel("未载入")
        self.lbl_file.setObjectName("hint")
        self.lbl_file.setWordWrap(True)
        self.lbl_size = QLabel("—")
        self.lbl_capacity = QLabel("—")
        self.lbl_capacity.setObjectName("hint")
        self.lbl_capacity.setWordWrap(True)

        form.addRow("文件", self.lbl_file)
        form.addRow("尺寸", self.lbl_size)
        form.addRow("容量", self.lbl_capacity)
        return group

    def _build_payload_group(self) -> QGroupBox:
        group = QGroupBox("水印内容")
        v = QVBoxLayout(group)
        v.setSpacing(4)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["短文本", "二值图片 (32×32)"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)

        self.edit_text = QLineEdit("DEMO 水印 2027")
        self.edit_text.textChanged.connect(self._on_payload_changed)

        self.spin_bytes = QSpinBox()
        self.spin_bytes.setRange(4, 64)
        self.spin_bytes.setValue(16)
        self.spin_bytes.setSuffix(" 字节")
        self.spin_bytes.valueChanged.connect(self._on_payload_changed)

        self.btn_pick_bmp = QPushButton("选择水印图片…")
        self.btn_pick_bmp.clicked.connect(self.on_pick_payload_image)
        self.lbl_bmp = QLabel("未选择")
        self.lbl_bmp.setObjectName("hint")

        self.lbl_bits = QLabel("—")
        self.lbl_bits.setObjectName("hint")
        self.lbl_bits.setWordWrap(True)

        v.addWidget(self.combo_mode)
        v.addWidget(self.edit_text)
        v.addWidget(self.spin_bytes)
        v.addWidget(self.btn_pick_bmp)
        v.addWidget(self.lbl_bmp)
        v.addWidget(self.lbl_bits)
        return group

    def _build_param_group(self) -> QGroupBox:
        group = QGroupBox("嵌入参数")
        form = QFormLayout(group)
        form.setSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.spin_strength = QDoubleSpinBox()
        self.spin_strength.setRange(0.1, 3.0)
        self.spin_strength.setSingleStep(0.1)
        self.spin_strength.setDecimals(1)
        self.spin_strength.setValue(1.2)

        self.spin_repeat = QSpinBox()
        self.spin_repeat.setRange(1, 64)
        self.spin_repeat.setValue(8)

        self.chk_full = QCheckBox("用满全部块")
        self.chk_full.toggled.connect(lambda on: self.spin_repeat.setEnabled(not on))

        repeat_row = QWidget()
        repeat_layout = QHBoxLayout(repeat_row)
        repeat_layout.setContentsMargins(0, 0, 0, 0)
        repeat_layout.setSpacing(6)
        repeat_layout.addWidget(self.spin_repeat)
        repeat_layout.addWidget(self.chk_full, 1)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999_999)
        self.spin_seed.setValue(2027)

        self.combo_codec = QComboBox()
        self.combo_codec.addItems(["无编码（基线）", "Polar (256,128) · 未接入"])

        form.addRow("强度 α", self.spin_strength)
        form.addRow("重复次数", repeat_row)
        form.addRow("选块种子", self.spin_seed)
        form.addRow("信道编码", self.combo_codec)
        return group

    def _build_attack_group(self) -> QGroupBox:
        group = QGroupBox("攻击模拟")
        v = QVBoxLayout(group)
        v.setSpacing(4)

        self.combo_attack = QComboBox()
        for key, spec in ATTACK_SPECS.items():
            self.combo_attack.addItem(spec["label"], key)
        self.combo_attack.currentIndexChanged.connect(self._on_attack_changed)

        self.spin_attack = QDoubleSpinBox()
        self.spin_attack.setDecimals(0)

        self.btn_attack = QPushButton("施加攻击")
        self.btn_attack.clicked.connect(self.on_attack)
        self.lbl_attack_psnr = QLabel("—")
        self.lbl_attack_psnr.setObjectName("hint")

        v.addWidget(self.combo_attack)
        v.addWidget(self.spin_attack)
        v.addWidget(self.btn_attack)
        v.addWidget(self.lbl_attack_psnr)
        return group

    def _build_footer(self) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(10)

        self.lbl_status = QLabel("")
        self.lbl_status.setObjectName("status")
        h.addWidget(self.lbl_status, 1)

        for key, name in (
            ("psnr", "PSNR"),
            ("ssim", "SSIM"),
            ("ber", "BER"),
            ("nc", "NC"),
            ("attack_psnr", "攻击 PSNR"),
        ):
            h.addWidget(self._make_chip(name, key))
        return box

    def _make_chip(self, name: str, key: str) -> QWidget:
        chip = QWidget()
        chip.setMinimumWidth(80)
        v = QVBoxLayout(chip)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        name_label = QLabel(name)
        name_label.setObjectName("chipName")
        value_label = QLabel("—")
        value_label.setObjectName("chipValue")
        self.metric_labels[key] = value_label

        v.addWidget(name_label)
        v.addWidget(value_label)
        return chip

    def _toggle_advanced(self, on: bool) -> None:
        self.advanced.setVisible(on)
        self.btn_advanced.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    # ---------------- 状态与提示 ----------------
    def log(self, message: str) -> None:
        if hasattr(self, "lbl_status"):
            self.lbl_status.setText(f"{message}")

    def _set_metric(self, key: str, text: str) -> None:
        if key in self.metric_labels:
            self.metric_labels[key].setText(text)

    def _clear_metrics(self) -> None:
        for key in self.metric_labels:
            self._set_metric(key, "—")
        self.txt_result.clear()
        self.lbl_result_hint.setText("点击「提取信息」读出文本，或「去水印」抹除水印")
        self.lbl_result_img.clear()
        self.lbl_attack_psnr.setText("—")

    def _update_buttons(self) -> None:
        self.btn_embed.setEnabled(self.orig is not None)
        self.btn_extract.setEnabled(self.wm is not None)
        self.btn_remove.setEnabled(self.wm is not None)
        self.btn_attack.setEnabled(self.wm is not None)
        self.btn_save_wm.setEnabled(self.wm is not None)
        self.btn_save_ext.setEnabled(self._last_extracted is not None)

    # ---------------- 输入项联动 ----------------
    def _on_mode_changed(self) -> None:
        is_text = self.combo_mode.currentIndex() == 0
        for widget in (self.edit_text, self.spin_bytes):
            widget.setVisible(is_text)
        self.btn_pick_bmp.setVisible(not is_text)
        self.lbl_bmp.setVisible(not is_text)
        self._on_payload_changed()

    def _on_payload_changed(self) -> None:
        try:
            bits = self._payload_bits()
            self.lbl_bits.setText(f"{bits.size} bit（{bits.size / 8:.0f} 字节）")
        except ValueError as exc:
            self.lbl_bits.setText(str(exc))

    def _on_attack_changed(self) -> None:
        key = self.combo_attack.currentData()
        spec = ATTACK_SPECS[key]
        self.spin_attack.blockSignals(True)
        self.spin_attack.setDecimals(spec["dec"])
        self.spin_attack.setRange(spec["lo"], spec["hi"])
        self.spin_attack.setSingleStep(spec["step"])
        self.spin_attack.setSuffix(spec["suffix"])
        self.spin_attack.setValue(spec["default"])
        self.spin_attack.setEnabled(key != "none")
        self.spin_attack.blockSignals(False)

    def _payload_bits(self) -> np.ndarray:
        if self.combo_mode.currentIndex() == 0:
            return payload.text_to_bits(self.edit_text.text(), self.spin_bytes.value())
        if self.payload_image is None:
            raise ValueError("请先选择二值水印图片")
        return payload.image_to_bits(self.payload_image, (32, 32))

    def _codec(self):
        if self.combo_codec.currentIndex() == 1:
            raise ValueError("Polar 码尚未接入，请先使用无编码方案")
        return make_codec("none")

    # ---------------- 打开 / 保存 ----------------
    def on_open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", str(Path.cwd()), "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path: str | Path) -> bool:
        """载入载体图像（界面与测试共用，跳过文件对话框）。"""
        img = imread(path)
        if img is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图像：\n{path}")
            return False
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]
        if max(img.shape[:2]) > 1600:
            scale = 1600 / max(img.shape[:2])
            img = cv2.resize(
                img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA
            )

        self.orig = img
        self.wm = None
        self.attacked = None
        self.removed = None
        self.bits = None
        self._last_extracted = None
        self.reference = prepare_reference(img)

        self.panel_orig.set_image(img)
        self.panel_wm.set_image(None)
        self.panel_processed.set_image(None)
        self.panel_processed.setVisible(False)
        self.panel_diff.set_image(None)
        self.panel_diff.setVisible(False)
        self._clear_metrics()

        h, w = img.shape[:2]
        cap = capacity_bits(img.shape)
        self.lbl_file.setText(Path(path).name)
        self.lbl_size.setText(f"{w} × {h}")
        self.lbl_capacity.setText(f"{cap} bit（≈{cap // 8} 字节）")
        self._on_payload_changed()
        self._update_buttons()
        self.log(f"已载入 {Path(path).name}（{w}×{h}），可嵌入 {cap} bit")
        return True

    def on_pick_payload_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择水印图片", str(Path.cwd()), "图像文件 (*.png *.jpg *.jpeg *.bmp)"
        )
        if not path:
            return
        img = imread(path)
        if img is None:
            QMessageBox.warning(self, "读取失败", f"无法读取图像：\n{path}")
            return
        self.payload_image = img
        self.lbl_bmp.setText(Path(path).name)
        self._on_payload_changed()
        self.log(f"已选择水印图片：{Path(path).name}（会缩放到 32×32 并二值化）")

    def on_save_watermarked(self) -> None:
        image = self.removed if self.removed is not None else self.wm
        if image is None:
            return
        default = "removed.png" if self.removed is not None else "watermarked.png"
        path, _ = QFileDialog.getSaveFileName(
            self, "保存图片", default, "PNG 图像 (*.png);;BMP 图像 (*.bmp);;JPEG 图像 (*.jpg)"
        )
        if not path:
            return
        if imwrite(path, image):
            self.log(f"已保存图片：{Path(path).name}")
        else:
            QMessageBox.warning(self, "保存失败", f"无法写入：\n{path}")

    def on_save_extracted(self) -> None:
        if self._last_extracted is None:
            QMessageBox.information(self, "提示", "请先执行「提取信息」")
            return
        if self.combo_mode.currentIndex() == 0:
            path, _ = QFileDialog.getSaveFileName(self, "保存提取的文本", "extracted.txt", "文本 (*.txt)")
            if not path:
                return
            Path(path).write_text(payload.bits_to_text(self._last_extracted), encoding="utf-8")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "保存提取的水印图", "extracted.png", "PNG 图像 (*.png)")
            if not path:
                return
            imwrite(path, payload.bits_to_image(self._last_extracted, (32, 32)))
        self.log(f"已保存提取结果：{Path(path).name}")

    def on_reset(self) -> None:
        self.orig = self.wm = self.attacked = self.removed = self.bits = None
        self.reference = None
        self.payload_image = None
        self.embed_params = None
        self._last_extracted = None
        for panel in (self.panel_orig, self.panel_wm, self.panel_processed, self.panel_diff):
            panel.set_image(None)
        self.panel_processed.setVisible(False)
        self.panel_diff.setVisible(False)
        self._clear_metrics()
        self.lbl_file.setText("未载入")
        self.lbl_size.setText("—")
        self.lbl_capacity.setText("—")
        self.lbl_bmp.setText("未选择")
        self._on_payload_changed()
        self._update_buttons()
        self.log("已重置。")

    # ---------------- 核心操作 ----------------
    def on_embed(self) -> None:
        if self.orig is None:
            return
        try:
            bits = self._payload_bits()
            codec = self._codec()
        except ValueError as exc:
            QMessageBox.warning(self, "水印内容有误", str(exc))
            return

        max_repeat = None if self.chk_full.isChecked() else self.spin_repeat.value()
        seed = self.spin_seed.value()
        strength = self.spin_strength.value()

        try:
            wm, info = embed(
                self.orig, bits, codec=codec, strength=strength, max_repeat=max_repeat, seed=seed
            )
        except ValueError as exc:
            QMessageBox.warning(self, "加水印失败", str(exc))
            return

        self.bits = bits
        self.wm = wm
        self.attacked = None
        self.removed = None
        self._last_extracted = None
        self.reference = prepare_reference(self.orig)
        self.embed_params = dict(codec="none", max_repeat=max_repeat, seed=seed)

        self.panel_wm.set_image(wm)
        self.panel_processed.set_image(None)
        self.panel_processed.setVisible(False)
        self.panel_diff.set_image(difference_view(self.reference, wm))
        self.panel_diff.setVisible(True)
        self.lbl_attack_psnr.setText("—")
        self._set_metric("attack_psnr", "—")
        for key in ("ber", "nc"):
            self._set_metric(key, "—")
        self.txt_result.clear()
        self.lbl_result_img.clear()
        self.lbl_result_hint.setText("点击「提取信息」读出文本，或「去水印」抹除水印")

        p = psnr(self.reference, wm)
        s = ssim(self.reference, wm)
        self._set_metric("psnr", f"{p:.2f} dB")
        self._set_metric("ssim", f"{s:.4f}")
        self._update_buttons()
        self.log(
            f"加水印完成：{bits.size} bit，重复 {info.repeat} 次，PSNR {p:.2f} dB / SSIM {s:.4f}"
        )

    def on_attack(self) -> None:
        if self.wm is None:
            return
        key = self.combo_attack.currentData()
        value = self.spin_attack.value()

        try:
            attacked = apply_attack(self.wm, key, value)
        except Exception as exc:  # noqa: BLE001 界面层兜底
            QMessageBox.warning(self, "攻击参数错误", str(exc))
            return

        self.attacked = attacked
        self.panel_processed.title_label.setText("攻击后的图像")
        self.panel_processed.set_image(attacked)
        self.panel_processed.setVisible(True)

        p = psnr(self.wm, attacked)
        text = "∞" if not np.isfinite(p) else f"{p:.2f}"
        self.lbl_attack_psnr.setText(f"{text} dB")
        self._set_metric("attack_psnr", f"{text} dB")
        self.log(f"已施加攻击：{ATTACK_SPECS[key]['label']}（{value:g}）")

    def on_extract(self) -> None:
        if self.wm is None or self.bits is None or self.embed_params is None:
            return

        target = self.attacked if self.attacked is not None else self.wm
        codec = make_codec(self.embed_params["codec"])

        res = extract(
            target,
            self.bits.size,
            codec=codec,
            max_repeat=self.embed_params["max_repeat"],
            seed=self.embed_params["seed"],
        )
        self._last_extracted = res.info_bits

        b = ber(self.bits, res.info_bits)
        c = nc(self.bits, res.info_bits)
        errors = int(np.sum(self.bits != res.info_bits))
        self._set_metric("ber", f"{b:.4f}")
        self._set_metric("nc", f"{c:.4f}")
        self._set_metric("ber_count", f"{errors} / {self.bits.size}")

        if self.combo_mode.currentIndex() == 0:
            self.txt_result.setPlainText(payload.bits_to_text(res.info_bits))
            self.lbl_result_img.setVisible(False)
            self.lbl_result_hint.setText(
                f"提取出的水印文本（错误 {errors}/{self.bits.size} bit，BER {b:.4f}）"
            )
        else:
            img = payload.bits_to_image(res.info_bits, (32, 32))
            pix = to_qpixmap(img).scaled(92, 92, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.lbl_result_img.setPixmap(pix)
            self.lbl_result_img.setVisible(True)
            self.txt_result.setPlainText("二值水印 32×32")
            self.lbl_result_hint.setText(
                f"提取出的 32×32 水印（错误 {errors}/{self.bits.size} bit，BER {b:.4f}）"
            )

        self._update_buttons()
        self.log(f"提取完成：BER {b:.4f}，NC {c:.4f}，错误 {errors}/{self.bits.size} bit")

    def on_remove(self) -> None:
        """去水印：抹掉系数对幅值差，使水印信息不可读。"""
        if self.wm is None or self.bits is None or self.embed_params is None:
            return

        target = self.attacked if self.attacked is not None else self.wm
        codec = make_codec(self.embed_params["codec"])

        try:
            removed = remove(
                target,
                self.bits.size,
                codec=codec,
                max_repeat=self.embed_params["max_repeat"],
                seed=self.embed_params["seed"],
            )
        except ValueError as exc:
            QMessageBox.warning(self, "去水印失败", str(exc))
            return

        self.removed = removed
        self.panel_processed.title_label.setText("去水印后的图像")
        self.panel_processed.set_image(removed)
        self.panel_processed.setVisible(True)

        reference = self.reference if self.reference is not None else prepare_reference(self.orig)
        p = psnr(reference, removed)

        residual = extract(
            removed,
            self.bits.size,
            codec=codec,
            max_repeat=self.embed_params["max_repeat"],
            seed=self.embed_params["seed"],
        )
        b = ber(self.bits, residual.info_bits)
        c = nc(self.bits, residual.info_bits)
        errors = int(np.sum(self.bits != residual.info_bits))
        self._last_extracted = residual.info_bits
        self._set_metric("ber", f"{b:.4f}")
        self._set_metric("nc", f"{c:.4f}")
        self._set_metric("ber_count", f"{errors} / {self.bits.size}")

        if self.combo_mode.currentIndex() == 0:
            self.txt_result.setPlainText(payload.bits_to_text(residual.info_bits))
            self.lbl_result_img.setVisible(False)
        else:
            img = payload.bits_to_image(residual.info_bits, (32, 32))
            pix = to_qpixmap(img).scaled(92, 92, Qt.KeepAspectRatio, Qt.FastTransformation)
            self.lbl_result_img.setPixmap(pix)
            self.lbl_result_img.setVisible(True)
            self.txt_result.setPlainText("二值水印 32×32（去水印后再提取）")

        self.lbl_result_hint.setText(
            f"去水印后再次提取：BER {b:.4f}（文本已乱码）；去除图 PSNR {p:.2f} dB"
        )
        self._update_buttons()
        self.log(f"去水印完成：去除图 PSNR {p:.2f} dB，再次提取 BER {b:.4f}")


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 9))
    app.setStyleSheet(STYLE)

    win = MainWindow()
    win.show()
    win._ensure_on_screen()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
