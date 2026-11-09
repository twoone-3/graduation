# -*- coding: utf-8 -*-
"""图形界面启动入口。

用法（项目根目录）：
    .venv\\Scripts\\python.exe scripts\\run_gui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dwtm.gui import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
