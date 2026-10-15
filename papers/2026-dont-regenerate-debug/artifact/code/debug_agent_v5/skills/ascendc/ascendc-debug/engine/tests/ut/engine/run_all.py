"""run_all.py — 无 pytest 依赖的 UT 入口 (本机无 pytest)。

用法 (在 skills/ascendc/ascendc-debug 下):
    python -m engine.tests.ut.engine.run_all
"""
from __future__ import annotations

import os
import sys
import unittest


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.TestLoader().discover(start_dir=here, pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
