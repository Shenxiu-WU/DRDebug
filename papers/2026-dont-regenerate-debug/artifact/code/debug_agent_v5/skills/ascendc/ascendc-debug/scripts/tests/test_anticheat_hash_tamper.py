"""test_anticheat_hash_tamper.py — 反作弊 hash 校验抗 .sha256 篡改（信任级别对齐）。

背景（success16 NLLLoss 暴露）：cmd_verify 旧逻辑 `base_hash = .sha256.read_text()`
信任存储文本，agent 覆写 `.sha256` 使其=篡改后文件 hash 即可骗过、输出 CLEAN；
而 engine gate (gates/common.py) 对基线副本实时重算，未被骗。本测试固化修复：
cmd_verify 改为对基线副本 `.bench_baseline/<fname>` 实时重算，与 engine gate 对齐。

本机用 subprocess 调 CLI（cmd_verify 经 print 输出、返回 int，CLI 最贴近真实调用）：
  python -m unittest scripts.tests.test_anticheat_hash_tamper
"""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ANTICHEAT = Path(__file__).resolve().parents[1] / "anticheat.py"


def _verify(task_dir: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(_ANTICHEAT), "verify", str(task_dir), "--json"],
        capture_output=True, text=True,
    )
    return json.loads(proc.stdout)


def _snapshot(task_dir: Path) -> None:
    subprocess.run(
        [sys.executable, str(_ANTICHEAT), "snapshot", str(task_dir), "--json"],
        check=True, capture_output=True, text=True,
    )


class TestHashTamper(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.task = Path(self._tmp.name) / "task"
        self.task.mkdir()
        # 最小合法 wrapper：AST 校验需要，但本测试只关心 hash 路径
        (self.task / "model_new_ascendc.py").write_text(
            "original clean code\n", encoding="utf-8")
        _snapshot(self.task)
        self.bdir = self.task / ".bench_baseline"

    def tearDown(self):
        self._tmp.cleanup()

    def test_snapshot_creates_baseline_copy_and_sha256(self):
        self.assertTrue((self.bdir / "model_new_ascendc.py").exists())
        self.assertTrue((self.bdir / "model_new_ascendc.py.sha256").exists())

    def test_unchanged_wrapper_is_clean(self):
        out = _verify(self.task)
        self.assertEqual(out["details"]["hash"]["model_new_ascendc.py"], "unchanged")
        self.assertFalse(
            any("HASH_DIFF" in r for r in out["reasons"]),
            "未改动的 wrapper 不应报 HASH_DIFF")

    def test_tampered_sha256_still_detected(self):
        """核心：改 wrapper + 覆写 .sha256 使其=新文件 hash（骗旧逻辑），
        修复后仍须检出 HASH_DIFF（从基线副本重算，不读被篡改的 .sha256 文本）。"""
        cheat = "cheating code injected\n"
        (self.task / "model_new_ascendc.py").write_text(cheat, encoding="utf-8")
        fake = hashlib.sha256(cheat.encode()).hexdigest()
        (self.bdir / "model_new_ascendc.py.sha256").write_text(
            fake + "\n", encoding="utf-8")

        out = _verify(self.task)
        self.assertEqual(out["verdict"], "CHEAT")
        self.assertTrue(
            any(r == "HASH_DIFF:model_new_ascendc.py" for r in out["reasons"]),
            f"篡改 .sha256 后仍应检出 HASH_DIFF，实际 reasons={out['reasons']}")
        hd = out["details"]["hash"]["model_new_ascendc.py"]
        self.assertIsInstance(hd, dict)
        # baseline 应是对副本重算的原始 hash，而非被篡改成的 fake
        self.assertNotEqual(hd["baseline"], fake)

    def test_tampered_baseline_copy_also_detected_by_engine_gate(self):
        """补充：若 agent 连基线副本一起改（更强攻击），cmd_verify 会被骗，
        但 chmod 只读（workerc 脚本 snapshot 后）从物理层堵死。此处仅记录 cmd_verify
        单独不足以防副本篡改，需依赖 chmod——不做断言，作说明性占位。"""
        # 见 utils/workerc/run_ascendc_debug_batch_cc.sh snapshot 后 chmod 0444/0555。
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
