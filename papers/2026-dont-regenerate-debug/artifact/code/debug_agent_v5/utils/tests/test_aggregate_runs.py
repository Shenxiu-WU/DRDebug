"""test_aggregate_runs.py — N3 跨 run 聚合脚本单元测试。

本机无 pytest, 用 unittest。直跑: python -m unittest utils.tests.test_aggregate_runs
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import aggregate_runs as agg  # noqa: E402


def _make_task(run_root, op_name, *, attempt=0, failure_type="success",
               verify_status="passed", level=None, correctness=True,
               match_rate=100.0, cheats=None, kb_action=None):
    """在 run_root 下造一个任务目录及其产物。返回任务目录路径。"""
    base = run_root if level is None else os.path.join(run_root, level)
    task = os.path.join(base, op_name)
    os.makedirs(os.path.join(task, ".verify_status"))
    os.makedirs(os.path.join(task, ".debug_events"))
    os.makedirs(os.path.join(task, "precision_tuning"))

    with open(os.path.join(task, ".verify_status", "latest.json"), "w",
              encoding="utf-8") as f:
        json.dump({"attempt": attempt, "failure_type": failure_type,
                   "verify": {"status": verify_status}}, f)
    with open(os.path.join(task, ".debug_events", "events.jsonl"), "w",
              encoding="utf-8") as f:
        f.write(json.dumps({"type": "session_started", "op_name": op_name}) + "\n")
    with open(os.path.join(task, "precision_tuning",
                           f"validation_result_attempt_{attempt}.json"), "w",
              encoding="utf-8") as f:
        json.dump({"correctness_passed": correctness,
                   "match_rate": match_rate}, f)
    if cheats is not None:
        with open(os.path.join(task, "precision_tuning", "cheat_history.json"),
                  "w", encoding="utf-8") as f:
            json.dump({"cheating_attempts":
                       [{"severity": s} for s in cheats]}, f)
    if kb_action is not None:
        with open(os.path.join(task, "precision_tuning",
                               "candidate_kb_entry.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"action": kb_action}, f)
    return task


class TestExtract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_clean_success(self):
        run = os.path.join(self.tmp, "runA")
        _make_task(run, "010_LayerNorm")
        recs = agg.scan_run(run)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["run_id"], "runA")
        self.assertEqual(r["op_name"], "010_LayerNorm")
        self.assertTrue(r["objective_success"])
        self.assertTrue(r["reportable_success"])
        self.assertEqual(r["cheat_count"], 0)

    def test_cheat_excluded_from_reportable(self):
        """作弊任务: 客观成功但 reportable_success 必须为 False (N3 口径)。"""
        run = os.path.join(self.tmp, "runC")
        _make_task(run, "023_Hyena", cheats=["violation"])
        r = agg.scan_run(run)[0]
        self.assertTrue(r["objective_success"])
        self.assertFalse(r["reportable_success"])
        self.assertEqual(r["cheat_count"], 1)
        self.assertEqual(r["cheat_severities"], ["violation"])

    def test_warning_also_blocks_reportable(self):
        run = os.path.join(self.tmp, "runW")
        _make_task(run, "x_Op", cheats=["warning"])
        self.assertFalse(agg.scan_run(run)[0]["reportable_success"])

    def test_level_inferred_from_path(self):
        run = os.path.join(self.tmp, "runL")
        _make_task(run, "005_Cumsum", level="level2")
        self.assertEqual(agg.scan_run(run)[0]["level"], "level2")

    def test_level_unknown_when_absent(self):
        run = os.path.join(self.tmp, "runU")
        _make_task(run, "005_Cumsum")
        self.assertEqual(agg.scan_run(run)[0]["level"], "unknown")

    def test_provenance_recorded(self):
        run = os.path.join(self.tmp, "runP")
        task = _make_task(run, "y_Op")
        r = agg.scan_run(run)[0]
        self.assertEqual(r["source_task_dir"], os.path.abspath(task))


class TestCrossRunIsolation(unittest.TestCase):
    """N3 核心: 同名 op 在两 run 状态相反, 不得混算。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))

    def test_same_op_two_runs_not_merged(self):
        run1 = os.path.join(self.tmp, "hyena_run")    # 作弊
        run2 = os.path.join(self.tmp, "deepseek_run")  # 真实成功
        _make_task(run1, "023_HyenaFft", cheats=["violation"])
        _make_task(run2, "023_HyenaFft", attempt=1)
        recs = agg.scan_run(run1) + agg.scan_run(run2)
        self.assertEqual(len(recs), 2)
        by_run = {r["run_id"]: r for r in recs}
        # 同名 op 两条独立记录, run_id 区分
        self.assertFalse(by_run["hyena_run"]["reportable_success"])
        self.assertTrue(by_run["deepseek_run"]["reportable_success"])
        # 渲染小计: 各 run 独立统计, 不混算
        md = agg.render_markdown(recs)
        self.assertIn("hyena_run", md)
        self.assertIn("deepseek_run", md)


class TestBestEffort(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(
            self.tmp, ignore_errors=True))

    def test_corrupt_cheat_history_does_not_crash(self):
        run = os.path.join(self.tmp, "runX")
        task = _make_task(run, "z_Op")
        with open(os.path.join(task, "precision_tuning", "cheat_history.json"),
                  "w", encoding="utf-8") as f:
            f.write("{not valid json")
        r = agg.scan_run(run)[0]  # 不抛
        self.assertEqual(r["cheat_count"], 0)  # 损坏 → 缺省 0

    def test_missing_verify_status_skips_field(self):
        """只有 events.jsonl 也应识别为任务, 缺失字段置 None。"""
        run = os.path.join(self.tmp, "runM")
        task = os.path.join(run, "w_Op")
        os.makedirs(os.path.join(task, ".debug_events"))
        with open(os.path.join(task, ".debug_events", "events.jsonl"), "w",
                  encoding="utf-8") as f:
            f.write(json.dumps({"type": "session_started",
                                "op_name": "w_Op"}) + "\n")
        recs = agg.scan_run(run)
        self.assertEqual(len(recs), 1)
        self.assertIsNone(recs[0]["attempt"])
        self.assertFalse(recs[0]["objective_success"])

    def test_empty_dir_yields_no_records(self):
        run = os.path.join(self.tmp, "runE")
        os.makedirs(os.path.join(run, "junk", "nested"))
        self.assertEqual(agg.scan_run(run), [])


if __name__ == "__main__":
    unittest.main()
