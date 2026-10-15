from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


UTILS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(UTILS))

from assign_mixed_providers import (  # noqa: E402
    Provider,
    choose_eligible,
    provider_capacity,
    remaining_percent,
    weighted_assign,
    write_assignment,
    write_fixed_assignment,
)


def _usage(name: str, five: int | None, weekly: int | None, parallel: int = 20):
    def window(remaining):
        return {"limit": "100", "remaining": remaining, "used": None}

    return {
        "name": name,
        "supported": True,
        "success": True,
        "five_hour": window(five),
        "weekly_limit": window(weekly),
        "parallel_limit": str(parallel),
    }


class MixedProviderAssignmentTests(unittest.TestCase):
    def setUp(self):
        self.providers = [
            Provider("a", "key-a", "model", "https://api.kimi.com/coding/"),
            Provider("b", "key-b", "model", "https://api.kimi.com/coding/"),
            Provider("c", "key-c", "model", "https://api.kimi.com/coding/"),
        ]

    def test_null_remaining_with_full_used_is_exhausted(self):
        self.assertEqual(
            remaining_percent({"limit": "100", "remaining": None, "used": "100"}),
            0.0,
        )

    def test_capacity_uses_tighter_window(self):
        self.assertEqual(provider_capacity(_usage("a", 80, 25)), 25.0)

    def test_exhausted_and_low_margin_providers_are_excluded(self):
        usage = {
            "a": _usage("a", 0, 90),
            "b": _usage("b", 8, 90),
            "c": _usage("c", 70, 40),
        }
        eligible, policy = choose_eligible(self.providers, usage, 10)
        self.assertEqual(policy, "safety_margin")
        self.assertEqual([item[0].name for item in eligible], ["c"])

    def test_degrades_to_positive_quota_when_all_below_margin(self):
        usage = {
            "a": _usage("a", 5, 90),
            "b": _usage("b", 0, 90),
            "c": _usage("c", 3, 90),
        }
        eligible, policy = choose_eligible(self.providers, usage, 10)
        self.assertEqual(policy, "degraded_positive_quota")
        self.assertEqual([item[0].name for item in eligible], ["a", "c"])

    def test_weighted_assignment_spreads_tasks(self):
        eligible = [
            (self.providers[0], 90.0, 20),
            (self.providers[1], 70.0, 20),
            (self.providers[2], 50.0, 20),
        ]
        result = weighted_assign([f"task{i}" for i in range(5)], eligible)
        names = [provider.name for _, provider in result]
        self.assertEqual(names[:3], ["a", "b", "c"])
        self.assertEqual({name: names.count(name) for name in set(names)}, {"a": 2, "b": 2, "c": 1})

    def test_spreads_before_reusing_high_capacity_provider(self):
        eligible = [
            (self.providers[0], 90.0, 20),
            (self.providers[1], 20.0, 20),
        ]
        names = [
            provider.name
            for _, provider in weighted_assign(["task0", "task1"], eligible)
        ]
        self.assertEqual(names, ["a", "b"])

    def test_manifest_and_queue_never_contain_api_keys(self):
        usage = [_usage("a", 90, 90), _usage("b", 80, 80), _usage("c", 0, 80)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / ".queue"
            payload = write_assignment(
                output=root,
                queue_path=queue,
                tasks=["/tasks/one", "/tasks/two"],
                providers=self.providers,
                usage=usage,
                min_remaining=10,
            )
            queue_text = queue.read_text(encoding="utf-8")
            self.assertNotIn("key-a", queue_text)
            self.assertNotIn("key-b", queue_text)
            self.assertEqual(payload["status"], "assigned")
            self.assertEqual(len(queue_text.splitlines()), 2)
            self.assertEqual((root / ".provider_env" / "a.env").stat().st_mode & 0o777, 0o600)

    def test_fixed_assignment_reuses_exact_mapping_without_old_env_paths(self):
        fixed = {
            "provider_names": ["a", "b", "c"],
            "assignments": [
                {"task_dir": "/tasks/one", "provider": "b",
                 "provider_env": "/stale/secret.env"},
                {"task_dir": "/tasks/two", "provider": "a",
                 "provider_env": "/stale/other.env"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            queue = root / ".queue"
            payload = write_fixed_assignment(
                output=root,
                queue_path=queue,
                tasks=["/tasks/one", "/tasks/two"],
                providers=self.providers,
                fixed_payload=fixed,
                source_path=Path("/prior/provider_assignments.json"),
            )
            rows = queue.read_text(encoding="utf-8").splitlines()
            self.assertEqual(rows[0].split("\t")[:2], ["/tasks/one", "b"])
            self.assertEqual(rows[1].split("\t")[:2], ["/tasks/two", "a"])
            self.assertNotIn("/stale/", "\n".join(rows))
            self.assertEqual(payload["policy"], "fixed_manifest")
            self.assertEqual(
                (root / ".provider_env" / "a.env").stat().st_mode & 0o777,
                0o600,
            )

    def test_fixed_assignment_rejects_task_set_drift(self):
        fixed = {
            "provider_names": ["a", "b", "c"],
            "assignments": [
                {"task_dir": "/tasks/one", "provider": "a"},
                {"task_dir": "/tasks/extra", "provider": "b"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "task set mismatch"):
                write_fixed_assignment(
                    output=root,
                    queue_path=root / ".queue",
                    tasks=["/tasks/one", "/tasks/two"],
                    providers=self.providers,
                    fixed_payload=fixed,
                    source_path=Path("/prior/provider_assignments.json"),
                )

    def test_fixed_assignment_rejects_provider_pool_drift(self):
        fixed = {
            "provider_names": ["a", "b"],
            "assignments": [
                {"task_dir": "/tasks/one", "provider": "a"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "provider pool mismatch"):
                write_fixed_assignment(
                    output=root,
                    queue_path=root / ".queue",
                    tasks=["/tasks/one"],
                    providers=self.providers,
                    fixed_payload=fixed,
                    source_path=Path("/prior/provider_assignments.json"),
                )


if __name__ == "__main__":
    unittest.main()
