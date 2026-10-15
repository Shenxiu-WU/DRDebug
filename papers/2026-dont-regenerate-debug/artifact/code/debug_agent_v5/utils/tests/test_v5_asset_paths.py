from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = REPO_ROOT / "experiments" / "v5_ablation"
DESIGN = REPO_ROOT / "docs" / "v5_ablation_experiment_master_design_20260724.md"
ASSETS_ROOT = "/workspace/debug_agent_assets"
OLD_PATHS = (
    str(Path("/root") / "cannbot_debug_inputs_n27_20260723"),
    str(Path("/root") / "v5_ablation_control_20260724"),
)


class V5AssetPathTests(unittest.TestCase):
    def test_v5_files_use_sibling_asset_root(self) -> None:
        paths = [
            path for path in V5_ROOT.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        # The public research artifact omits the internal master-design document.
        # Include it when available; keep every packaged control-file check below.
        if DESIGN.is_file():
            paths.append(DESIGN)
        payload = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in paths
        )

        self.assertIn(ASSETS_ROOT, payload)
        for old_path in OLD_PATHS:
            self.assertNotIn(old_path, payload)

    def test_runtime_mounts_freeze_code_control_and_dataset(self) -> None:
        manager = (
            V5_ROOT / "manage_v5_container.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('-v "$ROOT:$ROOT:ro"', manager)
        self.assertIn('-v "$ROOT/outputs:$ROOT/outputs"', manager)
        self.assertIn('-v "$DATASET:$DATASET:ro"', manager)
        self.assertIn('-v "$CONTROL:$CONTROL:ro"', manager)
        self.assertIn(
            '-v "$preflight_output:$preflight_output"', manager)
        self.assertIn(
            '[[ "$preflight_output_rw" == "true" ]]', manager)
        self.assertIn("--cap-drop SYS_ADMIN", manager)
        self.assertIn('docker rm -f "$CONTAINER"', manager)

    def test_preflight_refuses_old_output_before_container_creates_mount(self) -> None:
        prepare = (
            V5_ROOT / "prepare_v5_runtime.sh"
        ).read_text(encoding="utf-8")
        self.assertLess(
            prepare.index('[[ -e "$CONTROL/dataset_preflight" ]]'),
            prepare.index('"$CONTAINER_MANAGER" recreate preflight'),
        )

    def test_launcher_closes_each_arm_before_continuing(self) -> None:
        launcher = (
            V5_ROOT / "launch_v5_ablation.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--verify-arm \"$arm\"", launcher)
        self.assertIn("verify_v5_no_workers.py", launcher)
        self.assertIn("--transient-rechecks 2", launcher)
        self.assertIn('"$CONTAINER_MANAGER" recreate "$arm"', launcher)
        self.assertLess(
            launcher.index("--verify-arm \"$arm\""),
            launcher.index("posthoc completed arm=$arm"),
        )


if __name__ == "__main__":
    unittest.main()
