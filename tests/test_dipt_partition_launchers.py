"""Shell-only checks for DiPT partition experiment arguments and run names."""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts/overfit-sr-interpolants.sh"
ROTATION_ABLATION = ROOT / "experiments/dipt-rotation-range.sh"


class DiPTPartitionLauncherTests(unittest.TestCase):
    def launcher(self, mode=None, predictor="dipt", output_dir=None):
        environment = dict(os.environ, PREDICTOR=predictor)
        environment.pop("DIPT_PARTITION_MODE", None)
        environment.pop("OUTPUT_DIR", None)
        if mode is not None:
            environment["DIPT_PARTITION_MODE"] = mode
        if output_dir is not None:
            environment["OUTPUT_DIR"] = output_dir
        command = 'python() { printf "%s\\n" "$@"; }; source "$1"'
        return subprocess.run(["bash", "-c", command, "test", str(LAUNCHER)],
                              env=environment, capture_output=True, text=True)

    def test_modes_forward_to_gin_and_keep_distinct_default_names(self):
        names = {}
        for mode in (None, "single", "rotated", "shifted", "both"):
            result = self.launcher(mode)
            self.assertEqual(result.returncode, 0, result.stderr)
            effective_mode = mode or "single"
            self.assertIn(f"DiffusionGaussianTransformer.partition_mode='{effective_mode}'", result.stdout)
            names[effective_mode] = next(line for line in result.stdout.splitlines() if line.startswith("--output_dir="))
            if effective_mode != "single":
                self.assertIn(f"_partition_{effective_mode}", names[effective_mode])
        self.assertEqual(len(set(names.values())), 4)
        self.assertNotIn("_partition_", names["single"])

    def test_override_invalid_mode_and_ptv3_isolation(self):
        explicit = self.launcher("rotated", output_dir="/tmp/partition run")
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertIn("--output_dir=/tmp/partition run", explicit.stdout)
        invalid = self.launcher("invalid")
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("DIPT_PARTITION_MODE", invalid.stderr)
        ptv3 = self.launcher("invalid", predictor="ptv3")
        self.assertEqual(ptv3.returncode, 0, ptv3.stderr)
        self.assertNotIn("partition_mode", ptv3.stdout)
        self.assertNotIn("_partition_", ptv3.stdout)

    def test_rotation_ablation_uses_one_mode_for_each_angle(self):
        command = ('bash() { printf "mode=%s predictor=%s output=%s\\n" '
                   '"$DIPT_PARTITION_MODE" "$PREDICTOR" "$OUTPUT_DIR"; }; source "$1"')
        for mode in (None, "shifted"):
            environment = dict(os.environ, ROTATION_RANGES="0 90", OUTPUT_ROOT="/tmp/partition-ablation")
            environment.pop("DIPT_PARTITION_MODE", None)
            if mode is not None:
                environment["DIPT_PARTITION_MODE"] = mode
            result = subprocess.run(["bash", "-c", command, "test", str(ROTATION_ABLATION)],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = [line for line in result.stdout.splitlines() if line.startswith("mode=")]
            self.assertEqual(len(lines), 2)
            suffix = "" if mode is None else f"_partition_{mode}"
            for angle, line in zip((0, 90), lines):
                self.assertIn(f"mode={mode or 'single'} predictor=dipt", line)
                self.assertIn(f"_{angle}deg{suffix}", line)


if __name__ == "__main__":
    unittest.main()
