from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC_BETAFN = ROOT / "src" / "betafn"


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses and self-references resolve.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

try:
    setup_module = _load_module("setup", SRC_BETAFN / "setup.py")
    betafn_module = _load_module("betafn_analysis", SRC_BETAFN / "betafn.py")
    IMPORT_OK = True
except Exception:  # pragma: no cover - environment-dependent
    IMPORT_OK = False


@unittest.skipUnless(IMPORT_OK, "Scientific dependencies are not available in this environment")
class TestAnalysisDesign(unittest.TestCase):
    def test_discover_dataset_filters_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Valid files.
            (tmp / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
            (tmp / "7p00_l20l20l20t40_0p001_wilson.bin").touch()
            (tmp / "7p00_l20l20l20t40_0p001_symanzik.bin").touch()
            (tmp / "7p25_l24l24l24t48_0p0025_wilson.bin").touch()
            # Invalid shape should be ignored.
            (tmp / "not_a_dataset_name.bin").touch()

            data = setup_module.SetupBetaFunction.discover_dataset(tmp, flows=["wilson"])

            self.assertIn("7p00", data)
            self.assertIn("7p25", data)
            self.assertEqual(data["7p00"]["l20l20l20t40"]["0p001"], ["wilson"])
            self.assertNotIn("symanzik", data["7p00"]["l20l20l20t40"]["0p001"])

    def test_stage_summary_aggregation(self):
        bf = betafn_module.BetaFunction(nf=4)
        bf.chiral.quality = {
            "beta1": {
                "vol1": {
                    "g2": {
                        "wilson": {
                            "p": {
                                "3.0": {"chi2": 3.0, "dof": 2, "p-value": 0.2},
                                "3.5": {"chi2": 2.0, "dof": 1, "p-value": 0.4},
                            }
                        }
                    }
                }
            }
        }

        summary = bf.stage_summary("chiral")
        self.assertEqual(summary["stage"], "chiral")
        self.assertEqual(summary["n_fits"], 2)
        self.assertAlmostEqual(summary["mean_chi2_per_dof"], (3.0 / 2.0 + 2.0 / 1.0) / 2.0)
        self.assertAlmostEqual(summary["mean_pvalue"], (0.2 + 0.4) / 2.0)

    def test_quality_table_contains_targets(self):
        bf = betafn_module.BetaFunction(nf=4)
        bf.interpolation.quality = {
            "wilson": {
                "p": {
                    "3.0": {"chi2": 1.0, "dof": 1, "p-value": 0.9}
                }
            }
        }
        rows = bf.quality_table("interpolation")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["stage"], "interpolation")
        self.assertIn("wilson/p/3.0", rows[0]["target"])

    def test_stage_aliases(self):
        bf = betafn_module.BetaFunction(nf=4)
        names = bf.available_stage_names()
        self.assertIn("iv", names)
        self.assertIn("infinite_volume", names)
        self.assertIn("cnt", names)
        self.assertIn("continuum", names)


if __name__ == "__main__":
    unittest.main()
