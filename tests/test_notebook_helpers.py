from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
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
class TestNotebookFacingHelpers(unittest.TestCase):
    def test_analysis_summary_schema(self):
        bf = betafn_module.BetaFunction(nf=4)
        summary = bf.analysis_summary()

        expected_stages = {"chiral", "infinite_volume", "interpolation", "continuum"}
        self.assertEqual(set(summary.keys()), expected_stages)

        for stage, payload in summary.items():
            self.assertIn("stage", payload)
            self.assertIn("n_fits", payload)
            self.assertIn("mean_chi2_per_dof", payload)
            self.assertIn("mean_pvalue", payload)
            self.assertEqual(payload["stage"], stage)
            self.assertEqual(payload["n_fits"], 0)

    def test_stage_alias_resolution_surface(self):
        bf = betafn_module.BetaFunction(nf=4)
        aliases = set(bf.available_stage_names())
        self.assertTrue({"chiral", "iv", "infinite_volume", "ntrp", "interpolation", "cnt", "continuum"}.issubset(aliases))

        self.assertEqual(bf.stage_store("iv").name, "infinite_volume")
        self.assertEqual(bf.stage_store("ntrp").name, "interpolation")
        self.assertEqual(bf.stage_store("cnt").name, "continuum")

    def test_quality_table_empty_when_no_fits(self):
        bf = betafn_module.BetaFunction(nf=4)
        rows = bf.quality_table("continuum")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
