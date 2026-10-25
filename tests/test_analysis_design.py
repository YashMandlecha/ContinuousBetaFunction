from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import betafn.betafn as betafn_module
import betafn.setup as setup_api

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

            data = setup_api.SetupBetaFunction.discover_dataset(tmp, flows=["wilson"])

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
