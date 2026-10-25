from __future__ import annotations

import unittest

import betafn.betafn as betafn_module


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
