"""CPU tests for the PROFILE kill rules in kit/uma_watch.py (no ssh, no docker).

    python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "kit"))

import uma_watch  # noqa: E402

# spark2 PROFILE, evidence/e3a-kpool-det/boot1-killed/uma.tsv. The profile run's normal
# ~1.8 GiB step fell across two 0.25 s samples, and the old two-sample rule killed the boot.
E3A_BOOT1 = [14.922, 14.909, 14.909, 14.909, 13.902, 13.069]
# spark2 PROFILE, evidence/iter-nvidia-linear-marlin/uma.tsv, a boot that reached ready: two
# drops >= 0.5 GiB below 12 GiB in a row (the closest any recorded boot comes), then it levels off.
MARLIN_BOOT = [11.709, 11.574, 11.729, 11.198, 10.619, 10.394, 10.215, 10.046, 9.898, 10.127, 9.969]
# The 2026-09-16 collapse shape: 19 -> 0.5 GiB in about 7 s at 0.25 s sampling.
COLLAPSE = [19.0 - 18.5 * i / 28 for i in range(29)]


class ProfileKillRules(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        a = argparse.Namespace(evidence=self.tmp.name, worker="spark2", cm_dir=self.tmp.name, phase="PROFILE")
        self.w = uma_watch.Watch(a)
        self.kills: list[str] = []
        self.w.kill = lambda reason, sample: self.kills.append(reason)

    def tearDown(self):
        for f in (self.w.uma, self.w.comp, self.w.wlog):
            f.close()
        self.tmp.cleanup()

    def feed(self, samples):
        """Feed spark2 samples; return the index of the first kill, or None."""
        for i, gib in enumerate(samples):
            self.w.sample("spark2", gib, 0.0)
            if self.kills:
                return i
        return None

    def test_e3a_profile_step_does_not_fire(self):
        self.assertIsNone(self.feed(E3A_BOOT1), self.kills)

    def test_two_fast_drops_below_the_level_do_not_fire(self):
        self.assertIsNone(self.feed(MARLIN_BOOT), self.kills)

    def test_collapse_fires_on_the_slope_before_the_floor(self):
        i = self.feed(COLLAPSE)
        self.assertIsNotNone(i)
        self.assertIn("PROFILE slope", self.kills[0])
        floor = uma_watch.FLOOR_GIB["PROFILE"]
        self.assertGreaterEqual(COLLAPSE[i], floor)
        self.assertLess(i, next(j for j, gib in enumerate(COLLAPSE) if gib < floor))

    def test_floor_still_fires_on_one_sample(self):
        self.assertEqual(self.feed([9.0, 7.9]), 1)
        self.assertIn("< PROFILE floor 8 GiB", self.kills[0])


if __name__ == "__main__":
    unittest.main()
