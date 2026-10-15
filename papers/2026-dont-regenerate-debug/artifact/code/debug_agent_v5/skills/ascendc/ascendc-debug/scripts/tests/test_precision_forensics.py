"""Focused regression tests for deterministic precision forensics."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precision_forensics import DiffAnalyzer  # noqa: E402


class TestMagnitudeCorrelation(unittest.TestCase):
    def test_zero_mismatch_magnitude_does_not_divide_by_zero(self):
        golden = np.concatenate((np.zeros(10), np.ones(10)))
        mismatch_mask = np.concatenate(
            (np.ones(10, dtype=bool), np.zeros(10, dtype=bool))
        )

        result = DiffAnalyzer()._check_magnitude_correlation(
            golden, mismatch_mask
        )

        self.assertEqual(result["pattern"], "magnitude_correlated")
        self.assertEqual(result["confidence"], 0.85)


if __name__ == "__main__":
    unittest.main()
