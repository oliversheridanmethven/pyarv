#!/usr/bin/env python3
import unittest
import numpy as np
from pyarv.gaussian.polynomial import polynomial


class TestBasicProperties(unittest.TestCase):
    def test_zero_median(self):
        u = np.array([0.5], dtype=np.float32)
        z = u * np.nan
        polynomial(input=u, output=z)
        self.assertEqual([0], z.tolist(), "The Gaussian should have a median value of zero.")


if __name__ == '__main__':
    unittest.main()
