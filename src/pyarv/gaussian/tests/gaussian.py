#!/usr/bin/env python3
import unittest
import numpy as np
from pyarv.gaussian.linear import linear


class TestBasicProperties(unittest.TestCase):
    def test_zero_median(self):
        u = np.array([0.5])
        z = u * np.nan
        linear(input=u, output=z)
        self.assertEqual([0], z.tolist(), "The Gaussian should have a median value of zero.")


if __name__ == '__main__':
    unittest.main()
