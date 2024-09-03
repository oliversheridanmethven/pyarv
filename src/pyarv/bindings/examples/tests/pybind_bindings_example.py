#!/usr/bin/env python3
"""
Testing the binding module.
"""
import unittest
from pyarv.bindings.examples import add
import numpy as np


class NumpyBindings(unittest.TestCase):
    @unittest.skip("Not implemented yet...")
    def test_multiply(self):
        a = np.array([1.0, 2.0, 3.0])
        factor = 2.0
        expected = np.multiply(a, factor)
        add(1, 2)


if __name__ == '__main__':
    unittest.main()
