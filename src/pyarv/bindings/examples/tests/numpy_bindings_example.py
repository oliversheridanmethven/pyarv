#!/usr/bin/env python3
"""
Testing the binding module.
"""
import unittest
from pyarv.bindings.examples import multiply, multiply_into
import numpy as np


class NumpyBindings(unittest.TestCase):
    @unittest.skip("Not implemented yet...")
    def test_multiply(self):
        a = np.array([1.0, 2.0, 3.0])
        factor = 2.0
        expected = np.multiply(a, factor)
        result = multiply(a, factor)
        self.assertEqual(expected, result)

    def test_multiply_into(self):
        a = np.array([1.0, 2.0, 3.0])
        factor = 2.0
        expected = np.multiply(a, factor)
        result = np.zeros_like(a)
        multiply_into(a)
        # multiply_into(a, factor, result)
        self.assertEqual(expected.tolist(), result.tolist())


if __name__ == '__main__':
    unittest.main()
