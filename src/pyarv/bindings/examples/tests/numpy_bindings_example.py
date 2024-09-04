#!/usr/bin/env python3
"""
Testing the binding module.
"""
import unittest
from pyarv.bindings.examples import multiply_into
# from .np_bench import multiply_into
import numpy as np


class NumpyBindings(unittest.TestCase):
    def test_multiply_into(self):
        print(f"Running the test of my function.")
        a = np.arange(100, dtype=np.double)
        a_orig = 1.0 * a
        b = np.zeros_like(a)
        factor = 2
        reference = a * factor
        multiply_into(input=a, output=b, factor=factor)
        self.assertEqual(a.tolist(), a_orig.tolist(), "The inpout array appears to have been modified.")
        self.assertEqual(b.tolist(), reference.tolist(), "The output is not as expected.")
        b *= 2.0
        self.assertEqual(a.tolist(), a_orig.tolist(), "The inpout array appears to have been modified through the output array.")
        multiply_into(factor=factor, input=a, output=b)


if __name__ == '__main__':
    unittest.main()
