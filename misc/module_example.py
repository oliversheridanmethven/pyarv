#!/usr/bin/env python
import numpy as np
from module_examples import foo
import unittest


class TestNumpyFloatWrappers(unittest.TestCase):
    def test_numpy_wrapper(self):
        a = np.arange(10, dtype=float)
        b = np.arange(10, dtype=float)
        foo(input=a, output=b, factor=3.0)


if __name__ == "__main__":
    unittest.main()
