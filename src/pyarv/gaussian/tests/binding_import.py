#!/usr/bin/env python3
"""
Testing the binding module.
"""
import sys

print(sys.path)

import unittest
from pyarv.gaussian import foo, hello_world, fatal_failure, non_fatal_failure
import multiprocessing as mp


class Bindings(unittest.TestCase):
    def test_bound_functions_run(self):
        hello_world()
        foo(1, 'foo')

    def test_fatal_failure(self):
        # Tests which expect an exit from a C extension must be launched in a sub process,
        # else it will take down the Python interpretor session with it.
        # cf. https://stackoverflow.com/a/73070027/5134817
        mp.set_start_method('fork')
        # ^ The Fork is neccessary to bypass pickling issues when objects come from submodules
        # (which aren't by default pickable for whatever reason), so we follow this
        # trick outlined here:
        # https://medium.com/devopss-hole/python-multiprocessing-pickle-issue-e2d35ccf96a9
        ps = mp.Process(target=fatal_failure)
        ps.start()
        ps.join(timeout=1)
        self.assertNotEqual(ps.exitcode, 0, "The exit code from the failing process should not be 0.")

    def test_non_fatal_failure(self):
        try:
            non_fatal_failure()
        except Exception as e:
            print(f"We were successfully able to catch an exception: {e}")


if __name__ == '__main__':
    unittest.main()
