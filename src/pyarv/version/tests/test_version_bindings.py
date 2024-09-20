#!/usr/bin/env python3
import unittest
from varname.helpers import jsobj
import pyarv.version as version

import os


class VersionInfo(unittest.TestCase):

    def test_basic_version_info(self):
        for variable, get_string_function in jsobj(version.repo_name, version.repo_author, version.repo_version, version.repo_email).items():
            output = get_string_function()
            self.assertIsInstance(output, str, msg=f"The {variable = } with {output = } is not the right type.")
            self.assertTrue(output, msg=f"The {variable = } with {output = } is meant to be non-trivial.")


if __name__ == '__main__':
    unittest.main()
