#!/usr/bin/env python3
import unittest
from common.variables import variable_names_and_objects
import version


class VersionInfo(unittest.TestCase):

    def test_basic_version_info(self):
        for variable, get_string_function in variable_names_and_objects(version.repo_name, version.repo_author, version.repo_version, version.repo_email):
            output = get_string_function()
            self.assertIsInstance(output, str, msg=f"The {variable = } with {output = } is not the right type.")
            self.assertTrue(output, msg=f"The {variable = } with {output = } is meant to be non-trivial.")


if __name__ == '__main__':
    unittest.main()
