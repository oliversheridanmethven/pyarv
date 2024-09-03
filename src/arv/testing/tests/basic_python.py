import unittest

class Basic(unittest.TestCase):
    def test_pass(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    unittest.main()
