import unittest
from epidfit import Data


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual('y' in dir(Data([1, 2, 3], y=[1, 2, 3])), True)


if __name__ == '__main__':
    unittest.main()
