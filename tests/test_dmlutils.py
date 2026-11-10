"""Tests to make sure we depend on dmlutils properly."""

import os
import sys
import unittest

import dmlutils


class DmlUtilsTestCase(unittest.TestCase):
    """Test access to dependency `dm.utils`."""

    def test_version(self):
        """Test accessing `dmlutils.version`."""
        version = dmlutils.version

        self.assertIsNotNone(version)
        self.assertEqual(str, type(version))


if os.environ.get("DATABRICKS_RUNTIME_VERSION", None):
    # If we are developing on databricks we have to manually
    # instantiate a test suite and load all the tests from this module.
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))
    unittest.TextTestRunner().run(test_suite)
elif __name__ == "__main__":
    unittest.main()
