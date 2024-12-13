"""
Unit and regression test for the seekr package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import seekr


def test_seekr_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "seekr" in sys.modules
