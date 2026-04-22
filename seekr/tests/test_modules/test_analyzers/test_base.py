import pytest

import numpy as np

import seekr.modules.analyzers.base as analyze_base

def test_pretty_string_value_error():
    """
    Test the function to print 'pretty' scientific notation to the
    terminal.
    """
    mystr = analyze_base.pretty_string_value_error(
        5.6e-2, 2.0e-3, error_digits=1, use_unicode=False)
    expectedstr = "(5.6 +/- 0.2) * 10^-02"
    assert(mystr == expectedstr)
    mystr = analyze_base.pretty_string_value_error(
        5.6e-2, 2.0e-1, error_digits=1, use_unicode=False)
    expectedstr = "(5.6 +/- 20.0) * 10^-02"
    assert(mystr == expectedstr)
    mystr = analyze_base.pretty_string_value_error(
        1.23456789e8, 4.5678e5, error_digits=2, use_unicode=False)
    expectedstr = "(1.2346 +/- 0.0046) * 10^+08"
    assert(mystr == expectedstr)
    return

@pytest.mark.parametrize("a, b, c",
                         [(2.0, 0.0, 2.0),
                          (0.0, 3.5, 3.5),
                          (3.0, 4.0, 5.0),
                          (0.125, 0.0036, 0.1250518293)])
def test_quadriture(a, b, c):
    c_output = analyze_base.quadriture(a, b)
    assert np.isclose(c, c_output)
    return