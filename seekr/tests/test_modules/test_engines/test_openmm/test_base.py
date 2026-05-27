from __future__ import annotations
"""
test_base.py

Test the modules/engines/openmm/base.py module.
"""

import seekr.modules.engines.openmm.base as openmm_base

def test_convert_openmm_to_python_expr():
    result = openmm_base.convert_openmm_to_python_expr("x^2 + y^2 + z^2")
    assert result=="x**2 + y**2 + z**2"
    return