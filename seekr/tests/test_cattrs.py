from __future__ import annotations
"""
tests/test_cattrs.py

Test cases for cattrs serialization and deserialization of structures.
"""

from attrs import define, field
import cattrs

@define
class A:
    x: int = field(default=1)
    y: int = field(default=2)

@define
class B_base:
    pass

@define
class B_type_1(B_base):
    type: str = field(default="B_type_1")
    a: A | None = None
    z1: int = field(default=3)

@define
class B_type_2(B_base):
    type: str = field(default="B_type_2")
    a: A | None = None
    z2: int = field(default=4)

@define
class C:
    b: B_base | None = field(default=None)
    w: int = field(default=5)

def test_cattrs_by_base_class():
    c = C()
    c.b = B_type_1()
    c.b.a = A(x=10, y=20)
    converter: cattrs.Converter = cattrs.Converter()
    c_dict: dict = converter.unstructure(c)

    c_new = converter.structure(c_dict, C)
    print("c_new:", c_new)
    return
