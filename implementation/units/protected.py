"""
Protected symbolic primitives for the dimension checker.

Exposes SymPy function heads for:
  protected_div(x, y), maximum(a, b), clip(x, lo, hi)

These names are provided as aliases so tests and sympify locals can import
either the classes or the callables by name without boilerplate.
"""

from __future__ import annotations
import sympy as sp


class ProtectedDiv(sp.Function):
	nargs = 2


class Maximum(sp.Function):
	nargs = 2


class Clip(sp.Function):
	nargs = 3


protected_div = ProtectedDiv
maximum = Maximum
clip = Clip

