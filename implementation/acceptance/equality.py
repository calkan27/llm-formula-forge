"""Equality checks for the acceptance harness (class-based).

Provides:
  • EqualityChecks.symbolic_equal(f, g)
  • EqualityChecks.rational_grid_counterexample(f, g, syms) -> dict | None
"""

from __future__ import annotations
from typing import Tuple, List, Dict, Optional
import sympy as sp


class EqualityChecks:
	"""Encapsulates symbolic and exact arithmetic equality checks."""

	def __init__(self) -> None:
		"""Initialize stateless checker (hook for future options)."""

	def symbolic_equal(self, f: sp.Expr, g: sp.Expr) -> bool:
		"""
		Return True iff simplify(f - g) is exactly zero (symbolic certificate).
		Zero-false-negative guarantee per paper/spec when this triggers.
		"""
		d = sp.simplify(f - g)
		if d == 0:
			return True
		else:
			return False

	@staticmethod
	def _is_finite_exact(v) -> bool:
		"""
		Return True iff v is a finite exact SymPy value (not zoo/oo/-oo/nan).
		"""
		vs = sp.sympify(v)
		if vs.is_finite is True:
			return True
		if vs.is_finite is False:
			return False
		if vs.has(sp.zoo) or vs.has(sp.oo) or vs.has(-sp.oo):
			return False
		if vs.has(sp.nan):
			return False
		return True

	def rational_grid_counterexample(self, f: sp.Expr, g: sp.Expr, syms: Tuple[sp.Symbol, ...]) -> Dict[str, object] | None:
		"""
		Search for an exact-arithmetic counterexample on the fixed rational lattice.

		Lattice:
		  R = (-2, -1, -1/2, 0, 1/2, 1, 2), with x_j = r[(2k+(j-1)) mod 7], k=0..255
		Skip non-finite points; return a structured witness or None if inconclusive.
		"""
		vals = [
			sp.Rational(-2, 1), sp.Rational(-1, 1), sp.Rational(-1, 2),
			sp.Rational(0, 1),
			sp.Rational(1, 2), sp.Rational(1, 1), sp.Rational(2, 1),
		]

		if len(syms) == 0:
			vf = sp.simplify(f)
			vg = sp.simplify(g)
			if self._is_finite_exact(vf) and self._is_finite_exact(vg):
				if sp.simplify(vf - vg) != 0:
					diff = sp.simplify(vf - vg)
					return {"point": {}, "f_val": str(vf), "g_val": str(vg), "diff": str(diff)}
			return None

		fun_f = sp.lambdify(syms, f, modules="sympy")
		fun_g = sp.lambdify(syms, g, modules="sympy")

		n = len(vals)
		limit = 256
		for k in range(limit):
			pt: List[sp.Rational] = []
			for j in range(len(syms)):
				pt.append(vals[(2 * k + j) % n])

			vf = fun_f(*pt)
			vg = fun_g(*pt)

			if not self._is_finite_exact(vf) or not self._is_finite_exact(vg):
				continue

			sf = sp.sympify(vf)
			sg = sp.sympify(vg)
			if sp.simplify(sf - sg) != 0:
				point: Dict[str, str] = {}
				for idx in range(len(syms)):
					point[str(syms[idx])] = str(pt[idx])
				diff = sp.simplify(sf - sg)
				return {"point": point, "f_val": str(sf), "g_val": str(sg), "diff": str(diff)}
		return None

