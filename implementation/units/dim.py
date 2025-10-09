"""
Class Dim7 models SI exponents as an immutable 7-tuple over (M, L, T, I, Θ, N, J).
It supports group-like arithmetic used by the unit checker:

  • d1 * d2     → exponent-wise addition
  • d1 / d2     → exponent-wise subtraction
  • d.pow(p)    → scale by a rational p (must yield integers)
  • d ** p      → same as d.pow(p) for int or Fraction p
  • d1 + d2     → require equal dims; returns that same dim (addition typing rule)
  • d1 - d2     → require equal dims; returns that same dim (subtraction typing rule)
  • same, is_dimensionless, tuple, pretty

The module also exposes convenient base constants:
  DIMLESS, M, L, T, I, TH, N, J
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Tuple, Union


DimTuple = Tuple[int, int, int, int, int, int, int]
PowLike = Union[int, Fraction]


@dataclass(frozen=True)
class Dim7:
	"""Immutable SI dimension vector as Z^7 over bases (M, L, T, I, Θ, N, J)."""
	m: int = 0
	l: int = 0
	t: int = 0
	i: int = 0
	th: int = 0
	n: int = 0
	j: int = 0

	def add(self, other: "Dim7") -> "Dim7":
		return Dim7(
			self.m + other.m, self.l + other.l, self.t + other.t,
			self.i + other.i, self.th + other.th, self.n + other.n, self.j + other.j
		)

	def sub(self, other: "Dim7") -> "Dim7":
		return Dim7(
			self.m - other.m, self.l - other.l, self.t - other.t,
			self.i - other.i, self.th - other.th, self.n - other.n, self.j - other.j
		)

	def scale_by(self, p: Fraction) -> "Dim7":
		if not isinstance(p, Fraction):
			p = Fraction(p)
		vals = (self.m * p, self.l * p, self.t * p, self.i * p, self.th * p, self.n * p, self.j * p)
		ints = []
		for v in vals:
			if v.denominator != 1:
				raise ValueError("Non-integer exponent application on dimension vector.")
			ints.append(int(v.numerator))
		return Dim7(*ints)

	def pow(self, p: PowLike) -> "Dim7":
		if isinstance(p, int):
			return self.scale_by(Fraction(p, 1))
		if isinstance(p, Fraction):
			return self.scale_by(p)
		raise TypeError("pow expects int or Fraction")

	def __mul__(self, other: "Dim7") -> "Dim7":
		return self.add(other)

	def __truediv__(self, other: "Dim7") -> "Dim7":
		return self.sub(other)

	def __pow__(self, p: PowLike) -> "Dim7":
		return self.pow(p)

	def __add__(self, other: "Dim7") -> "Dim7":
		if not self.same(other):
			raise TypeError("Dimension mismatch in '+' (addition requires equal dimensions).")
		return self

	def __sub__(self, other: "Dim7") -> "Dim7":
		if not self.same(other):
			raise TypeError("Dimension mismatch in '-' (subtraction requires equal dimensions).")
		return self

	def same(self, other: "Dim7") -> bool:
		return (self.m == other.m and self.l == other.l and self.t == other.t and
				self.i == other.i and self.th == other.th and self.n == other.n and self.j == other.j)

	def is_dimensionless(self) -> bool:
		return self.same(DIMLESS)

	def to_tuple(self) -> DimTuple:
		return (self.m, self.l, self.t, self.i, self.th, self.n, self.j)

	def tuple(self) -> DimTuple:
		return self.to_tuple()

	def pretty(self) -> str:
		bases = ["M", "L", "T", "I", "Th", "N", "J"]
		exps = [self.m, self.l, self.t, self.i, self.th, self.n, self.j]
		parts = []

		for b, e in zip(bases, exps):
			if e == 0:
				continue
			if e == 1:
				parts.append(b)
			else:
				parts.append(f"{b}^{e}")

		if not parts:
			return "1"
		else:
			return " ".join(parts)



DIMLESS = Dim7()
M = Dim7(m=1)
L = Dim7(l=1)
T = Dim7(t=1)
I = Dim7(i=1)
TH = Dim7(th=1)
N = Dim7(n=1)
J = Dim7(j=1)

