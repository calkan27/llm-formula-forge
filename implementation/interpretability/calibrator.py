"""
Implements S(x) = σ(a (x - b)) with fitting under a cap a ≤ 0.02 so that
the global Lipschitz constant L = a/4 ≤ 0.005, ensuring monotonicity and
global Lipschitz continuity.
"""

from __future__ import annotations
from typing import Iterable, Tuple, Dict
import math

from .rubric import Rubric


class CalibratorS:
	"""
	Logistic calibrator for interpretability score S(x) = 1 / (1 + exp(-a(x-b))).

	Fit strategy (deterministic):
	  • Convert anchors (x_i, S_i) to (x_i, logit(S_i)).
	  • Two-point nominal slope a_nom = (ℓ2-ℓ1)/(x2-x1). Enforce 0 < a ≤ A_MAX.
		If a_nom ≤ 0, fallback to a = min(A_MAX, 0.01).
		If a_nom > A_MAX, clip to A_MAX.
	  • For fixed a, least-squares choice of b solves:
			b = mean(x) - mean(logit(S))/a
	  • Clip all x to [-100, 100] for stability; clamp S to (ε, 1-ε).

	Evaluation:
	  • S_of(x) returns S ∈ (0,1) for x ∈ [-100, 100], strictly increasing.
	  • lipschitz_L() returns a/4 ≤ 0.005 per theory.

	No randomness: the result depends only on anchors.
	"""

	A_MAX: float = 0.02

	@staticmethod
	def _logit(p: float) -> float:
		"""logit(p) with clamping to (ε, 1-ε) for numerical robustness."""
		eps = 1e-9
		q = min(max(float(p), eps), 1.0 - eps)
		return math.log(q / (1.0 - q))

	def __init__(self, a: float, b: float) -> None:
		self.a = float(a)
		self.b = float(b)

	@classmethod
	def fit(cls, anchors: Iterable[Tuple[float, float]]) -> "CalibratorS":
		"""
		Fit (a, b) from anchors (x_i, S_i), enforcing 0 < a ≤ A_MAX and
		choosing b by least squares in logit-space.
		"""
		xs: list[float] = []
		ls: list[float] = []
		for x, s in anchors:
			xx = min(max(float(x), -100.0), 100.0)
			xs.append(xx)
			ls.append(cls._logit(s))
		if len(xs) < 2:
			return cls(a=cls.A_MAX, b=0.0)
		i_min = 0
		i_max = 0
		i = 1
		while i < len(xs):
			if xs[i] < xs[i_min]:
				i_min = i
			if xs[i] > xs[i_max]:
				i_max = i
			i += 1
		x1 = xs[i_min]
		x2 = xs[i_max]
		l1 = ls[i_min]
		l2 = ls[i_max]
		dx = x2 - x1
		if abs(dx) > 1e-12:
			a_nom = (l2 - l1) / dx
		else:
			a_nom = 0.0
		if a_nom <= 0.0:
			a = min(cls.A_MAX, 0.01)
		elif a_nom > cls.A_MAX:
			a = cls.A_MAX
		else:
			a = a_nom
		mean_x = sum(xs) / len(xs)
		mean_l = sum(ls) / len(ls)
		b = mean_x - (mean_l / a)
		return cls(a=a, b=b)


	def S_of(self, x: float) -> float:
		"""Return S(x) in (0, 1), with x clipped to [-100, 100]."""
		xx = min(max(float(x), -100.0), 100.0)
		z = self.a * (xx - self.b)
		if z >= 0.0:
			ez = math.exp(-z)
			s = 1.0 / (1.0 + ez)
		else:
			ez = math.exp(z)
			s = ez / (1.0 + ez)
		eps = 1e-15
		return min(max(s, eps), 1.0 - eps)

	def lipschitz_L(self) -> float:
		"""Global Lipschitz constant L = a/4 (bounded by 0.005 via a ≤ 0.02)."""
		return self.a / 4.0

	@staticmethod
	def raw_from_rubric(rubric: Rubric) -> float:
		"""Get raw total x from a rubric (already clipped by Rubric)."""
		return rubric.raw_total()

	def evaluate(self, rubric: Rubric) -> Dict[str, float]:
		"""
		Produce a calibrated evaluation:
			{"x": raw_total, "S": S(x), "a": a, "b": b, "L": a/4}
		"""
		x = self.raw_from_rubric(rubric)
		return {"x": x, "S": self.S_of(x), "a": self.a, "b": self.b, "L": self.lipschitz_L()}

