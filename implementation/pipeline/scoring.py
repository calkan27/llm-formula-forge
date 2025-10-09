from __future__ import annotations
from typing import Callable, List, Tuple

from implementation.complexity import Complexity
from implementation.loss.loss_calibrator import LossCalibrator
import sympy as sp


class ScoreComputer:
	"""
	Deterministic scorer:

	  • C from Complexity (canonical)
	  • E injected by caller (dataset/task-specific)
	  • S from a caller-provided monotone map S_of(x)
	  • L via LossCalibrator with a fixed per-batch snapshot (monotone in S)
	"""

	def __init__(self,
				 error_fn: Callable[[sp.Expr], float],
				 S_of: Callable[[float], float]) -> None:
		self._C = Complexity()
		self._E = error_fn
		self._S_of = S_of
		self._lc = LossCalibrator()

	def score_many(self, exprs: List[sp.Expr]) -> List[Tuple[float, int, float, float]]:
		triples: List[Tuple[float, int, float]] = []
		for e in exprs:
			Cval = self._C.C_min(e)
			Eval = float(self._E(e))
			Sval = float(self._S_of(Eval))
			triples.append((Eval, Cval, Sval))
			self._lc.update_observation(Eval, float(Cval), Sval)
		out: List[Tuple[float, int, float, float]] = []
		for (E, Cval, Sval) in triples:
			L = self._lc.loss_with_snapshot(E, float(Cval), Sval)
			out.append((E, Cval, Sval, float(L)))
		return out

