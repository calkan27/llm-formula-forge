"""Metrics computation for acceptance: C, E, S, L (class-based)."""

from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np
import sympy as sp

from implementation.loss.loss_calibrator import LossCalibrator
from implementation.feature_selection.dedup import PreScoreDeduper
from implementation.complexity import Complexity
from implementation.numeric.protected_eval import protected_lambdify

class MetricsComputer:
	"""Encapsulates computation of (C, E, S, L) with deterministic sampling."""

	def __init__(self, C: Complexity, S_of: Callable[[float], float] | None) -> None:
		"""
		Bind the canonical complexity functional and optional S mapping.
		"""
		self._C = C
		self._S_of = S_of

	def _safe_eval(self, e: sp.Expr):
		"""Return the unified protected evaluator."""
		return protected_lambdify(e)

	@staticmethod
	def _sorted_syms(e: sp.Expr) -> Tuple[str, ...]:
		"""Return variable names used by e, sorted deterministically."""
		return tuple(sorted((s.name for s in e.free_symbols)))

	def _grid_reference_domain(self, variables: Tuple[str, ...], n: int) -> Dict[str, np.ndarray]:
		"""
		Construct a deterministic uniform grid over the *fixed reference domain* [-1, 1]^d.
		"""
		rng = np.random.default_rng(947)
		X = rng.uniform(-1.0, 1.0, size=(int(n), len(variables))).astype(np.float64)

		out: Dict[str, np.ndarray] = {}
		for i in range(len(variables)):
			name = str(variables[i])
			out[name] = X[:, i]

		return out

	def _normalize01(self, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
		"""Normalize a vector to [0,1]; widen degenerate span by 1e-9."""
		if y.size > 0:
			lo = float(np.min(y))
			hi = float(np.max(y))
		else:
			lo, hi = 0.0, 1.0
		if hi <= lo:
			hi = lo + 1e-9
		return (y - lo) / (hi - lo), lo, hi

	@staticmethod
	def _eval_on_grid(
		fun,
		syms_needed: Tuple[str, ...],
		grid: Dict[str, np.ndarray],
		n: int,
	) -> np.ndarray:
		"""
		Evaluate a lambdified function on a provided grid subset, padding or
		truncating to length n as needed. Supports the constant case.
		"""
		n = int(n)

		if len(syms_needed) == 0:
			val = float(np.asarray(fun(), dtype=np.float64))
			out = np.ones(n, dtype=np.float64) * val
			return out

		args_subset: Dict[str, np.ndarray] = {}
		for name in syms_needed:
			args_subset[name] = grid[name]

		with np.errstate(all="ignore"):
			y = np.asarray(fun(**args_subset), dtype=np.float64).reshape(-1)

		if y.size != n:
			if y.size < n:
				pad = np.zeros(n - y.size, dtype=np.float64)
				y = np.concatenate([y, pad], axis=0)
			else:
				y = y[: n]

		return y


	def compute(
		self,
		truth: sp.Expr,
		cand: sp.Expr,
		variables: Tuple[str, ...],
		n: int,
		force_E: float | None = None,
	) -> Dict[str, float]:
		"""
		Compute canonical metrics on a deterministic float grid.

		Channels
		--------
		C : int
			Canonical complexity C_min(cand).
		E : float
			Normalized regression error on [0,1] targets.
		S : float
			If S_of was provided, S = S_of(E). Else neutral S := 0.5.
		L : float
			Scalarized loss under a fixed snapshot of normalization bounds.
		"""
		n_int = int(n)

		f_f = self._safe_eval(truth)
		f_g = self._safe_eval(cand)

		syms_f = self._sorted_syms(truth)
		syms_g = self._sorted_syms(cand)

		if len(variables) == 0:
			yf = np.ones(n_int, dtype=np.float64) * float(np.asarray(f_f(), dtype=np.float64))
			yg = np.ones(n_int, dtype=np.float64) * float(np.asarray(f_g(), dtype=np.float64))
		else:
			grid = self._grid_reference_domain(variables, n_int)
			yf = self._eval_on_grid(f_f, syms_f, grid, n_int)
			yg = self._eval_on_grid(f_g, syms_g, grid, n_int)

		yf01, lo, hi = self._normalize01(yf)
		yg01 = (yg - lo) / (hi - lo)

		if force_E is None:
			E = LossCalibrator.regression_error(yf01, np.clip(yg01, 0.0, 1.0))
		else:
			E = float(force_E)

		Cv = int(self._C.C_min(cand))

		if self._S_of is not None:
			S = float(self._S_of(E))
		else:
			S = 0.5

		eps = 1e-15
		if S <= eps:
			S = eps
		else:
			if S >= 1.0 - eps:
				S = 1.0 - eps

		lc = LossCalibrator()
		lc.update_many([(E, float(Cv), S)])
		L = float(lc.loss_with_snapshot(E, float(Cv), S))

		return {"C": float(Cv), "E": float(E), "S": float(S), "L": float(L)}


