"""
Deterministic multi-start least squares around SciPy’s trust-region reflective
solver (“trf”). Starts are generated on a cosine grid (with optional affine
mapping to the interior of box bounds) to guarantee repeatability. For
linear-in-parameter templates the solver recovers the noiseless ground-truth
up to floating-point tolerance; for general nonlinear templates it performs
a bounded multistart search.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares


class MultiStartLS:
	"""
	Deterministic multi-start least squares using SciPy's trust-region reflective solver.

	Interface
	---------
	fit(fun, X, y, p, n_starts=24, radius=1.0, bounds=None, max_nfev=2000) -> dict
	  fun(theta, X) -> yhat
	  X: np.ndarray of shape (n, d) or any object consumed by fun
	  y: np.ndarray of shape (n,)
	  p: int number of parameters
	  bounds: None or (lo, hi) arrays of shape (p,)
	"""

	def __init__(self) -> None:
		"""Initialize an empty multi-start least-squares helper (no state is stored)."""
		pass

	@staticmethod
	def _starts(p: int, n: int, radius: float, bounds: tuple[np.ndarray, np.ndarray] | None) -> list[np.ndarray]:
		"""
		Construct a deterministic set of n start points in R^p.
		"""
		if bounds is None:
			mid = None
			amp = None
		else:
			lo, hi = bounds
			mid = 0.5 * (lo + hi)
			amp = 0.5 * (hi - lo)
		S = []
		for k in range(n):
			v = []
			for i in range(p):
				val = np.cos(2.0 * np.pi * float((k + 1) * (i + 1)) / float(n))
				if bounds is None:
					v.append(radius * val)
				else:
					v.append(mid[i] + amp[i] * val)
			S.append(np.asarray(v, dtype=float))
		return S


	def fit(
		self,
		fun,
		X,
		y: np.ndarray,
		p: int,
		n_starts: int = 24,
		radius: float = 1.0,
		bounds: tuple[np.ndarray, np.ndarray] | None = None,
		max_nfev: int = 2000,
	) -> dict:
		"""
		Run bounded multi-start least squares and return the best solution.
		"""
		y = np.asarray(y, dtype=float).ravel()
		n = int(y.size)
		if p < 1:
			return {"theta": np.zeros(0, dtype=float), "resid": float("inf"), "yhat": np.zeros_like(y)}
		if bounds is None:
			lo = -np.inf * np.ones(p, dtype=float)
			hi = np.inf * np.ones(p, dtype=float)
		else:
			lo, hi = bounds
			lo = np.asarray(lo, dtype=float).ravel()
			hi = np.asarray(hi, dtype=float).ravel()
		if bounds is None:
			bh = None
		else:
			bh = (lo, hi)
		starts = self._starts(p, int(n_starts), float(radius), bh)
		best_theta = None
		best_res = float("inf")
		best_yhat = None
		for s in starts:
			resfun = lambda th: np.asarray(fun(th, X), dtype=float).ravel() - y
			res = least_squares(resfun, s, method="trf", bounds=(lo, hi), max_nfev=int(max_nfev))
			r = np.asarray(res.fun, dtype=float).ravel()
			val = float(np.dot(r, r))
			if val < best_res:
				best_res = val
				best_theta = np.asarray(res.x, dtype=float).ravel()
				best_yhat = y + r
		return {"theta": best_theta, "resid": best_res, "yhat": best_yhat}


