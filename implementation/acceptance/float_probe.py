from __future__ import annotations
from typing import Dict, Tuple
import math
import numpy as np
import sympy as sp

from implementation.acceptance.metrics import MetricsComputer
from implementation.io.feynman_loader import FeynmanLoader
from implementation.numeric.protected_eval import protected_lambdify

class FloatProbe:
	"""
	Hardened float probe with protected numerics and safe-domain sampling.

	Samples are drawn from the intersection of structure-inferred safe domains and
	evaluation uses the unified protected lambdify.
	"""

	def __init__(self) -> None:
		"""Initialize a stateless float-probe helper."""
		pass

	def _safe_eval(self, e: sp.Expr):
		"""
		Return the central protected evaluator for expression `e`.
		"""
		return protected_lambdify(e)

	@staticmethod
	def _sorted_syms(e: sp.Expr) -> Tuple[str, ...]:
		"""
		Return variable names used by expression `e`, sorted deterministically.
		"""
		names_list: list[str] = []
		for s in e.free_symbols:
			names_list.append(s.name)
		names_sorted = sorted(names_list)
		return tuple(names_sorted)

	def _grid_safe_domain(self, f_expr: sp.Expr, g_expr: sp.Expr, variables: Tuple[str, ...], n: int) -> Dict[str, np.ndarray]:
		"""
		Construct a deterministic uniform grid over the intersection of safe domains
		inferred from `f_expr` and `g_expr` for the given `variables`.
		"""
		vars_t = tuple(variables)
		dom_f = FeynmanLoader.infer_safe_domain(f_expr, vars_t)
		dom_g = FeynmanLoader.infer_safe_domain(g_expr, vars_t)
		rng = np.random.default_rng(1729)
		out: Dict[str, np.ndarray] = {}
		for v in vars_t:
			Lf, Uf = dom_f[v]
			Lg, Ug = dom_g[v]
			L = max(float(Lf), float(Lg))
			U = min(float(Uf), float(Ug))
			if not (U > L):
				U = L + 1e-6
			out[v] = rng.uniform(L, U, size=int(n)).astype(np.float64)
		return out

	def probe(
		self,
		f: sp.Expr,
		g: sp.Expr,
		variables: Tuple[str, ...],
		n: int,
		tol: float,
	) -> Tuple[bool, float, int, int]:
		"""
		Evaluate both forms on a deterministic float grid drawn from a structure-inferred
		safe domain. Returns (ok, max_abs_diff, count_exceeding_tol, total_evaluations).
		"""
		n = int(n)
		if n <= 0:
			return True, 0.0, 0, 0

		fun_f = self._safe_eval(f)
		fun_g = self._safe_eval(g)

		syms_f = self._sorted_syms(f)
		syms_g = self._sorted_syms(g)

		if len(variables) == 0:
			vf = float(np.asarray(fun_f(), dtype=np.float64))
			vg = float(np.asarray(fun_g(), dtype=np.float64))
			diff = abs(vf - vg)
			if diff > float(tol):
				over = 1
			else:
				over = 0
			return True, diff, over, 1

		grid = self._grid_safe_domain(f, g, variables, n)

		yf = MetricsComputer._eval_on_grid(fun_f, syms_f, grid, n)
		yg = MetricsComputer._eval_on_grid(fun_g, syms_g, grid, n)

		d = np.abs(yf - yg)
		if d.size > 0:
			max_abs = float(np.max(d))
		else:
			max_abs = 0.0
		if d.size > 0:
			over = int(np.count_nonzero(d > float(tol)))
		else:
			over = 0
		return True, max_abs, over, int(d.size)


	@staticmethod
	def prob_bound(over: int, m: int, alpha: float = 0.05) -> Tuple[float, float]:
		"""
		One-sided (1−α) upper bound on disagreement fraction δ and miss probability:
		  P_miss ≤ exp(−δ · m)
		"""
		m = max(0, int(m))
		over = max(0, int(over))
		alpha = float(alpha)
		if m == 0:
			return 1.0, 1.0
		if over == 0:
			delta_upper = 1.0 - (alpha ** (1.0 / float(m)))
		else:
			delta_upper = float(over) / float(m)
		p_upper = math.exp(-delta_upper * float(m))
		return float(delta_upper), float(p_upper)

