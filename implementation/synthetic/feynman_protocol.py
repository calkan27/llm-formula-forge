from __future__ import annotations
import numpy as np
import sympy as sp
from typing import Dict, Tuple, List

from implementation.io.feynman_loader import FeynmanLoader
from implementation.numeric.protected_eval import protected_lambdify
from implementation.complexity import Complexity
from implementation.population.frontier import Pareto2D
from implementation.loss.loss_calibrator import LossCalibrator


class SyntheticFeynmanProtocol:
	"""
	Deterministic synthetic-data generator and success-curve evaluator.
	"""

	def __init__(self) -> None:
		self._C = Complexity()

	def sample_safe(self, n: int, expr: sp.Expr, variables: Tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
		"""Uniform samples in the paper's safe domain (per-variable)."""
		n = int(n)
		variables = tuple(variables)
		dom = FeynmanLoader.infer_safe_domain(expr, variables)
		X = np.zeros((n, len(variables)), dtype=np.float64)
		for j in range(len(variables)):
			v = variables[j]
			L, U = dom[v]
			u = rng.random(n)
			X[:, j] = u * (U - L) + L
		return X

	@staticmethod
	def _affine_01(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
		"""Return (z, lo, hi) with z scaled into [0,1]; widens degenerate span by 1e-9."""
		v = np.asarray(y, dtype=np.float64).ravel()
		if v.size > 0:
			lo = float(np.min(v))
			hi = float(np.max(v))
		else:
			lo = 0.0
			hi = 1.0
		if hi <= lo:
			hi = lo + 1e-9
		return (v - lo) / (hi - lo), lo, hi

	def make_dataset(
		self,
		expr: sp.Expr,
		variables: Tuple[str, ...],
		n: int = 1024,
		sigma: float = 0.0,
		seed: int | None = None,
	) -> Dict[str, object]:
		"""
		Generate a dataset dict:
		  {"X_raw","X01","y_true","y01","x_mins","x_maxs","y_min","y_max","variables","n","sigma","seed"}
		"""
		variables = tuple(variables)
		if seed is None:
			seed = 0
		rng = np.random.default_rng(int(seed))

		X_raw = self.sample_safe(int(n), expr, variables, rng)

		xmins = np.min(X_raw, axis=0).astype(np.float64)
		xmaxs = np.max(X_raw, axis=0).astype(np.float64)
		den = np.where((xmaxs - xmins) > 0.0, (xmaxs - xmins), 1.0)
		X01 = (X_raw - xmins[None, :]) / den[None, :]

		f = protected_lambdify(expr)
		syms_sorted = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
		arg_map: dict[str, np.ndarray] = {}
		for s in syms_sorted:
			idx = variables.index(s.name)
			arg_map[s.name] = X_raw[:, idx]

		with np.errstate(all="ignore"):
			y_true = np.asarray(f(**arg_map), dtype=np.float64).reshape(-1)

		y01, y_lo, y_hi = self._affine_01(y_true)
		if float(sigma) > 0.0:
			y01 = y01 + rng.normal(scale=float(sigma), size=y01.shape[0]).astype(np.float64)
		y01 = np.clip(y01, 0.0, 1.0)

		return {
			"X_raw": X_raw,
			"X01": X01,
			"y_true": y_true,
			"y01": y01,
			"x_mins": xmins,
			"x_maxs": xmaxs,
			"y_min": float(y_lo),
			"y_max": float(y_hi),
			"variables": variables,
			"n": int(n),
			"sigma": float(sigma),
			"seed": int(seed),
		}

	@staticmethod
	def _predict_normalized(expr: sp.Expr, variables: Tuple[str, ...], X_raw: np.ndarray, y_affine: Tuple[float, float]) -> np.ndarray:
		"""Predict with unified protected numerics and map to [0,1] via y's affine span."""
		if len(expr.free_symbols) == 0:
			y = np.full(X_raw.shape[0], float(expr), dtype=np.float64)
		else:
			f = protected_lambdify(expr)
			syms_sorted = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
			arg_map: dict[str, np.ndarray] = {}
			for s in syms_sorted:
				idx = variables.index(s.name)
				arg_map[s.name] = X_raw[:, idx]
			with np.errstate(all="ignore"):
				y = np.asarray(f(**arg_map), dtype=np.float64).reshape(-1)

		lo, hi = float(y_affine[0]), float(y_affine[1])
		if hi <= lo:
			hi = lo + 1e-9
		z = (y - lo) / (hi - lo)
		return np.clip(z, 0.0, 1.0)

	def success_on_frontier(
		self,
		expr: sp.Expr,
		variables: Tuple[str, ...],
		candidates: list[sp.Expr],
		n: int = 1024,
		sigma: float = 0.0,
		seed: int = 0,
		eps: float = 0.0,
	) -> bool:
		"""
		Generate a dataset and test whether `expr` appears on the ε-Pareto frontier
		among `candidates` in (C, E) space (E = MSE on normalized targets).

		Success is defined as:
		  the structural (canonical) form of `expr` is among the candidates chosen
		  by the Jensen ε-sweep frontier in (C, E).
		"""
		ds = self.make_dataset(expr, variables, n=int(n), sigma=float(sigma), seed=int(seed))
		X_raw = ds["X_raw"]
		y01 = ds["y01"]
		y_aff = (float(ds["y_min"]), float(ds["y_max"]))

		E_vals: list[float] = []
		C_vals: list[float] = []
		Cm = Complexity()
		for e in candidates:
			yhat = self._predict_normalized(e, variables, X_raw, y_aff)
			d = yhat - y01
			E_vals.append(float(np.mean(d * d)))
			C_vals.append(Cm.C_min(e))

		idx = Pareto2D.frontier_indices(
			np.asarray(C_vals, dtype=float),
			np.asarray(E_vals, dtype=float),
			eps=float(eps),
		)

		srepr_truth = sp.srepr(Cm.canonical(expr))
		srepr_frontier_set = set()
		for i in idx:
			srepr_frontier_set.add(sp.srepr(Cm.canonical(candidates[i])))
		return srepr_truth in srepr_frontier_set



	def success_curve(
		self,
		expr: sp.Expr,
		variables: Tuple[str, ...],
		candidates: list[sp.Expr],
		T: int = 40,
		sigmas: tuple[float, ...] = (0.0, 1e-3, 1e-2),
		n: int = 1024,
		seed: int = 0,
		eps: float = 0.0,
	) -> dict[float, float]:
		"""
		Return a mapping {sigma: success_rate} where 'success' is the probability
		(over T independent seeds) that the planted truth lies on the ε-frontier.
		"""
		out: dict[float, float] = {}
		T_int = int(T)
		for s in sigmas:
			hits = 0
			for t in range(T_int):
				ok = self.success_on_frontier(
					expr, variables, candidates,
					n=int(n), sigma=float(s),
					seed=int(seed) + t, eps=float(eps)
				)
				if ok:
					hits += 1
			out[float(s)] = hits / max(1, T_int)
		return out

