from __future__ import annotations
from typing import List
import numpy as np
import sympy as sp
from scipy.optimize import least_squares
from implementation.fitting.multistart_ls import MultiStartLS
from implementation.numeric.protected_eval import protected_lambdify

class ConstantFitter:
	"""
	Constant fitting for discovered expressions (Theory Sec. 3).

	Implements:
	- Linear least squares for linear-in-parameter models
	- Nonlinear least squares via MultiStartLS (TRF under the hood)
	"""

	def __init__(self, X: np.ndarray, y: np.ndarray):
		"""
		Initialize with data.

		Parameters
		----------
		X : np.ndarray, shape (n, d)
			Input features
		y : np.ndarray, shape (n,)
			Target values
		"""
		self.X = X
		self.y = y
		self.n, self.d = X.shape

	def _extract_constants(self, expr: sp.Expr) -> List[sp.Symbol]:
		"""Extract tunable constants (ERCs) from expression."""
		constants = []
		for sym in expr.free_symbols:
			if str(sym).lower().startswith('c'):
				constants.append(sym)
		return sorted(constants, key=lambda s: str(s))

	def _is_linear_in_params(self, expr: sp.Expr, params: List[sp.Symbol]) -> bool:
		"""Return True iff the template is linear in the parameters."""
		for param in params:
			deriv = sp.diff(expr, param)
			if param in deriv.free_symbols:
				return False
		return True

	@staticmethod
	def fun(theta, Xpayload, f, vars_syms, n: int) -> np.ndarray:
		"""
		Return residual model predictions yhat(theta; Xpayload) using the protected evaluator.
		"""
		vals = []
		for i in range(len(vars_syms)):
			vals.append(Xpayload[:, i])
		for t in theta:
			vals.append(t)
		with np.errstate(all="ignore"):
			pred = f(*vals)
		arr = np.asarray(pred, dtype=np.float64)
		if arr.shape == ():
			arr = np.ones(int(n), dtype=np.float64) * float(arr)
		else:
			if arr.ndim != 1:
				arr = arr.ravel()
			if arr.size != int(n):
				if arr.size < int(n):
					pad = np.zeros(int(n) - arr.size, dtype=np.float64)
					arr = np.concatenate([arr, pad], axis=0)
				else:
					arr = arr[: int(n)]
		arr = np.nan_to_num(arr, nan=0.0, posinf=1e9, neginf=-1e9)
		return arr



	def _fit_nonlinear(self, expr: sp.Expr, params: List[sp.Symbol]) -> sp.Expr:
		"""
		Fit nonlinear-in-parameter model using MultiStartLS with protected numerics
		and a separate residual function (no nested functions).
		"""
		if not params:
			return expr

		vars_pool: list[sp.Symbol] = []
		for t in expr.free_symbols:
			if t not in params:
				vars_pool.append(t)

		vars_pool_sorted: list[sp.Symbol] = sorted(vars_pool, key=lambda q: str(q))

		vars_syms: list[sp.Symbol] = []
		for q in vars_pool_sorted:
			vars_syms.append(q)

		order_names: list[str] = []
		for q in vars_syms:
			order_names.append(q.name)
		for p in params:
			order_names.append(str(p))

		f = protected_lambdify(expr, variables=tuple(order_names))
		if not callable(f):
			print("fit_nonlinear: lambdify failed; returning original expr")
			return expr

		ms = MultiStartLS()
		residual = lambda th: ConstantFitter.fun(th, self.X, f, vars_syms, self.n) - self.y
		out = ms.fit(lambda th, X: ConstantFitter.fun(th, X, f, vars_syms, self.n),
					 self.X, self.y, p=len(params), n_starts=24, radius=1.0, bounds=None, max_nfev=1000)

		theta_hat = np.asarray(out["theta"], dtype=np.float64)
		if theta_hat.shape[0] != len(params):
			print("fit_nonlinear: solution length mismatch; returning original expr")
			return expr
		if not np.all(np.isfinite(theta_hat)):
			print("fit_nonlinear: non-finite solution; returning original expr")
			return expr

		fitted = expr
		for param, value in zip(params, theta_hat):
			fitted = fitted.subs(param, float(value))
		return fitted

	def fit(self, expr: sp.Expr) -> sp.Expr:
		"""
		Fit constants in expression to data.
		"""
		params = self._extract_constants(expr)
		if not params:
			return expr
		if self._is_linear_in_params(expr, params):
			return self._fit_linear(expr, params)
		else:
			return self._fit_nonlinear(expr, params)

	def _fit_linear(self, expr: sp.Expr, params: List[sp.Symbol]) -> sp.Expr:
		"""
		Fit linear-in-parameter model using least squares with protected numerics.
		"""
		if not params:
			return expr

		vars_pool: list[sp.Symbol] = []
		for s in expr.free_symbols:
			if s not in params:
				vars_pool.append(s)

		vars_syms: list[sp.Symbol] = sorted(vars_pool, key=lambda q: str(q))

		order_names: list[str] = []
		for q in vars_syms:
			order_names.append(q.name)

		Phi = np.zeros((self.n, len(params)), dtype=np.float64)

		j = 0
		for param in params:
			coef_expr = expr.coeff(param, 1)
			if coef_expr == 0:
				j += 1
				continue

			f_coef = __import__("implementation").numeric.protected_eval.protected_lambdify(
				coef_expr, variables=tuple(order_names)
			)

			vals: list[np.ndarray] = []
			for i in range(len(vars_syms)):
				vals.append(self.X[:, i])

			if len(vals) > 0:
				with np.errstate(all="ignore"):
					col = np.asarray(f_coef(*vals), dtype=np.float64).reshape(-1)
				if col.size != self.n:
					if col.size < self.n:
						pad = np.zeros(self.n - col.size, dtype=np.float64)
						col = np.concatenate([col, pad], axis=0)
					else:
						col = col[: self.n]
				col = np.nan_to_num(col, nan=0.0, posinf=1e9, neginf=-1e9)
				Phi[:, j] = col
			else:
				val = float(coef_expr)
				Phi[:, j] = np.ones(self.n, dtype=np.float64) * val

			j += 1

		const_expr = expr
		for param in params:
			const_expr = const_expr.subs(param, 0)

		if const_expr != 0:
			f_const = __import__("implementation").numeric.protected_eval.protected_lambdify(
				const_expr, variables=tuple(order_names)
			)
			vals_c: list[np.ndarray] = []
			for i in range(len(vars_syms)):
				vals_c.append(self.X[:, i])
			if len(vals_c) > 0:
				with np.errstate(all="ignore"):
					y_adj = self.y - np.asarray(f_const(*vals_c), dtype=np.float64).reshape(-1)
			else:
				y_adj = self.y - (np.ones(self.n, dtype=np.float64) * float(const_expr))
		else:
			y_adj = self.y

		if Phi.shape[1] != len(params):
			print("fit_linear: design matrix column count mismatch; returning original expr")
			return expr
		if y_adj.shape[0] != self.n:
			print("fit_linear: adjusted target length mismatch; returning original expr")
			return expr
		if not np.all(np.isfinite(Phi)):
			print("fit_linear: non-finite values in Φ; returning original expr")
			return expr
		if not np.all(np.isfinite(y_adj)):
			print("fit_linear: non-finite values in y; returning original expr")
			return expr

		theta_hat = np.linalg.lstsq(Phi, y_adj, rcond=None)[0]
		if theta_hat.shape[0] != len(params):
			print("fit_linear: solution length mismatch; returning original expr")
			return expr

		result = expr
		for param, value in zip(params, theta_hat):
			result = result.subs(param, float(value))
		return result



