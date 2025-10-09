"""
Protected evaluator (single source of truth)
-------------------------------------------
Build a SymPy -> NumPy evaluator with unified guards:

  • log(x)   → log(|x| + ε)
  • sqrt(x)  → sqrt(max(x, 0))
  • Abs(x)   → np.abs(x)
  • exp(x)   → exp(clip(x, -CLIP, CLIP))
  • power(x,p) / Pow(x,p)
	   - protected for p ∈ {-3,-2,-1,-1/2,1/2,2,3}
	   - floors denominators at ε and uses sqrt-then-floor for ±1/2

Notes
-----
- `modules` passed to lambdify must use **string keys only**. Do NOT use SymPy
  function objects as keys (e.g., `sp.log`) or you'll trip the printer with
  "TypeError: keywords must be strings".
- Division is represented via `Mul(..., Pow(den,-1))`, so protecting power
  covers both `/` and `**(-k)`.
"""


from __future__ import annotations
from typing import Dict, Tuple
import numpy as np
import sympy as sp



_GUARD_EPS = 1e-12
_GUARD_CLIP = 80.0

def g_log(x):
	x64 = np.asarray(x, dtype=np.float64)
	return protect_log(x64, _GUARD_EPS)

def g_sqrt(x):
	x64 = np.asarray(x, dtype=np.float64)
	return protect_sqrt(x64)

def g_abs(x):
	x64 = np.asarray(x, dtype=np.float64)
	return protect_abs(x64)

def g_exp(x):
	x64 = np.asarray(x, dtype=np.float64)
	return protect_exp(x64, _GUARD_CLIP)

def g_power(x, p):
	x64 = np.asarray(x, dtype=np.float64)
	return protect_pow(x64, p, _GUARD_EPS)

def g_Pow(x, p):
	x64 = np.asarray(x, dtype=np.float64)
	return protect_pow(x64, p, _GUARD_EPS)

def _eps_val(eps: float) -> float:
	"""Return a positive epsilon constant as float64."""
	v = float(eps)
	if v <= 0.0:
		return 1e-12
	return v


def _clip_val(clip: float) -> float:
	"""Return a positive clipping magnitude for exponential guarding."""
	v = float(clip)
	if v <= 0.0:
		return 80.0
	return v


def protect_log(x: np.ndarray, eps: float) -> np.ndarray:
	"""Protected natural logarithm log(|x|+eps)."""
	return np.log(np.abs(np.asarray(x, dtype=np.float64)) + _eps_val(eps))


def protect_sqrt(x: np.ndarray) -> np.ndarray:
	"""Protected square root sqrt(max(x,0))."""
	return np.sqrt(np.clip(np.asarray(x, dtype=np.float64), 0.0, None))


def protect_abs(x: np.ndarray) -> np.ndarray:
	"""Absolute value as np.abs with float64 semantics."""
	return np.abs(np.asarray(x, dtype=np.float64))


def protect_exp(x: np.ndarray, clip: float) -> np.ndarray:
	"""Protected exponential with symmetric input clipping."""
	return np.exp(np.clip(np.asarray(x, dtype=np.float64), -_clip_val(clip), _clip_val(clip)))


def _floor_den(y: np.ndarray, eps: float) -> np.ndarray:
	"""Floor a denominator magnitude at eps with sign preservation."""
	y64 = np.asarray(y, dtype=np.float64)
	e = _eps_val(eps)
	return np.where(np.abs(y64) > e, y64, np.sign(y64) * e + (y64 == 0.0) * e)


def protect_pow(x: np.ndarray, p, eps: float) -> np.ndarray:
	"""
	Protected power for the small rational/integer set:
	{-3, -2, -1, -1/2, 1/2, 2, 3}. Falls back to np.power otherwise.
	"""
	x64 = np.asarray(x, dtype=np.float64)
	if isinstance(p, (int, float, np.integer, np.floating)):
		val = float(p)
		if val == 2.0:
			return x64 * x64
		if val == 3.0:
			return x64 * x64 * x64
		if val == 0.5:
			return protect_sqrt(x64)
		if val == -0.5:
			root = protect_sqrt(x64)
			root_safe = np.where(root > 0.0, root, _eps_val(eps))
			return 1.0 / root_safe
		if val in (-1.0, -2.0, -3.0):
			base = _floor_den(x64, eps)
			return np.power(base, val)
	return np.power(x64, p)


def _sorted_symbols(expr: sp.Expr, variables: Tuple[str, ...] | None) -> Tuple[sp.Symbol, ...]:
	"""Deterministic symbol order; honor explicit variables if provided."""
	if variables is not None:
		out: list[sp.Symbol] = []
		for v in variables:
			out.append(sp.Symbol(v))
		return tuple(out)
	names = []
	for s in expr.free_symbols:
		names.append(s.name)
	names.sort()
	out: list[sp.Symbol] = []
	for n in names:
		out.append(sp.Symbol(n))
	return tuple(out)


def g_inv(y):
	"""
	Return the protected reciprocal inv(y) using the same epsilon floor as protected powers.
	"""
	y64 = np.asarray(y, dtype=np.float64)
	return protect_pow(y64, -1.0, _GUARD_EPS)

def protected_lambdify(
	expr: sp.Expr,
	variables: Tuple[str, ...] | None = None,
	eps: float = 1e-12,
	exp_clip: float = 80.0,
):
	"""
	Build a protected numeric evaluator for `expr`:
	• Rewrites any rational form into num * inv(den) so all divisions are epsilon-guarded.
	• Routes log/sqrt/Abs/exp and small powers through unified guards.
	"""
	global _GUARD_EPS
	global _GUARD_CLIP
	_GUARD_EPS = _eps_val(eps)
	_GUARD_CLIP = _clip_val(exp_clip)

	mods: Dict[str, object] = {}
	mods["log"] = g_log
	mods["sqrt"] = g_sqrt
	mods["Abs"] = g_abs
	mods["exp"] = g_exp
	mods["power"] = g_power
	mods["Pow"] = g_Pow
	mods["sanitize"] = sanitize_array
	mods["inv"] = g_inv

	inv = sp.Function("inv")
	num, den = sp.fraction(sp.together(expr))
	rexpr = sp.Mul(num, inv(den), evaluate=False)

	syms = _sorted_symbols(expr, variables)
	sanitize = sp.Function("sanitize")
	expr2 = sanitize(rexpr)
	f = sp.lambdify(syms, expr2, modules=[mods, "numpy"])
	return f



def sanitize_array(y):
	"""
	Return a float64 array with NaN replaced by 0 and Inf values clipped to a large finite magnitude.
	"""
	arr = np.asarray(y, dtype=np.float64)
	huge = 1e12
	out = np.nan_to_num(arr, nan=0.0, posinf=huge, neginf=-huge)
	return out

