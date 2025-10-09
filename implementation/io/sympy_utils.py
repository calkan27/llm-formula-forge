"""SymPy utilities: strict parsing, canonicalization, and structural queries used by the Feynman loader.

Provides:
  • SympyUtils.to_sympy(expr_str, extra_symbols): strict, deterministic parser with alias normalization and grammar rejection.
  • SympyUtils.canonical_commutative(e): deterministic ordering of commutative args.
  • Structure tests: appears_in_denominator, has_negative_power, under_sqrt, inside_function.

Module-level functions proxy to SympyUtils methods for compatibility.
"""

from __future__ import annotations
from typing import Dict, Iterable, Tuple, Optional
import sympy as sp
from sympy.core.relational import Relational
from sympy import Piecewise, Heaviside
import re

class SympyUtils:
	"""Utility namespace for SymPy parsing and analysis."""

	@staticmethod
	def canonical_commutative(e: sp.Expr) -> sp.Expr:
		"""Return an expression with commutative arguments sorted deterministically."""
		if e.is_Atom:
			return e
		args_list = []
		for a in e.args:
			args_list.append(SympyUtils.canonical_commutative(a))
		args = tuple(args_list)
		if e.is_Add or e.is_Mul:
			args = tuple(sorted(args, key=lambda a: sp.srepr(a)))
		return e.func(*args)

	@staticmethod
	def to_sympy(expr_str: str, extra_symbols: Optional[Iterable[str]] = None) -> sp.Expr:
		"""
		Parse a string into a canonical SymPy expression deterministically, binding all
		identifiers as symbols except for the whitelisted math heads, so names like
		gamma are treated as variables per the protected language. Unknown function
		heads are rejected before sympify to avoid calling Symbols.
		"""
		s = (expr_str or "").strip().replace("^", "**")

		allowed: Dict[str, object] = {
			"sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
			"asin": sp.asin, "arcsin": sp.asin,
			"acos": sp.acos, "arccos": sp.acos,
			"atan": sp.atan, "arctan": sp.atan,
			"log": sp.log, "ln": sp.log, "exp": sp.exp, "sqrt": sp.sqrt,
			"Abs": sp.Abs, "abs": sp.Abs,
			"pi": sp.pi, "E": sp.E, "e": sp.E,
		}

		letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
		for ch in letters:
			if ch not in allowed:
				allowed[ch] = sp.Symbol(ch, real=True)

		if extra_symbols:
			for nm in extra_symbols:
				if nm and isinstance(nm, str):
					allowed[nm] = sp.Symbol(nm, real=True)

		call_heads = set(re.findall(r"([A-Za-z_][A-Za-z_0-9]*)\s*\(", s))
		for name in call_heads:
			if name not in allowed:
				raise ValueError(f"Function not allowed: {name}")

		names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s))
		for nm in names:
			if nm not in allowed:
				allowed[nm] = sp.Symbol(nm, real=True)

		expr = sp.sympify(s, locals=allowed, convert_xor=True, evaluate=True)
		if not isinstance(expr, sp.Basic):
			raise ValueError("Non-expression construct is not allowed")

		has_rel = False
		for r in expr.atoms(Relational):
			if isinstance(r, Relational):
				has_rel = True
				break
		if has_rel:
			raise ValueError("Relational constructs are not allowed")

		if expr.has(Piecewise):
			raise ValueError("Piecewise is not allowed")
		if expr.has(Heaviside):
			raise ValueError("Heaviside is not allowed")

		banned_syntax = (sp.Derivative, sp.Integral, sp.Sum, sp.Product, sp.DiracDelta, sp.KroneckerDelta)
		has_banned = False
		for bt in banned_syntax:
			if expr.has(bt):
				has_banned = True
				break
		if has_banned:
			raise ValueError("Out-of-grammar calculus/distributional construct")

		whitelist = {sp.sin, sp.cos, sp.tan, sp.tanh, sp.asin, sp.acos, sp.atan, sp.log, sp.exp, sp.sqrt, sp.Abs}
		for fnode in expr.atoms(sp.Function):
			if fnode.func not in whitelist:
				raise ValueError(f"Function not allowed: {fnode.func.__name__}")

		expr = sp.together(sp.simplify(sp.expand(expr)))
		expr = SympyUtils.canonical_commutative(expr)
		return expr


	@staticmethod
	def appears_in_denominator(expr: sp.Expr, sym: sp.Symbol) -> bool:
		"""Return True if the symbol appears in the denominator of the expression."""
		_, den = sp.fraction(sp.together(expr))
		return sym in den.free_symbols

	@staticmethod
	def has_negative_power(expr: sp.Expr, sym: sp.Symbol) -> bool:
		"""Return True if the symbol appears as a negative power x**(-k) with k>0."""
		for p in expr.atoms(sp.Pow):
			base, exp = p.as_base_exp()
			if base == sym and exp.is_Number and exp.is_real and exp < 0:
				return True
		return False

	@staticmethod
	def under_sqrt(expr: sp.Expr, sym: sp.Symbol) -> bool:
		"""Return True if the symbol appears under a square root, including Pow with exponent 1/2."""
		for f in expr.atoms(sp.Function):
			if f.func == sp.sqrt and sym in f.args[0].free_symbols:
				return True
		for p in expr.atoms(sp.Pow):
			base, exp = p.as_base_exp()
			if sym in base.free_symbols and exp == sp.Rational(1, 2):
				return True
		return False

	@staticmethod
	def inside_function(expr: sp.Expr, sym: sp.Symbol, func) -> bool:
		"""Return True if the symbol appears inside the given SymPy function."""
		for f in expr.atoms(sp.Function):
			if f.func == func and (sym in f.free_symbols):
				return True
		return False



def canonical_commutative(e: sp.Expr) -> sp.Expr:
	"""Proxy to SympyUtils.canonical_commutative."""
	return SympyUtils.canonical_commutative(e)

def to_sympy(expr_str: str, extra_symbols: Optional[Iterable[str]] = None) -> sp.Expr:
	"""Proxy to SympyUtils.to_sympy."""
	return SympyUtils.to_sympy(expr_str, extra_symbols=extra_symbols)

def appears_in_denominator(expr: sp.Expr, sym: sp.Symbol) -> bool:
	"""Proxy to SympyUtils.appears_in_denominator."""
	return SympyUtils.appears_in_denominator(expr, sym)

def has_negative_power(expr: sp.Expr, sym: sp.Symbol) -> bool:
	"""Proxy to SympyUtils.has_negative_power."""
	return SympyUtils.has_negative_power(expr, sym)

def under_sqrt(expr: sp.Expr, sym: sp.Symbol) -> bool:
	"""Proxy to SympyUtils.under_sqrt."""
	return SympyUtils.under_sqrt(expr, sym)

def inside_function(expr: sp.Expr, sym: sp.Symbol, func) -> bool:
	"""Proxy to SympyUtils.inside_function."""
	return SympyUtils.inside_function(expr, sym, func)

