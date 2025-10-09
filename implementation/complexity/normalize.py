"""
Canonical normalization for SymPy expressions (finite, terminating transform set).

Transforms:
  • Flatten/sort Add and Mul (commutative canonical order by srepr)
  • Eliminate neutral elements (0 in Add, 1 in Mul), short-circuit zero in Mul
  • Collect numeric factors into a single simplified factor
  • Normalize exact numeric literals (e.g., 2.0→2, 0.5→1/2) and map exp(1)→E, log(E)→1
  • Non-evaluating construction to preserve structure deterministically

The normal form is unique for a given α-equivalence class over commutative Add/Mul.
"""

from __future__ import annotations
from typing import Iterable, Tuple
import sympy as sp


class Normalizer:
	"""Stateless canonicalizer with a finite, terminating transform set."""

	def __init__(self) -> None:
		"""Initialize the normalizer (hook point for future options)."""


	def _sorted_args(self, args: Iterable[sp.Expr]) -> Tuple[sp.Expr, ...]:
		"""Sort a sequence of expressions deterministically by srepr."""
		return tuple(sorted((a for a in args), key=lambda x: sp.srepr(x)))

	def _normalize_number(self, n: sp.Expr) -> sp.Expr:
		"""
		Normalize numeric atoms:
		  • Integer/Rational/NumberSymbol: return as-is
		  • Float: if integer-valued -> Integer; else if a small dyadic rational (den ∈ {2,4,8,16})
				   -> exact Rational; otherwise leave as Float.
		"""
		if not getattr(n, "is_Number", False):
			return n
		if isinstance(n, (sp.Integer, sp.Rational, sp.NumberSymbol)):
			return n
		if isinstance(n, sp.Float):
			f = float(n)
			if f.is_integer():
				return sp.Integer(int(f))
			num, den = f.as_integer_ratio()  
			if den in (2, 4, 8, 16):
				return sp.Rational(num, den)
		return n

	def _flatten_add(self, e: sp.Add) -> Tuple[sp.Expr, ...]:
		"""Flatten nested Add and drop exact zeros."""
		out: list[sp.Expr] = []
		for a in e.args:
			if isinstance(a, sp.Add):
				out.extend(self._flatten_add(a))
			else:
				out.append(a)
		out2 = []
		for a in out:
			if a == 0:
				continue
			out2.append(a)
		return tuple(out2)

	def _flatten_mul(self, e: sp.Mul) -> Tuple[sp.Expr, ...]:
		"""Flatten nested Mul and drop exact ones."""
		out: list[sp.Expr] = []
		for a in e.args:
			if isinstance(a, sp.Mul):
				out.extend(self._flatten_mul(a))
			else:
				out.append(a)
		out2 = []
		for a in out:
			if a == 1:
				continue
			out2.append(a)
		return tuple(out2)

	def _numeric_factor_and_rest(self, args: Iterable[sp.Expr]) -> tuple[sp.Expr, tuple[sp.Expr, ...]]:
		"""Multiply numeric atoms to one factor; return (numeric, others)."""
		num = sp.Integer(1)
		rest: list[sp.Expr] = []
		for a in args:
			if getattr(a, "is_Number", False):
				num *= a
			else:
				rest.append(a)
		num_simple = sp.simplify(num)
		num_simple = self._normalize_number(num_simple)
		return num_simple, tuple(rest)


	def _canon_atom(self, e: sp.Expr) -> sp.Expr:
		"""Canonicalize atomic expressions, normalizing numbers exactly."""
		if getattr(e, "is_Number", False):
			return self._normalize_number(e)
		else:
			return e

	def _canon_add(self, e: sp.Add) -> sp.Expr:
		"""Canonicalize an Add: flatten, recurse, drop zeros, and rebuild sorted with evaluate=False."""
		flat = self._flatten_add(e)
		if not flat:
			return sp.Integer(0)
		childs_list = []
		for a in flat:
			childs_list.append(self._canon_once(a))
		filtered_childs_list = []
		for a in childs_list:
			if a != 0:
				filtered_childs_list.append(a)
		childs = tuple(filtered_childs_list)
		if not childs:
			return sp.Integer(0)
		if len(childs) == 1:
			return childs[0]
		sorted_childs = self._sorted_args(childs)
		return sp.Add(*sorted_childs, evaluate=False)

	def _canon_mul(self, e: sp.Mul) -> sp.Expr:
		"""Canonicalize a Mul: flatten, recurse, collect numeric factor, zero short-circuit, and rebuild sorted."""
		flat = self._flatten_mul(e)
		if not flat:
			return sp.Integer(1)
		childs_list = []
		for a in flat:
			childs_list.append(self._canon_once(a))
		childs = tuple(childs_list)
		num, rest = self._numeric_factor_and_rest(childs)
		if num == 0:
			return sp.Integer(0)
		rest_sorted = self._sorted_args(rest)
		if not rest_sorted:
			return num
		if num == 1:
			return sp.Mul(*rest_sorted, evaluate=False)
		return sp.Mul(num, *rest_sorted, evaluate=False)

	def _canon_pow(self, e: sp.Pow) -> sp.Expr:
		"""Canonicalize a Pow: recurse on base and exponent with exact-number short-circuits and non-evaluating rebuild."""
		b = self._canon_once(e.base)
		p = self._canon_once(e.exp)
		if getattr(p, "is_Number", False):
			if p == 1:
				return b
			if p == 0:
				if not (getattr(b, "is_Number", False) and b == 0):
					return sp.Integer(1)
		if getattr(b, "is_Number", False):
			if b == 1:
				return sp.Integer(1)
			if b == 0:
				if getattr(p, "is_Number", False):
					if p.is_real:
						if p > 0:
							return sp.Integer(0)
		return sp.Pow(b, p, evaluate=False)

	def _canon_func(self, e: sp.Function) -> sp.Expr:
		"""Canonicalize a Function: recurse on args, apply exp/log identities, and rebuild non-evaluating."""
		args_list = []
		for a in e.args:
			args_list.append(self._canon_once(a))
		args = tuple(args_list)
		if e.func is sp.exp:
			if len(args) == 1:
				if args[0] == sp.Integer(1):
					return sp.E
		if e.func is sp.log:
			if len(args) == 1:
				if args[0] == sp.E:
					return sp.Integer(1)
		return e.func(*args, evaluate=False)

	def _canon_once(self, e: sp.Expr) -> sp.Expr:
		"""Apply one pass of canonical transforms by dispatch on node type."""
		if e.is_Atom:
			return self._canon_atom(e)
		if isinstance(e, sp.Add):
			return self._canon_add(e)
		if isinstance(e, sp.Mul):
			return self._canon_mul(e)
		if isinstance(e, sp.Pow):
			return self._canon_pow(e)
		if isinstance(e, sp.Function):
			return self._canon_func(e)
		return e




	def canonical(self, e: sp.Expr) -> sp.Expr:
		"""
		Return the unique canonical normal form of a SymPy expression in the form of `e`.
		"""
		cur = e
		while True:
			nxt = self._canon_once(cur)
			if sp.srepr(nxt) == sp.srepr(cur):
				return nxt
			cur = nxt

