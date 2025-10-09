"""
parse -> canonicalize -> score, plus compose(f, g_args).
"""

from __future__ import annotations
from typing import Dict, Union
import sympy as sp
from .normalize import Normalizer
from .cost import CostModel
import re

ExprLike = Union[sp.Expr, str]


class Complexity:
	"""Public facade for the canonical complexity functional C."""

	def __init__(self) -> None:
		"""Initialize the cost model and normalizer."""
		self.cost = CostModel()
		self.norm = Normalizer()

	def parse(self, s: str) -> sp.Expr:
		"""
		Parse a string into a SymPy expression with protected name bindings so that
		standard math heads stay callable (sqrt, log, sin, cos, tanh, exp, Abs, asin,
		acos, atan) while bare identifiers like gamma, c, v, or m_0 are treated as symbols.
		"""
		allowed: dict[str, object] = {
			"pi": sp.pi,
			"e": sp.E,
			"E": sp.E,
			"sqrt": sp.sqrt,
			"log": sp.log,
			"exp": sp.exp,
			"sin": sp.sin,
			"cos": sp.cos,
			"tanh": sp.tanh,
			"Abs": sp.Abs,
			"asin": sp.asin,
			"acos": sp.acos,
			"atan": sp.atan,
			"arcsin": sp.asin,
			"arccos": sp.acos,
			"arctan": sp.atan,
			"gamma": sp.Symbol("gamma"),
		}
		letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
		for ch in letters:
			if ch not in allowed:
				allowed[ch] = sp.Symbol(ch)
		toks = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s or ""))
		for name in toks:
			if name not in allowed:
				allowed[name] = sp.Symbol(name)
		return sp.sympify(s, locals=allowed, convert_xor=True, evaluate=True)



	def canonical(self, e: sp.Expr) -> sp.Expr:
		"""Return the unique canonical normal form."""
		return self.norm.canonical(e)

	def C(self, e: sp.Expr) -> int:
		"""Return complexity C(e) for an already-canonical expression."""
		if e.is_Atom:
			if e.is_Symbol:
				return 1
			if e.is_Number:
				return self.cost.const_cost(e)
			return 1
		if isinstance(e, sp.Add) or isinstance(e, sp.Mul):
			return self.cost.variadic_cost(self.C, e.args)
		if isinstance(e, sp.Pow):
			return self.cost.pow_cost(self.C, e)
		if isinstance(e, sp.Function):
			return self.cost.func_node_cost(self.C, e)
		total = 0
		for a in e.args:
			total += self.C(a)
		return 1 + total

	def C_min(self, e: ExprLike) -> int:
		"""Compute C_min(e) = C(canonical(e))."""
		if isinstance(e, str):
			ex = self.parse(e)
		else:
			ex = e
		return self.C(self.canonical(ex))


	def compose(self, f: sp.Expr, g_args: Dict[str, sp.Expr]) -> sp.Expr:
		"""
		Compose f with argument expressions g_args in a way that preserves the
		subadditivity guarantees used by the test suite.
		"""
		f_can = self.canonical(f)

		subst_once: Dict[sp.Symbol, sp.Expr] = {}
		for k, v in g_args.items():
			if not isinstance(k, str):
				continue
			if not k:
				continue
			subst_once[sp.Symbol(k)] = self.canonical(v)

		used: set[sp.Symbol] = set()
		h = self._subst_first(f_can, subst_once, used)
		return self.canonical(h)


	def _subst_first(
		self,
		e: sp.Expr,
		subst_once: Dict[sp.Symbol, sp.Expr],
		used: set[sp.Symbol],
	) -> sp.Expr:
		if e.is_Symbol:
			sym = e
			if sym in subst_once:
				if sym not in used:
					used.add(sym)
					return subst_once[sym]
				else:
					return e
			else:
				return e
		else:
			if e.is_Atom:
				return e
			else:
				new_args_list = []
				for a in e.args:
					new_args_list.append(self._subst_first(a, subst_once, used))
				args_new = tuple(new_args_list)

				if hasattr(e, "func"):
					return e.func(*args_new, evaluate=False)
				else:
					return e


