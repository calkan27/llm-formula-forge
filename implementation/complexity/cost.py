"""
Node and leaf cost model for the complexity functional C.

Costs:
  • Symbols: 1
  • Small constants {0,1}: 0
  • Medium constants {2,-1,1/2,pi,E}: 1
  • Other numerics: description-length proxy via digit count
  • Add/Mul: (k-1) + sum(children)
  • Pow: small exponent set adds 2 + C(base); other numeric exponent 3 + C(base); symbolic exponent 4 + C(base)+C(exp)
  • Elementary funcs {log,exp,sin,cos,tanh}: 3 + sum(children); Abs: 2 + sum(children); other funcs default 3
"""

from __future__ import annotations
import sympy as sp


class CostModel:
	"""Encapsulate C costs for leaves and internal nodes."""

	def __init__(self) -> None:
		"""
		Initialize constant tiers and operator overheads for the C cost model.
		"""
		self.small_leaf_consts = {sp.Integer(0), sp.Integer(1)}
		self.medium_consts = {sp.Integer(2), sp.Integer(-1), sp.Rational(1, 2), sp.pi, sp.E}
		self.small_rational_pows = {
			sp.Integer(-3), sp.Integer(-2), sp.Integer(-1),
			sp.Rational(-1, 2), sp.Rational(1, 2),
			sp.Integer(2), sp.Integer(3)
		}
		self.func_cost = {sp.log: 3, sp.exp: 3, sp.sin: 3, sp.cos: 3, sp.tanh: 3, sp.Abs: 2}

	def digits_cost(self, n: sp.Expr) -> int:
		"""Return digit-length complexity for a general numeric constant."""
		if isinstance(n, sp.Integer):
			k = str(abs(int(n)))
			count = 0
			for ch in k:
				if ch.isdigit():
					count += 1
			if count < 1:
				return 1
			else:
				return count
		elif isinstance(n, sp.Rational):
			num = str(abs(int(n.p)))
			den = str(abs(int(n.q)))
			count = 0
			for ch in (num + den):
				if ch.isdigit():
					count += 1
			if count < 1:
				return 1
			else:
				return count
		else:
			if n.is_Float:
				s = sp.nsimplify(n)
			else:
				s = n
			t = str(s)
			d = []
			for ch in t:
				if ch.isdigit():
					d.append(ch)

			if len(d) < 1:
				return 1
			else:
				return len(d)


	def const_cost(self, e: sp.Expr) -> int:
		"""Cost for numeric leaves."""
		if e in self.small_leaf_consts:
			return 0
		if e in self.medium_consts:
			return 1
		return self.digits_cost(e)


	def pow_cost(self, C, e: sp.Pow) -> int:
		"""
		Return the cost for a power node, following the tiered exponent rules.
		"""
		base, exp = e.base, e.exp
		if getattr(exp, "is_Float", False):
			exp_coerced = sp.nsimplify(exp)
		else:
			exp_coerced = exp
		if exp_coerced.is_Number:
			if exp_coerced in self.small_rational_pows:
				return 2 + C(base)
			elif isinstance(exp_coerced, (sp.Integer, sp.Rational)):
				return 3 + C(base)
			else:
				return 3 + C(base)
		else:
			return 4 + C(base) + C(exp)


	def variadic_cost(self, C, args: tuple[sp.Expr, ...]) -> int:
		"""Cost for Add/Mul: (k-1)+sum(children)."""
		k = len(args)

		children_sum = 0
		for a in args:
			children_sum += C(a)

		base_cost = k - 1
		if base_cost < 0:
			base_cost = 0

		return base_cost + children_sum


	def func_node_cost(self, C, e: sp.Expr) -> int:
		"""Cost for elementary functions, defaulting to 3."""
		fc = e.func
		op = self.func_cost.get(fc, 3)
		args_sum = 0
		for a in e.args:
			args_sum += C(a)

		return op + args_sum


