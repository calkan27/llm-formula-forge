"""
Units/Dimensions Checker (soundness & progress)

Class: UnitsChecker
-------------------
A total, sound dimension checker for expressions in a restricted algebra:

  • Dimensions live in Z^7 over SI bases (M, L, T, I, Θ, N, J) — see Dim7.
  • +/− : require equal dims; result keeps that same dim.
  • */÷ : add/subtract exponent vectors (group law).
  • pow : if exponent is a rational number, scale exponents; else if exponent
		  is non-numeric, require dimensionless base -> result dimensionless.
  • exp/log/sin/cos/tanh: require dimensionless argument; return dimensionless.
  • Abs   : preserves the argument's dimension.
  • maximum/clip: require operands have equal dimensions; return that dimension.
  • protected_div(x,y): returns dim(x) - dim(y); REJECT if denominator is literal 0.

Also rejects out-of-grammar constructs (Piecewise, Heaviside, Derivative, etc.)

Progress (totality): check_expr either returns a single Dim7 or a structured error.
Soundness (preservation): each clause preserves the judgement by structural induction.

Public API
----------
- UnitsChecker(env_units: dict[str, Dim7] | None)
- check_expr(expr_str: str, target: Dim7 | None = None) -> tuple[bool, Dim7, str]
- infer(expr: sympy.Expr) -> Dim7
- pretty_dim(d: Dim7) -> str
"""

from __future__ import annotations
from fractions import Fraction
from typing import Dict, Tuple

import sympy as sp

from .dim import Dim7
from .protected import ProtectedDiv, Maximum, Clip


class UnitsChecker:
	"""Single entry point for unit/dimension inference & validation."""

	def __init__(self, env_units: Dict[str, Dim7] | None = None) -> None:
		base = {
			"x": Dim7(l=1), "y": Dim7(l=1), "z": Dim7(l=1),
			"t": Dim7(t=1), "v": Dim7(l=1, t=-1), "a": Dim7(l=1, t=-2),
			"m": Dim7(m=1), "F": Dim7(m=1, l=1, t=-2),
			"k": Dim7(), "c0": Dim7(), "theta": Dim7(),
		}
		self._env: Dict[str, Dim7] = dict(base)
		if env_units is not None:
			self._env.update(env_units)

	def _namespace(self) -> Dict[str, object]:
		return {
			"protected_div": ProtectedDiv,
			"maximum": Maximum,
			"clip": Clip,
			"sin": sp.sin, "cos": sp.cos, "tanh": sp.tanh,
			"exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
			"pi": sp.pi, "E": sp.E,
		}

	def sympify_expr(self, expr_str: str) -> sp.Expr:
		"""
		Parse without evaluation so we can keep function calls unevaluated and
		reject any function head that is not explicitly permitted. Also lexically
		block known special functions even if users bind their names as symbols.
		"""
		s = (expr_str or "").strip().replace("^", "**")
		_lower = s.lower()
		for tok in ("gamma", "erf", "floor"):
			if f"{tok}(" in _lower:
				raise ValueError(f"Function not permitted: {tok}")

		expr = sp.sympify(
			s,
			locals=self._namespace(),
			convert_xor=True,
			evaluate=False
		)

		banned_structs = (
			sp.Piecewise, sp.Heaviside, sp.Derivative, sp.Integral,
			sp.Sum, sp.Product, sp.DiracDelta, sp.KroneckerDelta
		)
		for b in banned_structs:
			if expr.has(b):
				raise ValueError("Out-of-grammar construct in expression.")

		allowed_heads = {
			sp.sin, sp.cos, sp.tanh, sp.exp, sp.log, sp.sqrt, sp.Abs,
			ProtectedDiv, Maximum, Clip
		}
		for fnode in expr.atoms(sp.Function):
			if fnode.func not in allowed_heads:
				raise ValueError(f"Function not permitted: {fnode.func.__name__}")

		return expr


	def infer(self, expr: sp.Expr | str) -> Dim7:
		"""
		Return the dimension for a SymPy expression or a source string; raise on
		out-of-grammar or dimension errors. This is a thin dispatcher to rule-specific
		helpers that preserve the original semantics.
		"""
		if isinstance(expr, str):
			expr = self.sympify_expr(expr)
		elif not isinstance(expr, sp.Basic):
			raise TypeError("infer expects a SymPy expression or a string")

		if isinstance(expr, (sp.Number, sp.NumberSymbol)) or bool(getattr(expr, "is_Number", False)):
			return Dim7()

		if expr.is_Symbol:
			return self._env.get(str(expr), Dim7())

		if isinstance(expr, sp.Add):
			return self._infer_add(expr)

		if isinstance(expr, sp.Mul):
			return self._infer_mul(expr)

		if isinstance(expr, sp.Pow):
			return self._infer_pow(expr)

		if isinstance(expr, sp.Function):
			return self._infer_function(expr)

		raise ValueError(f"Unsupported expr node: {type(expr).__name__}")


	def _infer_add(self, e: sp.Add) -> Dim7:
		"""
		Typing rule for addition/subtraction:
		  • Zero terms are neutral and can be ignored.
		  • Dimensionless numeric literals are allowed only if the resulting dimension is dimensionless.
		  • All nonzero dimensionful terms must share the same dimension.
		"""
		base_dim: Dim7 | None = None
		seen_nonzero_dimless_literal = False

		for a in e.args:
			is_exact_zero = self._is_exact_zero(a)
			if is_exact_zero:
				continue

			if getattr(a, "is_Number", False) or isinstance(a, sp.NumberSymbol):
				seen_nonzero_dimless_literal = True
				continue

			da = self.infer(a)
			if base_dim is None:
				base_dim = da
			else:
				if not da.same(base_dim):
					raise ValueError("Add/Sub requires equal dimensions.")

		if base_dim is None:
			return Dim7()

		if seen_nonzero_dimless_literal:
			if not base_dim.is_dimensionless():
				raise ValueError("Cannot add a nonzero dimensionless literal to a dimensionful term.")

		return base_dim

	def _infer_mul(self, e: sp.Mul) -> Dim7:
		"""
		Typing rule for multiplication/division composed from factors:
		  • Dimensions add component-wise across factors (group operation).
		"""
		d = Dim7()
		for a in e.args:
			da = self.infer(a)
			d = d.add(da)
		return d

	def _infer_pow(self, e: sp.Pow) -> Dim7:
		"""
		Typing rule for powers:
		  • If exponent is numeric and exact:
			  - Rational → scale by that Fraction
			  - Integer  → scale by that integer
			  - Otherwise → reject as non-exact numeric exponent
		  • If exponent is symbolic, the base must be dimensionless and the result is dimensionless.
		"""
		base, expo = e.as_base_exp()
		db = self.infer(base)

		if isinstance(expo, (sp.Number, sp.NumberSymbol)) or bool(getattr(expo, "is_Number", False)):
			if expo.is_Rational:
				p = Fraction(int(expo.p), int(expo.q))
				return db.scale_by(p)
			elif expo.is_Integer:
				p = Fraction(int(expo))
				return db.scale_by(p)
			else:
				raise ValueError("Non-exact numeric exponent not supported.")

		if not db.is_dimensionless():
			raise ValueError("Symbolic exponent on non-dimensionless base.")

		return Dim7()

	def _infer_function(self, e: sp.Function) -> Dim7:
		"""
		Typing rules for permitted function heads:
		  • protected_div(x,y) : dim(x) - dim(y); reject literal-zero denominator.
		  • maximum(a,b)       : require equal dimensions; reject if either argument contains divisions.
		  • clip(x, lo, hi)    : require x, lo, hi to share the same dimension.
		  • sin/cos/tanh/exp/log/sqrt : require dimensionless input; return dimensionless.
		  • Abs(x)             : preserves the dimension of x.
		"""
		f = e.func

		if f is ProtectedDiv:
			if len(e.args) != 2:
				raise ValueError("protected_div needs 2 args.")
			den = e.args[1]
			den_is_zero_literal = False
			if isinstance(den, (sp.Number, sp.NumberSymbol)):
				if den == 0:
					den_is_zero_literal = True
			if not den_is_zero_literal:
				if den.equals(0):
					den_is_zero_literal = True
			if not den_is_zero_literal:
				if isinstance(den, sp.Mul):
					has_zero_factor = False
					for arg in den.args:
						if arg.equals(0):
							has_zero_factor = True
							break
					if has_zero_factor:
						den_is_zero_literal = True
			if den_is_zero_literal:
				raise ValueError("protected_div denominator is exactly zero.")
			d0 = self.infer(e.args[0])
			d1 = self.infer(e.args[1])
			return d0.sub(d1)
		if f is Maximum:
			if len(e.args) != 2:
				raise ValueError("maximum needs 2 args.")
			if self._has_division(e.args[0]) or self._has_division(e.args[1]):
				raise ValueError("maximum arguments must not contain divisions.")
			d0 = self.infer(e.args[0])
			d1 = self.infer(e.args[1])
			if not d0.same(d1):
				raise ValueError("maximum requires equal dimensions.")
			return d0
		if f is Clip:
			if len(e.args) != 3:
				raise ValueError("clip needs 3 args.")
			dx = self.infer(e.args[0])
			dlo = self.infer(e.args[1])
			dhi = self.infer(e.args[2])
			if not (dx.same(dlo) and dx.same(dhi)):
				raise ValueError("clip requires x, lo, hi with equal dimensions.")
			return dx
		if f in (sp.sin, sp.cos, sp.tanh, sp.exp, sp.log, sp.sqrt):
			if len(e.args) != 1:
				raise ValueError(f"{f.__name__} expects 1 argument.")
			da = self.infer(e.args[0])
			if not da.is_dimensionless():
				raise ValueError(f"{f.__name__} requires dimensionless input.")
			return Dim7()
		if f is sp.Abs:
			if len(e.args) != 1:
				raise ValueError("Abs expects 1 argument.")
			return self.infer(e.args[0])
		raise ValueError(f"Function not permitted: {f.__name__}")


	def _is_exact_zero(self, a: sp.Expr) -> bool:
		"""
		Return True if the expression is an exact numeric zero or is flagged
		as zero by SymPy (is_zero is True).
		"""
		if getattr(a, "is_Number", False) and a == 0:
			return True
		if getattr(a, "is_zero", False) is True:
			return True
		return False

	def _has_division(self, e: sp.Expr) -> bool:
		"""
		Return True if the expression syntactically contains a division,
		identified via a negative power in Pow or as a factor in Mul.
		"""
		if isinstance(e, sp.Pow):
			base, p = e.as_base_exp()
			if p.is_number and p.is_real and p < 0:
				return True

		if isinstance(e, sp.Mul):
			for part in e.args:
				if isinstance(part, sp.Pow):
					b, p = part.as_base_exp()
					if p.is_number and p.is_real and p < 0:
						return True

		return False

	def check_expr(self, expr_str: str, target: Dim7 | None = None) -> Tuple[bool, Dim7, str]:
		"""
		Returns (ok, dim, msg). If target is provided, ok is True iff inferred == target.
		"""
		e = self.sympify_expr(expr_str)
		d = self.infer(e)
		if target is not None and not d.same(target):
			return False, d, f"dim {self.pretty_dim(d)} != target {self.pretty_dim(target)}"
		return True, d, "ok"

	@staticmethod
	def pretty_dim(d: Dim7) -> str:
		"""Pretty-print a dimension vector."""
		return d.pretty()

	def dim_of_symbol(self, name: str) -> Dim7:
		"""Return the registered dimension of a symbol, or dimensionless if unknown."""
		return self._env.get(name, Dim7())

