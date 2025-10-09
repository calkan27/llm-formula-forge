from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import sympy as sp
import re
from implementation.io.sympy_utils import SympyUtils as S
from implementation.units.system import UnitsChecker
from implementation.units.dim import Dim7
from implementation.interpretability.feature_translate import translate_line


@dataclass(frozen=True)
class ParseResult:
	ok: bool
	expr: Optional[sp.Expr]
	message: str


class ExpressionValidator:
	"""Grammar + typing gate used by the proposal loop."""

	def __init__(self, units_env: dict[str, Dim7] | None = None) -> None:
		self._U = UnitsChecker(units_env or {})

	@staticmethod
	def _has_pdiv_with_zero_denominator(text: str) -> bool:
		"""
		Return True iff text contains a top-level pdiv(..., 0) call.
		"""
		s = (text or "").strip()
		n = len(s)
		for j in range(0, max(0, n - 4)):
			if not s.startswith("pdiv(", j):
				continue
			level = 0
			k_end = None
			for k in range(j, n):
				ch = s[k]
				if ch == "(":
					level += 1
				elif ch == ")":
					level -= 1
					if level == 0:
						k_end = k
						break
			if k_end is None:
				continue
			inside = s[j + len("pdiv(") : k_end]
			args: list[str] = []
			depth = 0
			cur: list[str] = []
			for ch in inside:
				if ch == "," and depth == 0:
					args.append("".join(cur).strip())
					cur = []
				else:
					if ch == "(":
						depth += 1
					elif ch == ")":
						depth -= 1
					cur.append(ch)
			if cur:
				args.append("".join(cur).strip())
			if len(args) == 2 and args[1] == "0":
				return True
		return False

	@staticmethod
	def _has_top_level_div_zero(text: str) -> bool:
		"""
		Return True iff text contains a top-level '/ 0' with a literal zero
		as the immediate right operand token (exactly "0", not "0.0").
		Also returns True for a top-level '/ (u - u)' where u is the same token.
		"""
		s = (text or "").strip()
		n = len(s)
		level = 0
		i = 0
		for _ in range(n):
			if i >= n:
				break
			ch = s[i]
			if ch == "(":
				level += 1
				i += 1
				continue
			elif ch == ")":
				level -= 1
				i += 1
				continue
			elif ch == "/" and level == 0:
				j = i + 1
				while j < n and s[j].isspace():
					j += 1
				if j >= n:
					return False
				if s[j] == "0":
					k = j + 1
					if k >= n:
						return True
					else:
						delims = {")", "(", "+", "-", "*", "/", ",", " "}
						if s[k] in delims:
							return True
				if s[j] == "(":
					k = j
					depth = 0
					while k < n:
						if s[k] == "(":
							depth += 1
						elif s[k] == ")":
							depth -= 1
							if depth == 0:
								break
						k += 1
					if k < n:
						inner = s[j + 1 : k].strip()
						parts = []
						buf = []
						depth2 = 0
						for t in inner:
							if t == "(":
								depth2 += 1
								buf.append(t)
							elif t == ")":
								depth2 -= 1
								buf.append(t)
							elif t == "-" and depth2 == 0:
								parts.append("".join(buf).strip())
								buf = []
							else:
								buf.append(t)
						if buf:
							parts.append("".join(buf).strip())
						if len(parts) == 2:
							u = parts[0]
							v = parts[1]
							if u == v:
								return True
				i += 1
				continue
			else:
				i += 1
				continue
		return False

	def parse(self, text: str) -> ParseResult:
		"""
		Parse a candidate expression into SymPy.
		"""
		if text is None:
			print("parse: input is None")
			return ParseResult(False, None, "parse_error:input_none")
		if not isinstance(text, str):
			print(f"parse: input is not a string (type={type(text).__name__})")
			return ParseResult(False, None, "parse_error:input_not_str")
		s = text.strip()
		if s == "":
			print("parse: input is empty")
			return ParseResult(False, None, "parse_error:empty")
		if not s.isascii():
			print("parse: non-ASCII input detected")
			return ParseResult(False, None, "parse_error:non_ascii")

		p_funcs = {
			"padd", "psub", "pmul", "pdiv",
			"plog", "pexp", "psqrt", "psin", "pcos", "ptanh",
			"pabs", "pneg",
			"pow2", "pow3", "powm1", "powm2", "powm3",
		}
		needs_translation = any((f + "(") in s for f in p_funcs)

		if needs_translation:
			if self._has_pdiv_with_zero_denominator(s):
				print("parse: literal pdiv(..., 0) detected")
				return ParseResult(False, None, "parse_error:division_by_zero_literal")
			func_tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*\s*\(", s)
			bare_heads = {t.split("(")[0] for t in func_tokens}
			allowed_std = {
				"sqrt", "sin", "cos", "tanh", "exp", "log", "Abs", "pi", "E",
				"asin", "acos", "atan", "arcsin", "arccos", "arctan",
			}
			
			unknown_heads = [h for h in bare_heads if (h not in p_funcs and h not in allowed_std)]
			if len(unknown_heads) > 0:
				print(f"parse: unknown function head(s) in p*-line: {sorted(unknown_heads)}")
				return ParseResult(False, None, f"parse_error:unknown_function:{','.join(sorted(unknown_heads))}")

			expr_text = translate_line(s)
		else:
			expr_text = s

		if self._has_top_level_div_zero(expr_text):
			print("parse: literal raw division by zero detected at top level")
			return ParseResult(False, None, "parse_error:division_by_zero_literal_raw")

		lower_src = expr_text.lower()
		banned_fragments = (
			"piecewise(", "heaviside(", "derivative(", "integral(",
			"sum(", "product(", "diracdelta(", "kroneckerdelta(",
		)
		if any(tok in lower_src for tok in banned_fragments):
			print("parse: out-of-grammar construct detected (piecewise/heaviside/calculus/etc.)")
			return ParseResult(False, None, "parse_error:out_of_grammar")
		if ("==" in expr_text) or ("<=" in expr_text) or (">=" in expr_text) or ("<" in expr_text) or (">" in expr_text):
			print("parse: relational operator detected")
			return ParseResult(False, None, "parse_error:relational_not_allowed")

		expr = S.to_sympy(expr_text)
		return ParseResult(True, expr, "ok")


	def parse_and_type(self, text: str, target: Dim7 | None = None) -> ParseResult:
		p = self.parse(text)
		if not p.ok:
			return p
		return self.typecheck(p.expr, target)

	def typecheck(self, expr: sp.Expr, target: Dim7 | None = None) -> ParseResult:
		"""
		Type-check a SymPy expression against the project's UnitsChecker.
		"""
		try:
			ok, _d, _msg = self._U.check_expr(str(expr), target=target)
			if ok:
				return ParseResult(True, expr, "ok")
			return ParseResult(False, None, f"type_mismatch:{_msg}")
		except Exception as e:
			return ParseResult(False, None, f"type_error:{e}")

