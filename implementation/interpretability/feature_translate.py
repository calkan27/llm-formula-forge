from __future__ import annotations
import re

"""
- _STD_UNARY: maps standard unary math names to formatters.
- _FUNC_MAP_UNARY: maps protected/unified unary primitives to safe string templates.
- _FUNC_MAP_BINARY maps protected binary primitives to templates that respect precedence via helper wrappers.

"""

_STD_UNARY = {
	"sqrt":   lambda a: f"sqrt({a})",
	"sin":    lambda a: f"sin({a})",
	"cos":    lambda a: f"cos({a})",
	"tanh":   lambda a: f"tanh({a})",
	"exp":    lambda a: f"exp({a})",
	"log":    lambda a: f"log({a})",
	"Abs":    lambda a: f"Abs({a})",
	"asin":   lambda a: f"asin({a})",
	"acos":   lambda a: f"acos({a})",
	"atan":   lambda a: f"atan({a})",
	"arcsin": lambda a: f"asin({a})",
	"arccos": lambda a: f"acos({a})",
	"arctan": lambda a: f"atan({a})",
}
_STD_SET_UNARY = set(_STD_UNARY.keys())


_FUNC_MAP_UNARY = {
	"plog":  lambda a: f"log(Abs({a}) + 1e-12)",
	"pexp":  lambda a: f"exp({a})",
	"psqrt": lambda a: f"sqrt({a})",
	"psin":  lambda a: f"sin({a})",
	"pcos":  lambda a: f"cos({a})",
	"ptanh": lambda a: f"tanh({a})",
	"pabs":  lambda a: f"Abs({a})",
	"pneg":  lambda a: f"-({a})",
	"pow2":  lambda a: f"({a})**2",
	"pow3":  lambda a: f"({a})**3",
	"powm1": lambda a: f"({a})**(-1)",
	"powm2": lambda a: f"({a})**(-2)",
	"powm3": lambda a: f"({a})**(-3)",
}
_FUNC_SET_UNARY = set(_FUNC_MAP_UNARY.keys())

_FUNC_MAP_BINARY = {
	"padd": lambda a, b: f"{_wrap_atom_for_addsub(a)}+{_wrap_atom_for_addsub(b)}",
	"psub": lambda a, b: f"{_wrap_atom_for_addsub(a)}-{_wrap_atom_for_addsub(b)}",
	"pmul": lambda a, b: f"{_wrap_atom_for_mul(a)}*{_wrap_atom_for_mul(b)}",
	"pdiv": lambda a, b: f"({a})/({b})",
}
_FUNC_SET_BINARY = set(_FUNC_MAP_BINARY.keys())


def _wrap_atom_for_mul(s: str) -> str:
	"""Wrap only atomic factors for multiplication."""
	t = s.strip()
	if "(" in t:
		return t
	if _has_op(t):
		return t
	if t.startswith("("):
		if t.endswith(")"):
			return t
	return f"({t})"

def _has_op(s: str) -> bool:
	"""Return True iff s contains a top-level operator token (+,-,*,/)."""
	t = str(s)
	level = 0
	i = 0
	for _ in range(len(t)):
		ch = t[i]
		if ch == "(":
			level += 1
		elif ch == ")":
			level -= 1
		else:
			if level == 0:
				if ch in {"+", "-", "*", "/"}:
					return True
		i += 1
	return False



def _wrap_atom_for_addsub(s: str) -> str:
	"""Wrap s in parentheses only when it is an atom (no top-level operator)."""
	t = s.strip()
	if _has_op(t):
		return t
	if t.startswith("("):
		if t.endswith(")"):
			return t
	return f"({t})"

def _split_top_args(s: str) -> list[str]:
	"""Split a raw 'a,b,c' string into top-level args (parenthesis aware)."""
	out, level, cur = [], 0, []
	i = 0
	for _ in range(len(s)):
		ch = s[i]
		if ch == '(':
			level += 1
			cur.append(ch)
		elif ch == ')':
			level -= 1
			cur.append(ch)
		elif ch == ',' and level == 0:
			out.append(''.join(cur).strip())
			cur = []
		else:
			cur.append(ch)
		i += 1
	if cur:
		out.append(''.join(cur).strip())
	return out


_NUM_RE = re.compile(r"""
	^[ \t]*
	[+-]?
	(?:
		(?:\d+\.\d*|\.\d+|\d+)
		(?:[eE][+-]?\d+)?
	)
	[ \t]*$
""", re.X)

_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")


def _is_number(s: str) -> bool:
	return bool(_NUM_RE.match(s))


def _is_name(s: str) -> bool:
	return bool(_NAME_RE.match(s))


def _strip_outer_parens(s: str) -> str:
	"""Return s with a single pair of outer parentheses removed when they enclose the entire string."""
	if not s:
		return s
	if s[0] != "(":
		return s
	if s[-1] != ")":
		return s
	level = 0
	n = len(s)
	ok = True
	for i in range(n):
		ch = s[i]
		if ch == "(":
			level += 1
		elif ch == ")":
			level -= 1
			if level < 0:
				ok = False
				break
	if level != 0:
		ok = False
	if ok:
		body = s[1:-1]
		return body
	return s


def _translate_embedded_p_calls(s: str) -> str:
	"""Translate only recognized p*-heads when they appear inside larger expressions."""
	P_ALLOWED = _FUNC_SET_UNARY.union(_FUNC_SET_BINARY)
	t = s
	n = len(t)
	out = []
	i = 0
	for _ in range(n):
		if i >= n:
			break
		ch = t[i]
		if ch == "p":
			m = re.match(r"p([A-Za-z_0-9]*)\(", t[i:])
			if m:
				head = "p" + m.group(1)
				if head in P_ALLOWED:
					start = i
					j = i + m.end() - 1
					level = 1
					for _k in range(n - j - 1):
						if level <= 0:
							break
						j += 1
						cj = t[j]
						if cj == "(":
							level += 1
						else:
							if cj == ")":
								level -= 1
					call = t[start : j + 1]
					out.append(_translate_expr(call))
					i = j + 1
					continue
		out.append(ch)
		i += 1
	return "".join(out)






def _translate_expr(s: str) -> str:
	"""Recursively translate a single p*-vocab expression to SymPy-friendly text, allowing mixed vocab.

	Rules:
	  • p*-heads are translated to the protected/allowed SymPy heads.
	  • Standard heads {sqrt,sin,cos,tanh,exp,log,Abs} are passed through,
		but their arguments are still recursively translated (so nested p* calls get handled).
	  • Unknown heads raise ValueError("cannot translate function: ...").
	"""
	t = re.sub(r"\s+", "", s)

	if t.startswith("(") and t.endswith(")"):
		u = _strip_outer_parens(t)
		if u != t:
			return _translate_expr(u)

	if "_" in t:
		if re.search(r"\d_\d", t):
			raise ValueError("underscored_numeric_literal")

	if _is_number(t):
		return t
	if _is_name(t):
		return t

	m = re.fullmatch(r"([A-Za-z_][A-Za-z_0-9]*)\((.*)\)", t)
	if m:
		name = m.group(1)
		raw_args = m.group(2)

		if name in _FUNC_SET_UNARY:
			args = _split_top_args(raw_args)
			if len(args) != 1:
				raise ValueError(f"{name} expects 1 arg: {t}")
			a = _translate_expr(args[0])
			return _FUNC_MAP_UNARY[name](a)

		if name in _FUNC_SET_BINARY:
			args = _split_top_args(raw_args)
			if len(args) != 2:
				raise ValueError(f"{name} expects 2 args: {t}")
			a = _translate_expr(args[0])
			b = _translate_expr(args[1])
			return _FUNC_MAP_BINARY[name](a, b)

		if name in _STD_SET_UNARY:
			args = _split_top_args(raw_args)
			if len(args) != 1:
				raise ValueError(f"{name} expects 1 arg: {t}")
			a = _translate_expr(args[0])
			return _STD_UNARY[name](a)

		raise ValueError(f"cannot translate function: {t}")

	return _translate_embedded_p_calls(t)



def _balance_parens(s: str) -> str:
	"""Return s with balanced parentheses by minimally adding missing '(' or ')'."""
	t = str(s)
	level = 0
	unmatched_close = 0
	i = 0
	for _ in range(len(t)):
		ch = t[i]
		if ch == "(":
			level += 1
		elif ch == ")":
			if level > 0:
				level -= 1
			else:
				unmatched_close += 1
		i += 1
	prefix = "(" * unmatched_close
	suffix = ")" * max(0, level)
	return prefix + t + suffix








def translate_line(line: str) -> str:
	"""Public entry: translate a single p*-vocab line to SymPy-friendly text."""
	return _balance_parens(_translate_expr(line))



def translate_many(lines: list[str]) -> list[str]:
	"""Translate many p*-vocab lines to SymPy-friendly strings."""
	out: list[str] = []
	for ln in lines:
		out.append(translate_line(ln))
	return out

