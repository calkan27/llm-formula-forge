"""
AST whitelist validator for the sandbox (no exceptions raised).

The validator returns (ok: bool, message: str) instead of raising.
"""

from __future__ import annotations
import ast
from typing import Tuple


class Validator(ast.NodeVisitor):
	"""AST whitelist validator returning (ok, message)."""

	def __init__(self) -> None:
		"""Initialize state for a single validation pass."""
		self._locals: set[str] = set()
		self._ok = True
		self._msg = "ok"
		self._seen_function = False
		self._seen_docstring = False

	def validate(self, code: str) -> Tuple[bool, str]:
		"""Validate source code against the whitelist."""
		tree = ast.parse(code, mode="exec", type_comments=True)
		self.visit(tree)
		return self._ok, self._msg

	def _fail(self, msg: str) -> None:
		"""Record failure and message."""
		if self._ok:
			self._ok = False
			self._msg = msg

	def visit_Module(self, node: ast.Module) -> None:
		"""Enforce a single top-level function and no imports or extra code."""
		if not self._ok:
			return
		has_import = False
		funcs = []
		others = []
		for n in node.body:
			if isinstance(n, (ast.Import, ast.ImportFrom)):
				has_import = True
			else:
				if isinstance(n, ast.FunctionDef):
					funcs.append(n)
				else:
					others.append(n)
		if has_import:
			self._fail("imports_forbidden")
			return
		if len(funcs) != 1:
			self._fail("module_must_contain_single_function_def")
			return
		else:
			if len(others) != 0:
				self._fail("module_must_contain_single_function_def")
				return
		self._seen_function = True
		self.visit(funcs[0])

	def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
		"""Enforce name f, single arg env, no decorators/annotations; restrict body."""
		if not self._ok:
			return
		if node.name != "f":
			self._fail("function_must_be_named_f"); return
		a = node.args
		if a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg:
			self._fail("signature_must_be_f_env"); return
		if len(a.args) != 1 or a.args[0].arg != "env":
			self._fail("signature_must_be_f_env"); return
		if node.decorator_list or node.returns:
			self._fail("decorators_or_annotations_forbidden"); return
		if any(getattr(x, "annotation", None) is not None for x in a.args):
			self._fail("decorators_or_annotations_forbidden"); return
		if getattr(a, "vararg", None) and getattr(a.vararg, "annotation", None) is not None:
			self._fail("decorators_or_annotations_forbidden"); return
		if getattr(a, "kwarg", None) and getattr(a.kwarg, "annotation", None) is not None:
			self._fail("decorators_or_annotations_forbidden"); return
		self._locals.clear()
		self._seen_docstring = False
		for i, stmt in enumerate(node.body):
			if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
				if i == 0 and not self._seen_docstring:
					self._seen_docstring = True
					continue
				self._fail("only_docstring_expr_allowed"); return
			if isinstance(stmt, ast.Assign):
				self.visit(stmt)
				if not self._ok: return
				continue
			if isinstance(stmt, ast.Return):
				self.visit(stmt)
				if not self._ok: return
				continue
			self.visit(stmt)
			if not self._ok: return

	def visit_Assign(self, node: ast.Assign) -> None:
		"""Allow assignments to simple local names not starting with '_' and not reserved."""
		if not self._ok:
			return
		for t in node.targets:
			if not isinstance(t, ast.Name) or t.id.startswith("_"):
				self._fail("assign_targets_must_be_simple_names"); return
			if t.id in {"np", "env"}:
				self._fail("assign_reserved_name_forbidden"); return
			self._locals.add(t.id)
		self.visit(node.value)

	def visit_Return(self, node: ast.Return) -> None:
		"""Require a return value and validate it."""
		if not self._ok:
			return
		if node.value is None:
			self._fail("must_return_value"); return
		self.visit(node.value)

	def visit_Name(self, node: ast.Name) -> None:
		"""Permit reading 'env', 'np', and locals; permit stores to locals."""
		if not self._ok:
			return
		if isinstance(node.ctx, ast.Store):
			return
		if node.id in {"env", "np"} or node.id in self._locals:
			return
		self._fail(f"name_forbidden:{node.id}")

	def visit_Attribute(self, node: ast.Attribute) -> None:
		"""Allow np.<allowed> only, including constants pi and e."""
		if not self._ok:
			return
		if not (isinstance(node.value, ast.Name) and node.value.id == "np"):
			self._fail("attribute_forbidden"); return
		allowed = {"sin", "cos", "tanh", "exp", "log", "sqrt", "maximum", "clip", "abs", "pi", "e"}
		if node.attr not in allowed:
			self._fail("np_attribute_forbidden"); return
		self.visit(node.value)

	def visit_Subscript(self, node: ast.Subscript) -> None:
		"""Allow env['<string>'] only."""
		if not self._ok:
			return
		if not (isinstance(node.value, ast.Name) and node.value.id == "env"):
			self._fail("subscript_only_env_allowed"); return
		sl = node.slice
		if isinstance(sl, ast.Index):
			sl = sl.value
		if not (isinstance(sl, ast.Constant) and isinstance(sl.value, str)):
			self._fail("env_subscript_must_be_str_key"); return
		self.visit(node.value)

	def visit_Call(self, node: ast.Call) -> None:
		"""Allow calls to np.{sin,cos,tanh,exp,log,sqrt,maximum,clip,abs} only."""
		if not self._ok:
			return
		if not isinstance(node.func, ast.Attribute):
			if isinstance(node.func, ast.Name) and node.func.id == "env":
				self._fail("call_must_be_np_attribute"); return
			self._fail("call_name_forbidden"); return
		if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "np"):
			self._fail("call_must_be_np_attribute"); return
		allowed = {"sin", "cos", "tanh", "exp", "log", "sqrt", "maximum", "clip", "abs"}
		if node.func.attr not in allowed:
			self._fail("np_function_forbidden"); return
		for a in node.args:
			if isinstance(a, ast.Starred):
				self._fail("starred_args_forbidden"); return
			self.visit(a)
			if not self._ok: return
		for kw in node.keywords:
			if kw.arg is None:
				self._fail("kw_unpack_forbidden"); return
			self.visit(kw.value)
			if not self._ok: return

	def visit_BinOp(self, node: ast.BinOp) -> None:
		"""Allow binary operations with +, -, *, / only."""
		if not self._ok:
			return
		if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
			self._fail("binop_forbidden"); return
		self.visit(node.left)
		if not self._ok: return
		self.visit(node.right)

	def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
		"""Allow unary + and - only."""
		if not self._ok:
			return
		if not isinstance(node.op, (ast.UAdd, ast.USub)):
			self._fail("unaryop_forbidden"); return
		self.visit(node.operand)

	def generic_visit(self, node: ast.AST) -> None:
		"""Reject non-whitelisted nodes deterministically."""
		if not self._ok:
			return
		disallowed = (
			ast.Import, ast.ImportFrom, ast.With, ast.Raise, ast.Try, ast.Assert,
			ast.If, ast.For, ast.While, ast.AsyncFunctionDef, ast.AsyncFor, ast.Await,
			ast.Lambda, ast.ClassDef, ast.Global, ast.Nonlocal, ast.Match,
			ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.AugAssign,
			ast.Delete, ast.Yield, ast.YieldFrom
		)
		if isinstance(node, disallowed):
			self._fail(f"node_forbidden:{type(node).__name__}")
			return
		super().generic_visit(node)

