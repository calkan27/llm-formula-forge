"""
Sandbox runner orchestrating validation, resource caps, and isolated execution.
"""

from __future__ import annotations
import ast
import multiprocessing as mp
import resource
import sys
from typing import Dict, Optional, Tuple
import numpy as np
from implementation.sandbox.mnp_safe import NPSafe
from implementation.sandbox.result import Validator
from implementation.sandbox.ipc import ExecResult


class SandboxRunner:
	"""Coordinator for sandboxed execution."""

	@staticmethod
	def extract_first_fenced(text: str) -> str:
		"""Return the first ``` fenced block body or stripped input if no fence exists."""
		s = text.strip()
		if "```" not in s:
			return s
		parts = s.split("```")
		body = parts[1]
		if body.lower().startswith("python"):
			body = body[len("python"):].lstrip("\n")
		return body.strip()

	@staticmethod
	def apply_cpu_limit(cpu_seconds: Optional[int]) -> None:
		"""Apply a CPU soft limit not exceeding the current hard limit."""
		if not isinstance(cpu_seconds, int) or cpu_seconds <= 0:
			return
		soft_req = int(cpu_seconds)
		soft_cur, hard_cur = resource.getrlimit(resource.RLIMIT_CPU)
		if hard_cur == resource.RLIM_INFINITY:
			soft_new = soft_req
		else:
			if soft_req < hard_cur:
				soft_new = soft_req
			else:
				soft_new = hard_cur
		resource.setrlimit(resource.RLIMIT_CPU, (soft_new, hard_cur))

	@staticmethod
	def apply_as_limit(mem_mb: int) -> None:
		"""Apply an address-space soft limit only when safe and supported."""
		if sys.platform == "darwin":
			return
		mem_bytes = max(128, int(mem_mb)) * 1024 * 1024
		soft_cur, hard_cur = resource.getrlimit(resource.RLIMIT_AS)
		if hard_cur == resource.RLIM_INFINITY and mem_bytes > 0:
			resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard_cur))
			return
		if hard_cur != resource.RLIM_INFINITY and mem_bytes <= hard_cur and soft_cur <= hard_cur:
			resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, hard_cur))

	@staticmethod
	def validate(code: str) -> Tuple[bool, str]:
		"""Validate code against the AST whitelist."""
		v = Validator()
		return v.validate(code)

	@staticmethod
	def build_safe_globals() -> Dict[str, object]:
		"""Construct the global namespace available to executed code."""
		return {"__builtins__": {}, "np": NPSafe()}

	@staticmethod
	def child_execute(code: str, env: Dict[str, object], mem_mb: int, cpu_seconds: Optional[int], q: mp.Queue) -> None:
		"""Validate, compile, and execute code in the child process, always reporting a result."""
		try:
			SandboxRunner.apply_cpu_limit(cpu_seconds)
			SandboxRunner.apply_as_limit(mem_mb)
			try:
				ok, msg = SandboxRunner.validate(code)
			except Exception as e:
				q.put(ExecResult(False, f"reject:validator_error:{e}", None))
				return
			if not ok:
				q.put(ExecResult(False, f"reject:{msg}", None))
				return
			try:
				safe_globals = SandboxRunner.build_safe_globals()
			except Exception as e:
				q.put(ExecResult(False, f"sandbox_init_error:{e}", None))
				return
			try:
				tree = ast.parse(code, mode="exec", type_comments=True)
			except SyntaxError as e:
				q.put(ExecResult(False, f"syntax_error:{e}", None))
				return
			except Exception as e:
				q.put(ExecResult(False, f"parse_error:{e}", None))
				return
			try:
				compiled = compile(tree, filename="<sandbox>", mode="exec")
			except Exception as e:
				q.put(ExecResult(False, f"compile_error:{e}", None))
				return
			local_ns: Dict[str, object] = {}
			try:
				exec(compiled, safe_globals, local_ns)
			except Exception as e:
				q.put(ExecResult(False, f"exec_error:{e}", None))
				return
			f = local_ns.get("f", None)
			if not callable(f):
				q.put(ExecResult(False, "function_f_not_found", None))
				return
			try:
				out = f(env)
			except Exception as e:
				q.put(ExecResult(False, f"runtime_error:{e}", None))
				return
			if out is None:
				q.put(ExecResult(False, "must_return_value", "NoneType"))
				return
			if isinstance(out, (int, float, np.floating)):
				q.put(ExecResult(True, "ok", type(out).__name__))
				return
			if isinstance(out, np.ndarray) and (getattr(out, "ndim", None) in (0, 1)):
				q.put(ExecResult(True, "ok", f"ndarray[{out.dtype},{out.shape}]"))
				return
			q.put(ExecResult(False, "bad_output_shape", type(out).__name__))
		except Exception as e:
			try:
				q.put(ExecResult(False, f"unexpected_error:{e}", None))
			except Exception:
				pass


	@staticmethod
	def safe_execute(
		code_text: str,
		env: Dict[str, object],
		timeout_s: float = 1.0,
		max_mem_mb: int = 256,
		cpu_seconds: int | None = None
	) -> Tuple[bool, str]:
		"""
		Validate and run a single function artifact in a sandboxed child process.
		"""
		code = SandboxRunner.extract_first_fenced(code_text)
		ctx = mp.get_context("spawn")
		q: mp.Queue = ctx.Queue()
		p = ctx.Process(
			target=SandboxRunner.child_execute,
			args=(code, env, max_mem_mb, cpu_seconds, q),
		)
		p.daemon = True
		p.start()
		res = None
		try:
			res = q.get(timeout=max(0.0, float(timeout_s)))
		except Exception:
			res = None
		alive = p.is_alive()
		if alive:
			p.kill()
		p.join(0.2)
		if res is None:
			if alive:
				return (False, "timeout")
			else:
				return (False, "no_result")
		else:
			return (getattr(res, "ok", False), getattr(res, "message", "no_result"))








