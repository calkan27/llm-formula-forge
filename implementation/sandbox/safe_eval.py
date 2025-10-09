"""
Public API surface for sandbox execution.
"""

from __future__ import annotations
from typing import Dict, Tuple
from implementation.sandbox.runner import SandboxRunner


def safe_execute(code_text: str, env: Dict[str, object], timeout_s: float = 1.0, max_mem_mb: int = 256, cpu_seconds: int | None = None) -> Tuple[bool, str]:
	"""Validate and run a single function artifact in a sandboxed child process."""
	return SandboxRunner.safe_execute(code_text, env, timeout_s=timeout_s, max_mem_mb=max_mem_mb, cpu_seconds=cpu_seconds)

