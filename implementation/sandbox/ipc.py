"""
IPC envelopes for the sandbox.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ExecResult:
	"""Structured result for child-process execution."""
	ok: bool
	message: str
	value_type: Optional[str] = None

