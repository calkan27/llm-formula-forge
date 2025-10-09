"""
Minimal NumPy proxy (whitelist)

This module provides a single class, NPSafe, which exposes a strict, immutable
subset of NumPy needed by the sandbox. The public surface is intentionally tiny:
attribute access is the only way to obtain functionality and only whitelisted
names are available.

Allowed names
-------------
sin, cos, tanh, exp, log, sqrt, maximum, clip, abs, pi, e

Design notes
------------
• No builtins are referenced here.
• No dynamic import tricks are used.
• No try/except blocks are used in production code.
• Attribute access strictly enforces the whitelist.
"""

from __future__ import annotations

from typing import Dict, Tuple, Any

import numpy as np


class NPSafe:
	"""
	Safe NumPy proxy exposing only a fixed whitelist of functions and constants.

	Instances of this class act like a tiny, read-only subset of NumPy: attempting
	to access any attribute not explicitly whitelisted raises AttributeError.
	"""

	__slots__ = ("_allowed",)

	def __init__(self) -> None:
		"""
		Construct the proxy and bind the exact set of allowed names.

		The bound objects are pure functions or float64 constants from NumPy.
		"""
		self._allowed: Dict[str, Any] = {
			"sin": np.sin,
			"cos": np.cos,
			"tanh": np.tanh,
			"exp": np.exp,
			"log": np.log,
			"sqrt": np.sqrt,
			"maximum": np.maximum,
			"clip": np.clip,
			"abs": np.abs,
			"pi": float(np.pi),
			"e": float(np.e),
		}

	def __getattr__(self, name: str) -> Any:
		"""
		Return the whitelisted attribute or raise AttributeError.
		"""
		if name in self._allowed:
			return self._allowed[name]
		raise AttributeError(f"np.{name} is not permitted")

	def allowed_names(self) -> Tuple[str, ...]:
		"""
		Return the tuple of whitelisted attribute names.
		"""
		return tuple(self._allowed.keys())

	def has(self, name: str) -> bool:
		"""
		Check whether a name is whitelisted.
		"""
		return name in self._allowed

	def get(self, name: str) -> Any:
		"""
		Fetch a whitelisted object by name.
		"""
		if self.has(name):
			return self._allowed[name]
		raise AttributeError(f"np.{name} is not permitted")

