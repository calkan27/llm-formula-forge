from __future__ import annotations
from dataclasses import dataclass, field
from typing import Deque, Iterable, Tuple
from collections import deque
import numpy as np


@dataclass
class QuantileMinMax:
	"""
	Rolling quantile-based min–max normalizer for a single scalar metric.

	Given a fixed batch B = {z_i}, define:
		lo = Q_low(B), hi = Q_high(B) with 0 <= low < high <= 1
		N(z; B) = clip((z - lo) / max(hi - lo, eps), 0, 1)

	Rolling behavior maintains a window of most recent values; callers may
	request a snapshot (lo, hi) and normalize many values against that *fixed*
	context to preserve per-batch monotonicity guarantees.
	"""
	window: int = 512
	q_low: float = 0.05
	q_high: float = 0.95
	_buf: Deque[float] = field(default_factory=deque, init=False, repr=False)

	def update(self, z: float) -> None:
		"""Add a single scalar observation to the rolling window, evicting the oldest if the window is full."""
		zf = float(z)
		if len(self._buf) >= self.window:
			self._buf.popleft()
		self._buf.append(zf)

	def extend(self, zs: Iterable[float]) -> None:
		"""Add a sequence of scalar observations to the rolling window in order."""
		for z in zs:
			self.update(z)

	def snapshot_bounds(self) -> Tuple[float, float]:
		"""Return (lo, hi) quantiles for the current buffer; robust fallback if insufficient."""
		if len(self._buf) == 0:
			return 0.0, 1.0
		arr = np.asarray(self._buf, dtype=float)
		if len(arr) == 1:
			v = float(arr[0])
			return min(v, v - 1.0), max(v, v + 1.0)
		lo = float(np.quantile(arr, self.q_low))
		hi = float(np.quantile(arr, self.q_high))
		if hi <= lo:  
			hi = lo + 1e-9
		return lo, hi

	@staticmethod
	def normalize(value: float, lo: float, hi: float) -> float:
		"""Normalize a value to [0,1] using provided quantile bounds via clip((value - lo)/max(hi - lo, eps), 0, 1)."""
		denom = max(hi - lo, 1e-12)
		x = (float(value) - lo) / denom
		if x < 0.0:
			return 0.0
		if x > 1.0:
			return 1.0
		return x

