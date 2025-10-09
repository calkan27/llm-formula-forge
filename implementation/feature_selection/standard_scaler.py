from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StandardScaler:
	"""
	Train-only standard scaler.

	• fit(X): store mean_ and scale_ (std with eps)
	• transform(X): (X - mean_) / scale_
	"""
	mean_: np.ndarray | None = None
	scale_: np.ndarray | None = None
	_eps: float = 1e-12

	def fit(self, X: np.ndarray) -> "StandardScaler":
		X = np.asarray(X, dtype=np.float64)
		mu = np.mean(X, axis=0)
		sd = np.std(X, axis=0)
		sd = np.where(sd > self._eps, sd, self._eps)
		self.mean_ = mu.astype(np.float64, copy=True)
		self.scale_ = sd.astype(np.float64, copy=True)
		return self

	def transform(self, X: np.ndarray) -> np.ndarray:
		assert self.mean_ is not None and self.scale_ is not None
		X = np.asarray(X, dtype=np.float64)
		return (X - self.mean_) / self.scale_

