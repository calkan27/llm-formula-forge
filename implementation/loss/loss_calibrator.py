from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple
import math
import numpy as np

from .quantile_minmax import QuantileMinMax


@dataclass
class LossCalibrator:
	"""
	Signed, normalized, scalarized loss with rolling quantile min–max:

		L = α * N_E(E) + β * N_C(C) - γ * N_S(S)

	Defaults: (α, β, γ) = (0.7, 0.2, 0.1)

	Theory notes:
	  • E ∈ [0,1] for regression via clip(1 - R^2, 0, 1), with a bounded normalized-MAE fallback.
	  • Each N_* is a quantile min–max over a fixed batch context; monotone in its argument
		within that batch. Hence for fixed (E, C) and fixed bounds, L is strictly
		nonincreasing in S when γ > 0.
	"""
	alpha: float = 0.7
	beta: float = 0.2
	gamma: float = 0.1
	window: int = 512
	q_low: float = 0.05
	q_high: float = 0.95

	_norm_E: QuantileMinMax = field(init=False, repr=False)
	_norm_C: QuantileMinMax = field(init=False, repr=False)
	_norm_S: QuantileMinMax = field(init=False, repr=False)

	def __post_init__(self) -> None:
		self._norm_E = QuantileMinMax(self.window, self.q_low, self.q_high)
		self._norm_C = QuantileMinMax(self.window, self.q_low, self.q_high)
		self._norm_S = QuantileMinMax(self.window, self.q_low, self.q_high)

	@staticmethod
	def regression_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
		"""
		E_reg = clip(1 - R^2, 0, 1), fallback to a bounded normalized MAE in edge cases.

		Fallback: NMAE = mae / (mad + eps), clipped to [0,1], where
		  mad = median(|y - median(y)|). If y is constant (mad≈0), scale with (|median| + 1).
		"""
		y_true = np.asarray(y_true, dtype=float).ravel()
		y_pred = np.asarray(y_pred, dtype=float).ravel()
		n = y_true.size
		if n == 0 or y_pred.size != n:
			return 1.0
		ybar = float(np.mean(y_true))
		ss_tot = float(np.sum((y_true - ybar) ** 2))
		ss_res = float(np.sum((y_true - y_pred) ** 2))
		if ss_tot > 0.0 and np.isfinite(ss_res):
			r2 = 1.0 - ss_res / ss_tot
			return float(np.clip(1.0 - r2, 0.0, 1.0))
		mae = float(np.mean(np.abs(y_true - y_pred)))
		med = float(np.median(y_true))
		mad = float(np.median(np.abs(y_true - med)))
		if mad > 1e-12:
			scale = mad
		else:
			scale = abs(med) + 1.0
		nmae = mae / scale
		return float(np.clip(nmae, 0.0, 1.0))



	@staticmethod
	def classification_error(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
		"""
		Normalized cross-entropy for multi-class classification:
			E = (1/n) * sum_i -log p_i[y_i]  divided by log(K), clipped to [0,1].

		"""
		y = np.asarray(y_true).ravel().astype(int, copy=False)
		P = np.asarray(y_pred_proba, dtype=float)
		if y.size == 0:
			return 1.0
		if P.ndim == 1:
			P = np.stack([1.0 - P, P], axis=1)
		if P.shape[0] != y.size or P.shape[1] < 2:
			return 1.0
		eps = 1e-12
		P = np.clip(P, eps, None)
		P /= P.sum(axis=1, keepdims=True)
		idx = (np.arange(P.shape[0]), y)
		ce = float(np.mean(-np.log(P[idx])))
		ce_norm = ce / math.log(P.shape[1])
		return float(np.clip(ce_norm, 0.0, 1.0))

	@staticmethod
	def E_from_regression(y_true: np.ndarray, y_pred: np.ndarray) -> float:
		return LossCalibrator.regression_error(y_true, y_pred)

	@staticmethod
	def E_from_classification(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
		return LossCalibrator.classification_error(y_true, y_pred_proba)


	def update_observation(self, E: float, C: float, S: float) -> None:
		"""Feed one (E, C, S) triple into rolling windows (does not compute a loss)."""
		self._norm_E.update(E)
		self._norm_C.update(C)
		self._norm_S.update(S)

	def update_many(self, triples: Iterable[Tuple[float, float, float]]) -> None:
		for (E, C, S) in triples:
			self.update_observation(E, C, S)

	def loss_with_snapshot(self, E: float, C: float, S: float) -> float:
		"""
		Compute loss using a *fixed* snapshot of normalization bounds for (E, C, S).
		This preserves the per-batch monotonicity: for fixed (E, C) and bounds,
		L is strictly nonincreasing in S if γ > 0.
		"""
		loE, hiE = self._norm_E.snapshot_bounds()
		loC, hiC = self._norm_C.snapshot_bounds()
		loS, hiS = self._norm_S.snapshot_bounds()

		nE = QuantileMinMax.normalize(E, loE, hiE)
		nC = QuantileMinMax.normalize(C, loC, hiC)
		nS = QuantileMinMax.normalize(S, loS, hiS)

		return self.alpha * nE + self.beta * nC - self.gamma * nS

	def losses_with_snapshot(self, triples: Iterable[Tuple[float, float, float]]) -> List[float]:
		loE, hiE = self._norm_E.snapshot_bounds()
		loC, hiC = self._norm_C.snapshot_bounds()
		loS, hiS = self._norm_S.snapshot_bounds()
		out: List[float] = []
		for (E, C, S) in triples:
			nE = QuantileMinMax.normalize(E, loE, hiE)
			nC = QuantileMinMax.normalize(C, loC, hiC)
			nS = QuantileMinMax.normalize(S, loS, hiS)
			out.append(self.alpha * nE + self.beta * nC - self.gamma * nS)
		return out


	def observe_regression(self,
						   y_true: np.ndarray,
						   y_pred: np.ndarray,
						   C: float,
						   S: float) -> float:
		"""
		Compute regression E, update (E,C,S) into the rolling windows, and return E.
		"""
		E = self.E_from_regression(y_true, y_pred)
		self.update_observation(E, C, S)
		return E

	def observe_classification(self,
							   y_true: np.ndarray,
							   y_pred_proba: np.ndarray,
							   C: float,
							   S: float) -> float:
		"""
		Compute classification E, update (E,C,S) into the rolling windows, and return E.
		"""
		E = self.E_from_classification(y_true, y_pred_proba)
		self.update_observation(E, C, S)
		return E

	def loss_from_regression_snapshot(self,
									  y_true: np.ndarray,
									  y_pred: np.ndarray,
									  C: float,
									  S: float) -> float:
		"""
		Compute regression E and return scalarized loss L using a fixed snapshot
		of normalization bounds (does NOT update rolling windows).
		"""
		E = self.E_from_regression(y_true, y_pred)
		return self.loss_with_snapshot(E, C, S)

	def loss_from_classification_snapshot(self,
										  y_true: np.ndarray,
										  y_pred_proba: np.ndarray,
										  C: float,
										  S: float) -> float:
		"""
		Compute classification E and return scalarized loss L using a fixed snapshot
		of normalization bounds (does NOT update rolling windows).
		"""
		E = self.E_from_classification(y_true, y_pred_proba)
		return self.loss_with_snapshot(E, C, S)

