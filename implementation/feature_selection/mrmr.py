from __future__ import annotations
from typing import Sequence, List
import numpy as np


class CorrelationDiversity:
	"""
	mRMR-like correlation diversity selection with a fixed cutoff and stable policy.

	API (static methods)
	--------------------
	pearsonr_stable(a, b) -> float
	select_diverse_by_corr(X, ranked_indices, top_k, corr_cutoff=0.9) -> List[int]
	"""

	@staticmethod
	def pearsonr_stable(a: np.ndarray, b: np.ndarray) -> float:
		"""
		Deterministic Pearson correlation:
		  r = cov(a,b) / (std(a)*std(b))

		Variance floor: if either variance ≤ 1e-18, returns r := 0.0.
		Uses float64 and a stable mean-subtraction.
		"""
		x = np.asarray(a, dtype=np.float64).ravel()
		y = np.asarray(b, dtype=np.float64).ravel()
		if x.size == 0 or y.size == 0 or x.size != y.size:
			return 0.0
		xm = float(np.mean(x))
		ym = float(np.mean(y))
		xd = x - xm
		yd = y - ym
		vx = float(np.dot(xd, xd))
		vy = float(np.dot(yd, yd))
		if vx <= 1e-18 or vy <= 1e-18:
			return 0.0
		num = float(np.dot(xd, yd))
		den = float(np.sqrt(vx * vy))
		return float(num / den)

	@staticmethod
	def select_diverse_by_corr(
		X: np.ndarray,
		ranked_indices: Sequence[int],
		top_k: int,
		corr_cutoff: float = 0.9,
	) -> List[int]:
		"""
		Walk ranked_indices (already MI-sorted), accept a candidate iff
		|corr(candidate, any selected)| <= corr_cutoff.
		Deterministic: first-come from ranked list; stable ties preserved upstream.
		"""
		X = np.asarray(X, dtype=np.float64)
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		n, d = X.shape
		out: List[int] = []
		for j in ranked_indices:
			if len(out) >= int(top_k):
				break
			accept = True
			for i_sel in out:
				r = CorrelationDiversity.pearsonr_stable(X[:, j], X[:, i_sel])
				if abs(r) > float(corr_cutoff):  
					accept = False
					break
			if accept:
				out.append(int(j))
		return out

