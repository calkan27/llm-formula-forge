from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, List
import math
import numpy as np


@dataclass(frozen=True)
class MIRankItem:
	"""Container for MI ranking with a deterministic tie key."""
	name: str
	score: float
	tie_key: str


class EqualWidthMI:
	"""
	Equal-width binning and plug-in mutual information (MI) estimator.
	"""

	@staticmethod
	def equal_width_bins(x: np.ndarray, b: int = 16) -> np.ndarray:
		"""
		Discretize a 1-D array into equal-width bins [b].
		"""
		x = np.asarray(x, dtype=float).ravel()
		b = int(b)
		if x.size == 0:
			return np.zeros(0, dtype=np.int64)
		if b < 1:
			return np.zeros_like(x, dtype=np.int64)

		mn = float(np.nanmin(x))
		mx = float(np.nanmax(x))
		if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
			return np.zeros_like(x, dtype=np.int64)

		edges = np.linspace(mn, mx, b + 1)
		idx = np.searchsorted(edges, x, side="right") - 1
		idx[idx < 0] = 0
		last = b - 1
		idx[idx > last] = last
		return idx.astype(np.int64, copy=False)

	@staticmethod
	def _mi_from_binned(xb: np.ndarray, yb: np.ndarray) -> float:
		"""
		MI estimator on *discrete* xb, yb via plug-in counts:
			I = sum_ij p_ij log( p_ij / (p_i p_j) )

		Skips zero cells.
		Returns MI in nats.
		"""
		xb = np.asarray(xb, dtype=np.int64).ravel()
		yb = np.asarray(yb, dtype=np.int64).ravel()

		if xb.size != yb.size:
			print(f"_mi_from_binned: size mismatch xb={xb.size}, yb={yb.size}; returning 0.0")
			return 0.0

		n = int(xb.size)
		if n == 0:
			return 0.0

		if xb.size > 0:
			kx = int(np.max(xb)) + 1
		else:
			kx = 1
		if yb.size > 0:
			ky = int(np.max(yb)) + 1
		else:
			ky = 1

		if kx <= 0 or ky <= 0:
			print(f"_mi_from_binned: nonpositive alphabet sizes kx={kx}, ky={ky}; returning 0.0")
			return 0.0

		counts = np.zeros((kx, ky), dtype=np.int64)
		np.add.at(counts, (xb, yb), 1)

		if counts.sum() != n or n <= 0:
			return 0.0

		px = counts.sum(axis=1) / float(n)
		py = counts.sum(axis=0) / float(n)
		nz = counts > 0

		if not np.any(nz):
			return 0.0

		i_idx, j_idx = np.nonzero(nz)
		pxy = counts[nz].astype(float) / float(n)

		valid_mask = (pxy > 0) & (px[i_idx] > 0) & (py[j_idx] > 0)
		if not np.all(valid_mask):
			pxy = pxy[valid_mask]
			i_idx = i_idx[valid_mask]
			j_idx = j_idx[valid_mask]
			if pxy.size == 0:
				return 0.0

		val = pxy * (np.log(pxy) - np.log(px[i_idx]) - np.log(py[j_idx]))
		return float(np.sum(val))



	@staticmethod
	def mi_equal_width(
		x: np.ndarray,
		y: np.ndarray,
		b: int = 16,
		return_bits: bool = False,
	) -> float:
		"""
		Equal-width binned MI estimator for *numeric* x,y (both discretized).
		"""
		xb = EqualWidthMI.equal_width_bins(x, b=b)
		yb = EqualWidthMI.equal_width_bins(y, b=b)
		I_nats = EqualWidthMI._mi_from_binned(xb, yb)
		if return_bits:
			return float(I_nats / math.log(2.0))
		return I_nats

	@staticmethod
	def rank_by_mi_with_ties(
		features: Sequence[Tuple[str, np.ndarray, str]],
		y: np.ndarray,
		b: int = 16,
	) -> List[MIRankItem]:
		"""
		Rank features by MI(y; f) descending with a deterministic tie policy.
		"""
		yy = np.asarray(y, dtype=float).ravel()
		items: List[MIRankItem] = []
		for (nm, fv, tk) in features:
			fv_arr = np.asarray(fv, dtype=float).ravel()
			if fv_arr.size != yy.size:
				score = 0.0
			else:
				score = EqualWidthMI.mi_equal_width(fv_arr, yy, b=b)
			items.append(MIRankItem(nm, float(score), str(tk)))
		items.sort(key=lambda it: (-it.score, it.tie_key, it.name))
		return items

