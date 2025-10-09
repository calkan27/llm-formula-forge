from __future__ import annotations
import numpy as np

"""
BaseSplitUtils
--------------
Deterministic, NumPy-only helpers shared by stratified and group-stratified
splitters. Utilities here centralize:

  • Seeded PRNG construction (PCG64) for reproducible shuffles
  • Stable 1-D int64 coercion for index vectors
  • Per-class binning (unique labels → index arrays)
  • Row shuffling via a provided Generator (no hidden global state)
  • Fractional rounding of per-class targets with deterministic tie policy

All helpers are side-effect free and return new arrays; callers own ordering
and concatenation semantics.
"""


class BaseSplitUtils:
	@staticmethod
	def rng(seed: int) -> np.random.Generator:
		"""
		Return a deterministic NumPy Generator seeded with PCG64.
		"""
		return np.random.Generator(np.random.PCG64(int(seed)))

	@staticmethod
	def to_1d_int(x) -> np.ndarray:
		"""
		Coerce an input array-like to a contiguous 1-D int64 array.
		"""
		a = np.asarray(x)
		if a.ndim != 1:
			a = a.reshape(-1)
		return a.astype(np.int64, copy=False)

	@staticmethod
	def class_bins(y: np.ndarray) -> tuple[np.ndarray, dict[int, np.ndarray]]:
		"""
		Bin sample indices by class label with a deterministic label order.
		"""
		labels = np.unique(y)
		bins: dict[int, np.ndarray] = {}
		for i in range(labels.shape[0]):
			lb = int(labels[i])
			idx = np.nonzero(y == lb)[0].astype(np.int64, copy=False)
			bins[lb] = idx
		return labels, bins

	@staticmethod
	def shuffle_rows(idx: np.ndarray, r: np.random.Generator) -> np.ndarray:
		"""
		Return a row-permutation of `idx` using the provided Generator.
		"""
		j = np.arange(idx.shape[0], dtype=np.int64)
		r.shuffle(j)
		return idx[j]

	@staticmethod
	def rounded_targets_per_class(bins: dict[int, np.ndarray], frac: float) -> dict[int, int]:
		"""
		Compute per-class integer targets via fractional rounding with a stable tie policy.
		"""
		target: dict[int, int] = {}
		rems: list[tuple[int, float]] = []
		total_raw = 0.0

		for lb, idx in bins.items():
			nc = int(idx.shape[0])
			raw = float(nc) * float(frac)
			k = int(np.floor(raw))
			target[int(lb)] = k
			rems.append((int(lb), raw - float(k)))
			total_raw += raw

		rems.sort(key=lambda t: (-t[1], t[0]))

		need = int(round(total_raw)) - int(sum(target.values()))

		for j in range(need):
			lb = rems[j][0]
			target[lb] = target[lb] + 1

		return target

