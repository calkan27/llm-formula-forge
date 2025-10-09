from __future__ import annotations
import numpy as np
from .base import BaseSplitUtils as U

class StratifiedKFold:
	@staticmethod
	def splits_stratified(y, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
		ya, labels, parts = StratifiedKFold._build_chunks(y, k, seed)
		out: list[tuple[np.ndarray, np.ndarray]] = []
		for f in range(k):
			tr, te = StratifiedKFold._assemble_fold(parts, labels, f, k)
			out.append((tr, te))
		return out

	@staticmethod
	def _build_chunks(y, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[int, list[np.ndarray]]]:
		ya = U.to_1d_int(y)
		k = int(max(2, k))
		r = U.rng(int(seed))
		labels, bins = U.class_bins(ya)
		parts: dict[int, list[np.ndarray]] = {}
		for i in range(labels.shape[0]):
			lb = int(labels[i])
			idx = U.shuffle_rows(bins[lb], r)
			n = int(idx.shape[0])
			base = n // k
			rem = n - base * k
			chunks: list[np.ndarray] = []
			p0 = 0
			for j in range(k):
				if j < rem:
					extra = 1
				else:
					extra = 0
				p1 = p0 + base + extra
				chunks.append(idx[p0:p1])
				p0 = p1
			parts[lb] = chunks
		return ya, labels, parts


	@staticmethod
	def _assemble_fold(parts: dict[int, list[np.ndarray]], labels: np.ndarray, f: int, k: int) -> tuple[np.ndarray, np.ndarray]:
		test_parts: list[np.ndarray] = []
		train_parts: list[np.ndarray] = []
		for i in range(labels.shape[0]):
			lb = int(labels[i])
			for j in range(k):
				if j == f:
					test_parts.append(parts[lb][j])
				else:
					train_parts.append(parts[lb][j])
		if len(train_parts) > 0:
			tr = np.concatenate(train_parts, axis=0)
		else:
			tr = np.zeros(0, dtype=np.int64)
		if len(test_parts) > 0:
			te = np.concatenate(test_parts, axis=0)
		else:
			te = np.zeros(0, dtype=np.int64)
		return tr.astype(np.int64, copy=False), te.astype(np.int64, copy=False)


