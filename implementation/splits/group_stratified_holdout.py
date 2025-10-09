"""
GroupStratifiedHoldout
----------------------
Deterministic, group-aware stratified p/1−p holdout splitter with zero leakage.

This splitter:
  • Treats `groups` as indivisible units (no entity appears in both splits).
  • Targets class totals in the test set via fractional rounding with a stable tie policy.
  • Uses a seeded PCG64 RNG for reproducible group ordering and decisions.
  • Applies a greedy assignment with a large overflow penalty to match per-class targets.
"""


from __future__ import annotations
import numpy as np
from .base import BaseSplitUtils as U




class GroupStratifiedHoldout:
	@staticmethod
	def split(groups, y, test_size: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
		"""
		Return stratified, group-disjoint train/test index arrays.
		"""
		ga, ya, labels, gids, order, mat = GroupStratifiedHoldout._prepare(groups, y, seed)
		target = GroupStratifiedHoldout._class_targets(ya, labels, float(test_size))
		assign_test = GroupStratifiedHoldout._assign(mat, target, order)
		tr, te = GroupStratifiedHoldout._emit_indices(ga, gids, assign_test)
		return tr, te

	@staticmethod
	def _prepare(groups, y, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Coerce inputs, derive deterministic group order, and build a group×class count matrix.
		"""
		ga = U.to_1d_int(groups)
		ya = U.to_1d_int(y)
		labels, _ = U.class_bins(ya)
		gids = np.unique(ga)
		r = U.rng(int(seed))
		order = np.arange(gids.shape[0], dtype=np.int64)
		r.shuffle(order)
		G = gids.shape[0]
		C = labels.shape[0]
		mat = np.zeros((G, C), dtype=np.int64)
		for i in range(G):
			gb = int(gids[i])
			idx = np.nonzero(ga == gb)[0].astype(np.int64, copy=False)
			for j in range(C):
				lb = int(labels[j])
				mat[i, j] = int(np.sum(ya[idx] == lb))
		return ga, ya, labels, gids, order, mat

	@staticmethod
	def _class_targets(ya: np.ndarray, labels: np.ndarray, frac: float) -> np.ndarray:
		"""
		Compute integer per-class test-set targets using fractional rounding with stable ties.
		"""
		C = labels.shape[0]
		nc = np.zeros(C, dtype=np.int64)
		for i in range(C):
			lb = int(labels[i])
			nc[i] = int(np.sum(ya == lb))
		raw = nc.astype(np.float64) * float(frac)
		base = np.floor(raw).astype(np.int64)
		rem = raw - base.astype(np.float64)
		idx = np.arange(C, dtype=np.int64)
		order = np.lexsort((idx, -rem))
		need = int(np.round(float(raw.sum()))) - int(base.sum())
		tgt = base.copy()
		for j in range(need):
			k = int(order[j])
			tgt[k] = tgt[k] + 1
		return tgt

	@staticmethod
	def _assign(mat: np.ndarray, target: np.ndarray, order: np.ndarray) -> np.ndarray:
		"""
		Select a subset of groups for the test split to meet class targets.

		Greedy loop adds one group at a time, minimizing a score that prefers
		reducing deficits and heavily penalizes overflow. Ties break by lower
		group index within the shuffled `order`.
		"""
		G = mat.shape[0]
		C = mat.shape[1]
		assign = np.zeros(G, dtype=np.int64)
		load = np.zeros(C, dtype=np.int64)
		unassigned = np.ones(G, dtype=bool)
		overflow_scale = 1_000_000
		deficit = target - load
		deficit = np.where(deficit > 0, deficit, 0)
		for _ in range(G):
			if not np.any(deficit > 0):
				break
			best_i = -1
			best_score = None
			for j in range(order.shape[0]):
				i = int(order[j])
				if unassigned[i]:
					proj = load + mat[i]
					def_after = target - proj
					def_after = np.where(def_after > 0, def_after, 0)
					ov_after = proj - target
					ov_after = np.where(ov_after > 0, ov_after, 0)
					score = int(def_after.sum()) + overflow_scale * int(ov_after.sum())
					if best_score is None:
						best_i = i
						best_score = score
					else:
						if score < best_score:
							best_i = i
							best_score = score
						else:
							if score == best_score:
								if i < best_i:
									best_i = i
									best_score = score
			if best_i < 0:
				break
			assign[best_i] = 1
			unassigned[best_i] = False
			load = load + mat[best_i]
			deficit = target - load
			deficit = np.where(deficit > 0, deficit, 0)
		return assign

	@staticmethod
	def _emit_indices(ga: np.ndarray, gids: np.ndarray, assign_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
		Materialize 1-D int64 train/test index arrays from group test assignments.

		All rows belonging to a selected test group go to the test indices.
		All remaining rows form the train indices.
		"""
		test_mask = np.zeros(ga.shape[0], dtype=bool)
		for i in range(gids.shape[0]):
			if assign_test[i] == 1:
				gb = int(gids[i])
				test_mask[ga == gb] = True
		te = np.nonzero(test_mask)[0].astype(np.int64, copy=False)
		tr = np.nonzero(~test_mask)[0].astype(np.int64, copy=False)
		return tr, te

