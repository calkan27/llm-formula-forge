from __future__ import annotations
import numpy as np
from .base import BaseSplitUtils as U

"""
GroupStratifiedKFold
--------------------
Deterministic, group-aware stratified K-fold splitter with zero leakage.
"""


class GroupStratifiedKFold:
	@staticmethod
	def splits_group_stratified(groups, y, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
		"""
		Build K stratified, group-disjoint folds with deterministic balancing.
		"""
		ga, ya, labels, gids, order, mat = GroupStratifiedKFold._prepare(groups, y, seed)
		target = GroupStratifiedKFold._fold_targets(ya, labels, int(k))
		assign = GroupStratifiedKFold._assign(mat, target, order)
		GroupStratifiedKFold._refine_assign(assign, mat, target)
		GroupStratifiedKFold._refine_pair_swaps(assign, mat, target)
		out = GroupStratifiedKFold._emit_folds(ga, gids, assign, int(k))
		return out

	@staticmethod
	def _prepare(groups, y, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"""
		Normalize inputs, derive deterministic group order, and build the group×class matrix.
		"""
		ga = U.to_1d_int(groups)
		ya = U.to_1d_int(y)
		r = U.rng(int(seed))
		labels, _ = U.class_bins(ya)
		gids = np.unique(ga)
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
	def _fold_targets(ya: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
		"""
		Compute per-fold class targets by distributing total class counts across k folds.
		"""
		C = labels.shape[0]
		nc = np.zeros(C, dtype=np.int64)
		for i in range(C):
			lb = int(labels[i])
			nc[i] = int(np.sum(ya == lb))
		base = nc // k
		rem = nc - base * k
		target = np.zeros((k, C), dtype=np.int64)
		for f in range(k):
			for j in range(C):
				if f < rem[j]:
					extra = 1
				else:
					extra = 0
				target[f, j] = int(base[j] + extra)
		return target

	@staticmethod
	def _score_group_for_fold(load: np.ndarray, mat: np.ndarray, target: np.ndarray, f: int, i: int) -> int:
		"""
		Score a candidate assignment of group i to fold f.
		"""
		proj = load[f] + mat[i]
		deficit = target[f] - proj
		deficit = np.where(deficit > 0, deficit, 0)
		overflow = proj - target[f]
		overflow = np.where(overflow > 0, overflow, 0)
		score = int(deficit.sum()) + int(1_000_000 * overflow.sum())
		return score

	@staticmethod
	def _assign_initial_fill(assign: np.ndarray, load: np.ndarray, mat: np.ndarray, target: np.ndarray, order: np.ndarray) -> None:
		"""
		Fill each fold once using greedy scores with stable ties.
		"""
		G = mat.shape[0]
		k = target.shape[0]
		for f in range(k):
			best = -1
			best_score = None
			for j in range(order.shape[0]):
				i = int(order[j])
				if assign[i] >= 0:
					continue
				score = GroupStratifiedKFold._score_group_for_fold(load, mat, target, f, i)
				if best_score is None:
					best = i
					best_score = score
				else:
					if score < best_score:
						best = i
						best_score = score
					else:
						if score == best_score:
							if i < best:
								best = i
								best_score = score
			if best >= 0:
				assign[best] = f
				load[f] = load[f] + mat[best]

	@staticmethod
	def _assign_remaining(assign: np.ndarray, load: np.ndarray, mat: np.ndarray, target: np.ndarray, order: np.ndarray) -> None:
		"""
		Assign remaining groups to folds using overflow-penalized scores and load tiebreakers.
		"""
		k = target.shape[0]
		for j in range(order.shape[0]):
			i = int(order[j])
			if assign[i] >= 0:
				continue
			best_f = 0
			best_score = None
			for f in range(k):
				proj = load[f] + mat[i]
				deficit = target[f] - proj
				deficit = np.where(deficit > 0, deficit, 0)
				overflow = proj - target[f]
				overflow = np.where(overflow > 0, overflow, 0)
				score = int(deficit.sum()) + int(1_000_000 * overflow.sum()) + int(load[f].sum())
				if best_score is None:
					best_f = f
					best_score = score
				else:
					if score < best_score:
						best_f = f
						best_score = score
					else:
						if score == best_score:
							if f < best_f:
								best_f = f
								best_score = score
			assign[i] = best_f
			load[best_f] = load[best_f] + mat[i]

	@staticmethod
	def _assign(mat: np.ndarray, target: np.ndarray, order: np.ndarray) -> np.ndarray:
		"""
		Initial greedy assignment of groups to folds with overflow penalty and stable ties.
		"""
		G = mat.shape[0]
		k = target.shape[0]
		C = target.shape[1]
		assign = -np.ones(G, dtype=np.int64)
		load = np.zeros((k, C), dtype=np.int64)
		GroupStratifiedKFold._assign_initial_fill(assign, load, mat, target, order)
		GroupStratifiedKFold._assign_remaining(assign, load, mat, target, order)
		return assign

	@staticmethod
	def _objective(load: np.ndarray, target: np.ndarray) -> int:
		"""
		Compute the L1 deviation between achieved per-fold class loads and targets.
		"""
		d = load.astype(np.int64) - target.astype(np.int64)
		d = np.abs(d)
		return int(d.sum())

	@staticmethod
	def _build_load_and_counts(assign: np.ndarray, mat: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
		"""
		Build per-fold load matrix and group counts.
		"""
		k = target.shape[0]
		C = target.shape[1]
		G = mat.shape[0]
		load = np.zeros((k, C), dtype=np.int64)
		counts = np.zeros(k, dtype=np.int64)
		for i in range(G):
			f = int(assign[i])
			load[f] = load[f] + mat[i]
			counts[f] = counts[f] + 1
		return load, counts

	@staticmethod
	def _best_relocation_for_group(i: int, assign: np.ndarray, load: np.ndarray, counts: np.ndarray, mat: np.ndarray, target: np.ndarray, base_obj: int) -> tuple[int, int]:
		"""
		Find the best destination fold for group i given the current objective.
		"""
		k = target.shape[0]
		f_cur = int(assign[i])
		best_f = f_cur
		best_obj = base_obj
		for f in range(k):
			if f == f_cur:
				continue
			if counts[f_cur] <= 1:
				continue
			load[f_cur] = load[f_cur] - mat[i]
			load[f] = load[f] + mat[i]
			obj = GroupStratifiedKFold._objective(load, target)
			if obj < best_obj:
				best_obj = obj
				best_f = f
			else:
				load[f] = load[f] - mat[i]
				load[f_cur] = load[f_cur] + mat[i]
		return best_f, best_obj

	@staticmethod
	def _apply_relocation(i: int, f_cur: int, f_new: int, assign: np.ndarray, load: np.ndarray, counts: np.ndarray, mat: np.ndarray) -> None:
		"""
		Apply a relocation of group i from f_cur to f_new.
		"""
		load[f_cur] = load[f_cur] - mat[i]
		load[f_new] = load[f_new] + mat[i]
		assign[i] = f_new
		counts[f_cur] = counts[f_cur] - 1
		counts[f_new] = counts[f_new] + 1

	@staticmethod
	def _refine_assign(assign: np.ndarray, mat: np.ndarray, target: np.ndarray) -> None:
		"""
		Single-move refinement: relocate a group to a different fold only if it improves the objective.
		"""
		load, counts = GroupStratifiedKFold._build_load_and_counts(assign, mat, target)
		base_obj = GroupStratifiedKFold._objective(load, target)
		passes = 4
		for _p in range(passes):
			improved = False
			G = mat.shape[0]
			for i in range(G):
				f_cur = int(assign[i])
				best_f, best_obj = GroupStratifiedKFold._best_relocation_for_group(i, assign, load, counts, mat, target, base_obj)
				if best_f != f_cur:
					GroupStratifiedKFold._apply_relocation(i, f_cur, best_f, assign, load, counts, mat)
					base_obj = best_obj
					improved = True
			if not improved:
				break

	@staticmethod
	def _build_buckets_and_load(assign: np.ndarray, mat: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
		"""
		Build per-fold load matrix and index buckets by fold.
		"""
		k = target.shape[0]
		C = target.shape[1]
		G = mat.shape[0]
		load = np.zeros((k, C), dtype=np.int64)
		buckets: list[list[int]] = [[] for _ in range(k)]
		for i in range(G):
			f = int(assign[i])
			load[f] = load[f] + mat[i]
			buckets[f].append(i)
		return load, buckets

	@staticmethod
	def _pick_surplus_deficit_folds(dev: np.ndarray, c_star: int, k: int) -> tuple[int | None, int | None]:
		"""
		Select surplus and deficit folds for the focused class.
		"""
		surplus_val = None
		deficit_val = None
		f_sur = None
		f_def = None
		for f in range(k):
			val = int(dev[f, c_star])
			if val > 0:
				if surplus_val is None or val > surplus_val:
					surplus_val = val
					f_sur = f
			if val < 0:
				if deficit_val is None or (-val) > deficit_val:
					deficit_val = -val
					f_def = f
		return f_sur, f_def

	@staticmethod
	def _try_best_single_move(f_sur: int, f_def: int, c_star: int, assign: np.ndarray, buckets: list[list[int]], load: np.ndarray, mat: np.ndarray, target: np.ndarray, base_obj: int) -> tuple[int, int]:
		"""
		Try moving a single group from surplus to deficit fold and return best move.
		"""
		best_i = -1
		best_obj = base_obj
		for i in buckets[f_sur]:
			if mat[i, c_star] <= 0:
				continue
			if len(buckets[f_sur]) <= 1:
				continue
			load[f_sur] = load[f_sur] - mat[i]
			load[f_def] = load[f_def] + mat[i]
			obj = GroupStratifiedKFold._objective(load, target)
			if obj < best_obj:
				best_obj = obj
				best_i = i
			load[f_def] = load[f_def] - mat[i]
			load[f_sur] = load[f_sur] + mat[i]
		return best_i, best_obj

	@staticmethod
	def _apply_single_move(i: int, f_sur: int, f_def: int, assign: np.ndarray, buckets: list[list[int]], load: np.ndarray, mat: np.ndarray) -> None:
		"""
		Apply a single-move relocation from surplus to deficit fold.
		"""
		assign[i] = f_def
		buckets[f_sur].remove(i)
		buckets[f_def].append(i)
		load[f_sur] = load[f_sur] - mat[i]
		load[f_def] = load[f_def] + mat[i]

	@staticmethod
	def _try_best_pair_swap(f_sur: int, f_def: int, c_star: int, buckets: list[list[int]], load: np.ndarray, mat: np.ndarray, target: np.ndarray, base_obj: int) -> tuple[tuple[int, int], int]:
		"""
		Try the best pair swap between surplus and deficit folds.
		"""
		best_pair = (-1, -1)
		best_obj = base_obj
		for i in buckets[f_sur]:
			for j in buckets[f_def]:
				load[f_sur] = load[f_sur] - mat[i] + mat[j]
				load[f_def] = load[f_def] - mat[j] + mat[i]
				obj = GroupStratifiedKFold._objective(load, target)
				if obj < best_obj:
					best_obj = obj
					best_pair = (i, j)
				load[f_sur] = load[f_sur] + mat[i] - mat[j]
				load[f_def] = load[f_def] + mat[j] - mat[i]
		return best_pair, best_obj

	@staticmethod
	def _apply_pair_swap(i: int, j: int, f_sur: int, f_def: int, assign: np.ndarray, buckets: list[list[int]], load: np.ndarray, mat: np.ndarray) -> None:
		"""
		Apply a pair swap between surplus and deficit folds.
		"""
		assign[i] = f_def
		assign[j] = f_sur
		buckets[f_sur].remove(i)
		buckets[f_sur].append(j)
		buckets[f_def].remove(j)
		buckets[f_def].append(i)
		load[f_sur] = load[f_sur] - mat[i] + mat[j]
		load[f_def] = load[f_def] - mat[j] + mat[i]

	@staticmethod
	def _refine_pair_swaps(assign: np.ndarray, mat: np.ndarray, target: np.ndarray) -> None:
		"""
		Pair-swap refinement: swap two groups across folds if the swap strictly improves the objective.
		"""
		load, buckets = GroupStratifiedKFold._build_buckets_and_load(assign, mat, target)
		base_obj = GroupStratifiedKFold._objective(load, target)
		max_cycles = 6
		for cycle in range(max_cycles):
			dev = load.astype(np.int64) - target.astype(np.int64)
			abs_dev = np.abs(dev).sum(axis=0)
			if abs_dev.size == 0:
				break
			c_star = int(np.argmax(abs_dev))
			k = target.shape[0]
			f_sur, f_def = GroupStratifiedKFold._pick_surplus_deficit_folds(dev, c_star, k)
			if f_sur is None or f_def is None:
				break
			best_i, best_obj = GroupStratifiedKFold._try_best_single_move(f_sur, f_def, c_star, assign, buckets, load, mat, target, base_obj)
			if best_i >= 0:
				GroupStratifiedKFold._apply_single_move(best_i, f_sur, f_def, assign, buckets, load, mat)
				base_obj = best_obj
				continue
			best_pair, best_obj2 = GroupStratifiedKFold._try_best_pair_swap(f_sur, f_def, c_star, buckets, load, mat, target, base_obj)
			if best_pair != (-1, -1):
				i, j = best_pair
				GroupStratifiedKFold._apply_pair_swap(i, j, f_sur, f_def, assign, buckets, load, mat)
				base_obj = best_obj2
				continue
			break

	@staticmethod
	def _emit_folds(ga: np.ndarray, gids: np.ndarray, assign: np.ndarray, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
		"""
		Materialize (train, test) index arrays per fold from group assignments.
		"""
		out: list[tuple[np.ndarray, np.ndarray]] = []
		for f in range(k):
			test_mask = np.zeros(ga.shape[0], dtype=bool)
			for i in range(gids.shape[0]):
				if assign[i] == f:
					gb = int(gids[i])
					test_mask[ga == gb] = True
			te = np.nonzero(test_mask)[0].astype(np.int64, copy=False)
			tr = np.nonzero(~test_mask)[0].astype(np.int64, copy=False)
			out.append((tr, te))
		return out

