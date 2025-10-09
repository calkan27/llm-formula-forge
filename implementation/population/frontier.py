"""
Pareto2D — ε-Pareto frontier utilities (2D Jensen sweep + quadratic baseline)

This module provides:
  • frontier_indices      — O(n log n) Jensen-style ε-frontier sweep in 2D
  • frontier_mask         — Boolean membership mask for the ε-frontier
  • frontier              — (C_frontier, E_frontier, indices) tuple
  • baseline_indices_n2   — O(n^2) reference using stable-lex ε-dominance
  • sweep_perC_then_global— (optional) helper that mirrors the scheduler’s
							 two-stage policy (per-C consolidation → ε sweep)
"""

from __future__ import annotations
import numpy as np


class Pareto2D:
	@staticmethod
	def _finite_view(C, E):
		"""
		Internal: return (c, e, idx) arrays after masking to finite rows.

		Rows where C or E is NaN/Inf are **dropped** from consideration. The
		returned `idx` maps positions in the filtered arrays back to the original
		input indices.
		"""
		c_full = np.asarray(C, dtype=float).ravel()
		e_full = np.asarray(E, dtype=float).ravel()
		n = int(c_full.size)
		mask = np.isfinite(c_full) & np.isfinite(e_full)
		if mask.all():
			return c_full, e_full, np.arange(n, dtype=np.int64)
		return c_full[mask], e_full[mask], np.nonzero(mask)[0].astype(np.int64, copy=False)

	@staticmethod
	def frontier_indices(C, E, eps: float = 0.0) -> list[int]:
		"""
		O(n log n) ε-Pareto frontier via a Jensen-style sweep.
		"""
		c, e, base_idx = Pareto2D._finite_view(C, E)
		n = int(c.size)
		if n == 0:
			return []
		idx_local = np.arange(n, dtype=np.int64)
		order_local = np.lexsort((idx_local, e, c))  
		best = float("inf")
		out_local: list[int] = []
		ep = max(0.0, float(eps))
		for k in order_local:
			val = float(e[k])
			if ep == 0.0:
				if val < best:
					out_local.append(int(k))
					best = val
			else:
				if val < best - ep:
					out_local.append(int(k))
					best = val
		return [int(base_idx[k]) for k in out_local]

	@staticmethod
	def frontier_mask(C, E, eps: float = 0.0) -> np.ndarray:
		"""
		Boolean mask over inputs indicating ε-Pareto membership.
		"""
		C = np.asarray(C)
		n = int(C.size)
		m = np.zeros(n, dtype=bool)
		for i in Pareto2D.frontier_indices(C, E, eps):
			m[i] = True
		return m

	@staticmethod
	def frontier(C, E, eps: float = 0.0):
		"""
		Tuple form of the ε-Pareto frontier.
		"""
		m = Pareto2D.frontier_mask(C, E, eps)
		c = np.asarray(C, dtype=float).ravel()
		e = np.asarray(E, dtype=float).ravel()
		return c[m], e[m], np.nonzero(m)[0]

	@staticmethod
	def baseline_indices_n2(C, E, eps: float = 0.0) -> list[int]:
		"""
		O(n^2) ε-frontier baseline using stable-lex ε-dominance.
		"""
		c, e, base_idx = Pareto2D._finite_view(C, E)
		n = int(c.size)
		if n == 0:
			return []
		idx_local = np.arange(n, dtype=np.int64)
		order_local = np.lexsort((idx_local, e, c))
		pos = {int(k): p for p, k in enumerate(order_local)}
		ep = max(0.0, float(eps))

		keep_local: list[int] = []
		for i in range(n):
			ci = float(c[i])
			ei = float(e[i])
			dominated = False
			for j in range(n):
				if j == i:
					continue
				if pos[j] < pos[i]:
					cj = float(c[j])
					ej = float(e[j])
					if cj <= ci and ej <= ei + ep:
						dominated = True
						break
			if not dominated:
				keep_local.append(i)

		keep_local.sort(key=lambda k: pos[k])
		out_local: list[int] = []
		last_c = last_e = None
		for k in keep_local:
			ck = float(c[k]); ek = float(e[k])
			if last_c is None:
				out_local.append(k); last_c, last_e = ck, ek
			else:
				if ck == last_c and ek == last_e:
					continue
				out_local.append(k); last_c, last_e = ck, ek

		return [int(base_idx[k]) for k in out_local]

	@staticmethod
	def sweep_perC_then_global(C, E, eps: float = 0.0) -> list[int]:
		"""
		Two-stage helper mirroring the scheduler’s policy:
		"""
		c, e, base_idx = Pareto2D._finite_view(C, E)
		n = int(c.size)
		if n == 0:
			return []
		idx_local = np.arange(n, dtype=np.int64)
		order_local = np.lexsort((idx_local, e, c))
		ep = max(0.0, float(eps))

		reps: list[int] = []
		seen_c = {}
		for k in order_local:
			ck = float(c[k]); ek = float(e[k])
			if ck not in seen_c:
				seen_c[ck] = (k, ek)
				reps.append(k)
			else:
				rep_k, rep_e = seen_c[ck]
				if ek < rep_e - ep:
					seen_c[ck] = (k, ek)
					reps[reps.index(rep_k)] = k

		reps.sort(key=lambda k: (c[k], e[k], k))
		best = float("inf")
		out_local: list[int] = []
		for k in reps:
			val = float(e[k])
			if ep == 0.0:
				if val < best:
					out_local.append(k); best = val
			else:
				if val < best - ep:
					out_local.append(k); best = val

		return [int(base_idx[k]) for k in out_local]

