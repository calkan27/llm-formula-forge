"""StratifiedHoldout
Deterministic, NumPy-only 80/20 (or generic p/1−p) stratified holdout splitter.

This module provides a small façade that:
  • Coerces labels to 1-D int64 and builds per-class bins (via BaseSplitUtils),
  • Shuffles indices deterministically using a seeded PCG64 Generator,
  • Allocates per-class test counts by fractional rounding with a stable tie policy,
  • Returns disjoint int64 train/test index arrays covering the dataset.
"""

from __future__ import annotations
import numpy as np
from .base import BaseSplitUtils as U


class StratifiedHoldout:
	"""Seeded, deterministic stratified holdout with per-class rounding and stable tie-breaks."""

	@staticmethod
	def split(y, test_size: float = 0.2, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
		"""
		Compute a stratified train/test split with fixed seed and stable rounding.
		"""
		ya, r, labels, shuf = StratifiedHoldout._prepare(y, seed)
		target = U.rounded_targets_per_class(shuf, float(test_size))
		train_parts, test_parts = StratifiedHoldout._split_bins(shuf, labels, target)
		tr, te = StratifiedHoldout._assemble(train_parts, test_parts)
		return tr, te

	@staticmethod
	def _prepare(y, seed: int) -> tuple[np.ndarray, np.random.Generator, np.ndarray, dict[int, np.ndarray]]:
		"""
		Coerce labels, build per-class bins, and deterministically shuffle each bin.
		"""
		ya = U.to_1d_int(y)
		r = U.rng(int(seed))
		labels, bins = U.class_bins(ya)
		shuf: dict[int, np.ndarray] = {}
		for i in range(labels.shape[0]):
			lb = int(labels[i])
			shuf[lb] = U.shuffle_rows(bins[lb], r)
		return ya, r, labels, shuf

	@staticmethod
	def _split_bins(shuf: dict[int, np.ndarray], labels: np.ndarray, target: dict[int, int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
		"""
		Slice per-class shuffled bins into train/test parts according to target counts.
		"""
		train_parts: list[np.ndarray] = []
		test_parts: list[np.ndarray] = []
		for i in range(labels.shape[0]):
			lb = int(labels[i])
			k = int(target[lb])
			test = shuf[lb][:k]
			train = shuf[lb][k:]
			train_parts.append(train)
			test_parts.append(test)
		return train_parts, test_parts

	@staticmethod
	def _assemble(train_parts: list[np.ndarray], test_parts: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
		"""
		Concatenate per-class parts and return final 1-D int64 train/test indices.
		"""
		if len(train_parts) > 0:
			tr = np.concatenate(train_parts, axis=0)
		else:
			tr = np.zeros(0, dtype=np.int64)
		if len(test_parts) > 0:
			te = np.concatenate(test_parts, axis=0)
		else:
			te = np.zeros(0, dtype=np.int64)
		return tr.astype(np.int64, copy=False), te.astype(np.int64, copy=False)

