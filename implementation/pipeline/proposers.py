from __future__ import annotations
from dataclasses import dataclass
from typing import List, Protocol, Sequence
import random
import numpy as np

from implementation.lm.manifest import LMDeterminismManifest



@dataclass(frozen=True)
class ProposalBatch:
	texts: List[str]
	provenance: str  


class Proposer(Protocol):
	def propose(self, k: int) -> ProposalBatch: ...


class LMProposer:
	"""Deterministic LM-text proposer (no external calls here)."""

	def __init__(self,
				 manifest: LMDeterminismManifest,
				 candidate_pool: Sequence[str]) -> None:
		self.manifest = manifest
		rnd = random.Random(self.manifest.derived_seed)
		self._pool = list(candidate_pool)
		rnd.shuffle(self._pool)

	def propose(self, k: int) -> ProposalBatch:
		out: List[str] = []
		n = len(self._pool)
		if n == 0 or k <= 0:
			return ProposalBatch([], "lm")
		stride = max(1, (self.manifest.derived_seed % 7) + 1)
		idx = (self.manifest.derived_seed % n)
		for _ in range(k):
			out.append(self._pool[idx])
			idx = (idx + stride) % n
		return ProposalBatch(out, "lm")


class MutationProposer:
	"""Local algebraic mutator over survivor texts (deterministic)."""

	def __init__(self, survivors_text: Sequence[str], seed: int = 1729) -> None:
		self._base = list(survivors_text)
		self._rnd = random.Random(seed)

	def _one_mutation(self, s: str) -> str:
		choices = [
			lambda t: f"padd({t}, pdiv(x, 1e6))",
			lambda t: f"pmul({t}, pcos(y))",
			lambda t: t.replace('psin(', 'pcos(') if 'psin(' in t else f"pmul(psin(x), {t})",
			lambda t: "psqrt(padd(pmul(x,x), pmul(y,y)))",
		]
		f = choices[self._rnd.randrange(len(choices))]
		return f(s)

	def propose(self, k: int) -> ProposalBatch:
		out: List[str] = []
		if not self._base or k <= 0:
			return ProposalBatch([], "mut")
		idx = 0
		for _ in range(k):
			src = self._base[idx % len(self._base)]
			out.append(self._one_mutation(src))
			idx += 1
		return ProposalBatch(out, "mut")


class ERCProposer:
	"""ERC-injecting proposer (deterministic templates)."""

	def __init__(self, seed: int = 7) -> None:
		self._rnd = np.random.default_rng(seed)

	def _emit(self) -> str:
		temps = [
			"pdiv(x,1e6)",
			"pow2(x)",
			"powm1(padd(psqrt(pmul(x,x)), pdiv(x,1e3)))",
			"pmul(psin(x), pcos(y))",
		]
		i = int(self._rnd.integers(0, len(temps)))
		return temps[i]

	def propose(self, k: int) -> ProposalBatch:
		out = [self._emit() for _ in range(max(0, k))]
		return ProposalBatch(out, "erc")

