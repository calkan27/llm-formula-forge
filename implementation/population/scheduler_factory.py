"""Factory for constructing a configured PopulationScheduler with fixed (N,K,J,T)
and budget settings, plus deterministic proposers and validator wiring.
"""
from __future__ import annotations
from typing import Sequence
from implementation.pipeline import LoopConfig, PopulationScheduler, ExpressionValidator, ScoreComputer, LMProposer, MutationProposer, ERCProposer
from implementation.lm.manifest import LMDeterminismManifest
from implementation.units.dim import DIMLESS, L

class SchedulerFactory:
	"""Initialize the factory and materialize a LoopConfig.

		Args:
			run_id: Stable identifier for the run/session used to derive seeds.
			round_index: Round index used in seed derivation for determinism.
			budget: Global LM budget (maximum number of LM proposals allowed).
			N: Dataset/corpus size hint (passed through to LoopConfig).
			K: Number of survivors to keep each round.
			J: Total proposals per round after rebalancing.
			J_lm: Target LM proposals per round (capped by remaining budget).
			J_mut: Target mutation proposals per round.
			J_erc: Target ERC proposals per round.
			T: Maximum number of rounds when running the loop.

		Side Effects:
			Creates and stores a prebuilt LoopConfig on self.cfg with the provided parameters.
		"""
	def __init__(
		self,
		run_id: str,
		round_index: int,
		budget: int,
		N: int = 200,
		K: int = 30,
		J: int = 10,
		J_lm: int = 4,
		J_mut: int = 4,
		J_erc: int = 2,
		T: int = 500,
	) -> None:
		self.run_id = str(run_id)
		self.round_index = int(round_index)
		self.budget = int(budget)
		self.N = int(N)
		self.K = int(K)
		self.J = int(J)
		self.J_lm = int(J_lm)
		self.J_mut = int(J_mut)
		self.J_erc = int(J_erc)
		self.T = int(T)
		self.cfg = LoopConfig(
			N=self.N,
			K=self.K,
			J=self.J,
			J_lm=self.J_lm,
			J_mut=self.J_mut,
			J_erc=self.J_erc,
			T=self.T,
			lm_budget_total=self.budget,
			eps_frontier=0.0,
		)

	def build(
		self,
		lm_pool: list[str],
		mut_seed_pool: list[str],
		erc_seed: int = 7,
		mut_seed: int = 1729,
	) -> PopulationScheduler:
		"""Construct a PopulationScheduler wired with deterministic components."""
		man = LMDeterminismManifest.build(
			model_id="llama",
			tokenizer_hash="tok",
			
			decoding={
				"temperature": 0.7,
				"top_k": 50,
				"top_p": 0.9,
				"repetition_penalty": 1.1,
				"max_tokens": 128
			},	
			run_id=self.run_id,
			round_index=self.round_index,
		)

		V = ExpressionValidator({"x": __import__("implementation").units.dim.L,
								 "y": __import__("implementation").units.dim.L,
								 "t": __import__("implementation").units.dim.DIMLESS})

		S = ScoreComputer(lambda e: 0.5, lambda E: 1.0 - 0.5 * float(E))

		lm = LMProposer(man, lm_pool)
		mut = MutationProposer(mut_seed_pool, seed=mut_seed)
		erc = ERCProposer(seed=erc_seed)

		return PopulationScheduler(self.cfg, man, V, lm, mut, erc, S)

