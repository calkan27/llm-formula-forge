"""Population experiment runner: drives a scheduler for multiple rounds,
collects simple metrics (memory growth per round, duplicate hit rate, wall-time),
and relies on the robust equivalence/deduplication already implemented in the
production modules.
"""
from __future__ import annotations
from typing import Dict, List
import time
from implementation.feature_selection.dedup import PreScoreDeduper
from implementation.population.equivalence import EquivalenceOrder
from implementation.population.scheduler_factory import SchedulerFactory

class PopulationExperiment:
	def __init__(self) -> None:
		"""Initialize the experiment helper with an equivalence checker."""
		self.eq = EquivalenceOrder()

	def run(self,
			factory: SchedulerFactory,
			seeds: List[str],
			lm_pool: List[str],
			rounds: int = 100) -> Dict[str, float]:
		"""Run an experiment for a fixed number of rounds."""
	
		sch = factory.build(lm_pool, seeds)
		uniq0, _ = PreScoreDeduper.dedup(list(seeds))
		sch.seed(uniq0)
		t0 = time.time()
		dup_hits = 0
		total_proposed = 0
		mem_sizes: List[int] = []
		r = 0
		while r < rounds:
			before = len(sch.canonical_hashes)
			j_lm = min(sch.cfg.J_lm, max(0, sch.cfg.lm_budget_total - sch.state.lm_calls_used))
			j_mut = sch.cfg.J_mut
			j_erc = sch.cfg.J_erc
			total_proposed += j_lm + j_mut + j_erc
			sch.run_round()
			after = len(sch.canonical_hashes)
			if after <= before:
				dup_hits += 1
			mem_sizes.append(after)
			r += 1
		t1 = time.time()
		mem_growth = 0.0
		if len(mem_sizes) > 1:
			mem_growth = float(mem_sizes[-1] - mem_sizes[0]) / float(len(mem_sizes) - 1)
		hit_rate = 0.0
		if rounds > 0:
			hit_rate = float(dup_hits) / float(rounds)
		wall = float(t1 - t0)
		return {"memory_growth_per_round": mem_growth, "duplicate_hit_rate": hit_rate, "wall_time_sec": wall}

