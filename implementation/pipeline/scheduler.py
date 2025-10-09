from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Set
import time
import hashlib

import numpy as np
import sympy as sp

from implementation.lm.manifest import LMDeterminismManifest
from implementation.feature_selection.dedup import PreScoreDeduper
from implementation.pipeline.config import LoopConfig, LogEvent, LoopState, Candidate
from implementation.pipeline.validation import ExpressionValidator
from implementation.pipeline.proposers import ProposalBatch, Proposer
from implementation.pipeline.scoring import ScoreComputer
from implementation.pipeline.constant_fitter import ConstantFitter


class PopulationScheduler:
	"""
	LLaMA-aware steady-state with budget, dedup, Pareto frontier,
	constant fitting, and loss-based selection.
	"""

	def __init__(
		self,
		config: LoopConfig,
		manifest: LMDeterminismManifest,
		validator: ExpressionValidator,
		proposer_lm: Proposer,
		proposer_mut: Proposer,
		proposer_erc: Proposer,
		score_computer: ScoreComputer,
		constant_fitter: Optional[ConstantFitter] = None,
	) -> None:
		"""
		Construct the scheduler.

		Guard policy (refined):
		  • Only raise at construction when every proposal in every round is LM
			(J_lm == J) **and** the overall plan exceeds the budget by more than
			one full LM round. This still catches hopeless configurations while
			allowing runs that can stop cleanly before the last round.
		  • Otherwise allow construction and stop early during run() when the
			budget is actually exhausted (or other stop conditions fire).
		"""
		self.cfg = config

		if self.cfg.J_lm == self.cfg.J:
			excess = self.cfg.T * self.cfg.J_lm - self.cfg.lm_budget_total
			if excess > self.cfg.J_lm:
				raise ValueError("LM budget violation: T * J_lm > lm_budget_total")

		self.manifest = manifest
		self.val = validator
		self.lm = proposer_lm
		self.mut = proposer_mut
		self.erc = proposer_erc
		self.score = score_computer
		self.fitter = constant_fitter
		self.state = LoopState()

		self.pareto_frontier: List[Candidate] = []
		self.frontier_stagnation_rounds = 0
		self.max_stagnation_rounds = 50  

		self.canonical_hashes: Dict[str, str] = {}



	@staticmethod
	def _stable_key_for_expr(expr: sp.Expr) -> str:
		return sp.srepr(expr)
	
	@staticmethod
	def _is_ascii(text: str) -> bool:
		"""Check if text is ASCII-only."""
		if not isinstance(text, str):
			print("_is_ascii: non-string input; returning False")
			return False

		if text.isascii():
			return True
		else:
			print("_is_ascii: non-ASCII characters detected; returning False")
			return False


	def _compute_canonical_hash(self, text: str) -> str:
		"""Compute and cache canonical form hash for logging."""
		if not isinstance(text, str):
			print("_compute_canonical_hash: non-string input; returning 'error'")
			return "error"

		s = text.strip()
		if s == "":
			print("_compute_canonical_hash: empty string; returning 'error'")
			return "error"

		key = PreScoreDeduper.canonical_struct_key(s)
		if not isinstance(key, str) or key == "":
			print("_compute_canonical_hash: invalid canonical key; returning 'error'")
			return "error"

		fp = PreScoreDeduper.numeric_fingerprint(s)
		if not isinstance(fp, str) or fp == "":
			print("_compute_canonical_hash: invalid fingerprint; returning 'error'")
			return "error"

		combined = f"{key}|{fp}"
		if combined == "":
			print("_compute_canonical_hash: invalid combined payload; returning 'error'")
			return "error"

		h = hashlib.sha256(combined.encode("utf-8")).hexdigest()
		if not isinstance(h, str):
			print("_compute_canonical_hash: non-string digest; returning 'error'")
			return "error"
		if len(h) < 16:
			print("_compute_canonical_hash: hexdigest too short; returning 'error'")
			return "error"

		return h[:16]



	def _pareto_sort_key(self, cand: Candidate) -> tuple:
		"""Deterministic sort key (C, E, srepr)."""
		expr = getattr(cand, "expr", None)
		if expr is not None:
			skey = self._stable_key_for_expr(expr)
		else:
			skey = ""

		c_raw = getattr(cand, "C", None)
		if isinstance(c_raw, (int, float)):
			c_val = float(c_raw)
		else:
			c_val = float("inf")

		e_raw = getattr(cand, "E", None)
		if isinstance(e_raw, (int, float)):
			e_val = float(e_raw)
		else:
			e_val = float("inf")

		return (c_val, e_val, skey)

	def _update_pareto_frontier(self, candidates: List[Candidate]) -> bool:
		"""
		Update the ε-Pareto frontier in (C, E) space using the simple Jensen sweep
		specified in the paper: stably sort by (C, E) and accept iff E < E* - ε.
		"""
		if not candidates:
			return False

		valid: List[Candidate] = []
		for c in candidates:
			is_ok = bool(getattr(c, "ok", False))
			c_has = getattr(c, "C", None)
			e_has = getattr(c, "E", None)
			if is_ok and c_has is not None and e_has is not None:
				if isinstance(c_has, (int, float)) and isinstance(e_has, (int, float)):
					c_num = float(c_has); e_num = float(e_has)
					if np.isfinite(c_num) and np.isfinite(e_num):
						valid.append(c)

		if not valid:
			return False

		eps_raw = getattr(self.cfg, "eps_frontier", 0.0)
		eps = float(eps_raw) if isinstance(eps_raw, (int, float)) else 0.0

		reps = list(valid)
		reps.sort(key=self._pareto_sort_key)

		new_frontier: List[Candidate] = []
		best_E = float("inf")
		for c in reps:
			ce = float(c.E)
			if ce < best_E - eps:
				new_frontier.append(c)
				best_E = ce

		if len(new_frontier) != len(self.pareto_frontier):
			frontier_changed = True
		else:
			if new_frontier:
				old_set = {(float(c.C), float(c.E), c.text) for c in self.pareto_frontier}
				new_set = {(float(c.C), float(c.E), c.text) for c in new_frontier}
				frontier_changed = old_set != new_set
			else:
				frontier_changed = False

		self.pareto_frontier = new_frontier
		return frontier_changed

	def _dedup(self, texts: List[str]) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
		"""Extended dedup with symbolic certificate."""
		return PreScoreDeduper.dedup(texts)




	def _validate_and_build(self, batch: ProposalBatch) -> List[Candidate]:
		"""Validate with ASCII check."""
		out: List[Candidate] = []
		for s in batch.texts:
			if not self._is_ascii(s):
				out.append(Candidate(s, None, False, batch.provenance, "non_ascii"))
				self.state.log.append(LogEvent("hygiene_reject", {"text": s, "reason": "non_ascii"}))
				continue
			pr = self.val.parse(s)
			if not pr.ok:
				out.append(Candidate(s, None, False, batch.provenance, pr.message))
				continue
			tr = self.val.typecheck(pr.expr)
			if tr.ok:
				out.append(Candidate(s, pr.expr, True, batch.provenance, "ok"))
			else:
				if batch.provenance == "seed":
					out.append(Candidate(s, pr.expr, True, batch.provenance, tr.message))
				else:
					out.append(Candidate(s, None, False, batch.provenance, tr.message))
		return out


	def _fit_and_score(self, cs: List[Candidate]) -> List[Candidate]:
		"""
		Fit constants then score. 
		"""
		valid = [c for c in cs if c.ok and c.expr is not None]
		if not valid:
			return cs
		
		out: List[Candidate] = []
		for c in cs:
			if not c.ok or c.expr is None:
				out.append(c)
				continue
			
			if self.fitter is not None:
				fitted_expr = self.fitter.fit(c.expr)
			else:
				fitted_expr = c.expr
			
			scores = self.score.score_many([fitted_expr])
			E, Cval, Sval, Lval = scores[0]
			
			out.append(Candidate(
				c.text, fitted_expr, True, c.provenance, "ok",
				C=Cval, E=E, S=Sval, L=Lval
			))
		
		return out

	def _score_valid(self, cs: List[Candidate]) -> List[Candidate]:
		"""Backward compatibility wrapper."""
		return self._fit_and_score(cs)


	def _survivor_sort_key(self, cand: Candidate) -> tuple:
		"""Deterministic survivor sort key (L asc, srepr asc)."""
		L_raw = getattr(cand, "L", None)
		if isinstance(L_raw, (int, float)):
			L_val = float(L_raw)
		else:
			L_val = float("inf")

		expr = getattr(cand, "expr", None)
		if expr is not None:
			skey = self._stable_key_for_expr(expr)
		else:
			skey = ""

		return (L_val, skey)


	def _choose_survivors(self, items: List[Candidate]) -> List[Candidate]:
		"""Pick up to K candidates by increasing L, with stable srepr tie-break."""
		ok_scored: List[Candidate] = []
		for c in items:
			has_all_scores = (
				bool(getattr(c, "ok", False)) and
				(getattr(c, "L", None) is not None) and
				(getattr(c, "C", None) is not None) and
				(getattr(c, "E", None) is not None) and
				(getattr(c, "S", None) is not None)
			)
			if has_all_scores:
				ok_scored.append(c)

		if not ok_scored:
			return []

		ok_scored.sort(key=self._survivor_sort_key)

		k_raw = getattr(self.cfg, "K", 0)
		if isinstance(k_raw, int):
			k_val = int(k_raw)
		else:
			k_val = 0

		return ok_scored[: k_val]


	def seed(self, initial_texts: List[str]) -> None:
		"""Seed with logging of canonical forms and hashes (Theory Sec. 16)."""
		uniq, meta = self._dedup(list(initial_texts))
		
		for txt in uniq:
			h = self._compute_canonical_hash(txt)
			self.canonical_hashes[txt] = h
		
		cand = self._validate_and_build(ProposalBatch(uniq, "seed"))
		scored = self._fit_and_score(cand)
		self.state.survivors = self._choose_survivors(scored)
		
		self._update_pareto_frontier(scored)
		
		self.state.log.append(LogEvent("seed_init", {
			"count_in": len(initial_texts),
			"unique": len(uniq),
			"survivors": len(self.state.survivors),
			"pareto_size": len(self.pareto_frontier),
			"canonical_hashes": len(self.canonical_hashes),
		}))

	def run_round(self) -> None:
		"""
		Execute one scheduler round.
		"""
		r = self.state.round_idx

		j_lm = min(self.cfg.J_lm, max(0, self.cfg.lm_budget_total - self.state.lm_calls_used))
		j_mut = self.cfg.J_mut
		j_erc = self.cfg.J_erc

		total = j_lm + j_mut + j_erc
		if total < self.cfg.J:
			j_mut += (self.cfg.J - total)
		elif total > self.cfg.J:
			j_mut = max(0, j_mut - (total - self.cfg.J))

		p_lm = self.lm.propose(j_lm)
		p_mut = self.mut.propose(j_mut)
		p_erc = self.erc.propose(j_erc)
		self.state.lm_calls_used += len(p_lm.texts)

		texts_all = p_lm.texts + p_mut.texts + p_erc.texts
		uniq, _meta = self._dedup(texts_all)

		set_lm  = set(p_lm.texts)
		set_mut = set(p_mut.texts)
		set_erc = set(p_erc.texts)

		lm_kept:  List[str] = []
		mut_kept: List[str] = []
		erc_kept: List[str] = []

		for txt in uniq:
			if txt not in self.canonical_hashes:
				h = self._compute_canonical_hash(txt)
				self.canonical_hashes[txt] = h

			if txt in set_lm:
				lm_kept.append(txt)
			if txt in set_mut:
				mut_kept.append(txt)
			if txt in set_erc:
				erc_kept.append(txt)

		self.state.log.append(LogEvent("dedup", {
			"round": r,
			"in": len(texts_all),
			"unique": len(uniq),
		}))

		batches: List[ProposalBatch] = [
			ProposalBatch(lm_kept,  "lm"),
			ProposalBatch(mut_kept, "mut"),
			ProposalBatch(erc_kept, "erc"),
		]
		validated: List[Candidate] = []
		for b in batches:
			validated.extend(self._validate_and_build(b))
		scored = self._fit_and_score(validated)

		frontier_changed = self._update_pareto_frontier(scored)
		if not frontier_changed:
			self.frontier_stagnation_rounds += 1
		else:
			self.frontier_stagnation_rounds = 0

		merged: List[Candidate] = list(self.state.survivors)
		for c in scored:
			if c.ok:
				merged.append(c)
		next_surv = self._choose_survivors(merged)

		frontier_payload: List[tuple[int, float, str]] = []
		for c in self.pareto_frontier:
			frontier_payload.append((int(c.C), float(c.E), self._stable_key_for_expr(c.expr)))

		self.state.log.append(LogEvent("selection", {
			"round": r,
			"accepted": len(next_surv),
			"pareto_size": len(self.pareto_frontier),
			"stagnation_rounds": self.frontier_stagnation_rounds,
			"frontier": frontier_payload,
		}))

		self.state.survivors = next_surv
		self.state.round_idx += 1

	def run(self, rounds: Optional[int] = None, max_time_seconds: Optional[float] = None) -> LoopState:
		"""
		Run with multiple stopping conditions:
		- Budget exhaustion
		- Stagnation of the (C,E) frontier
		- Wall-clock time limit
		- Round limit
		"""
		T = self.cfg.T if rounds is None else int(rounds)
		start_time = time.time() if max_time_seconds else None
		
		for _ in range(T):
			
			if self.state.lm_calls_used >= self.cfg.lm_budget_total:
				self.state.log.append(LogEvent("stop", {"reason": "lm_budget_exhausted"}))
				break
			
			if self.frontier_stagnation_rounds >= self.max_stagnation_rounds:
				self.state.log.append(LogEvent("stop", {"reason": "frontier_stagnation"}))
				break
			
			if start_time and (time.time() - start_time) >= max_time_seconds:
				self.state.log.append(LogEvent("stop", {"reason": "wall_clock_limit"}))
				break
			
			self.run_round()
		
		self.state.log.append(LogEvent("final_pareto", {
			"size": len(self.pareto_frontier),
			"frontier": [(c.C, c.E, c.text[:50]) for c in self.pareto_frontier[:10]]  
		}))
		
		return self.state

	def get_pareto_frontier(self) -> List[Candidate]:
		"""Return the current ε-Pareto frontier in (C,E) space."""
		return list(self.pareto_frontier)

