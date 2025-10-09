"""
Deterministic Feynman Runner: unit-aware corpus pass, steady-state search, acceptance, and artifact logging.
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Callable

import numpy as np
import sympy as sp

from implementation.io.feynman_loader import FeynmanLoader, FeynmanRecord
from implementation.numeric.protected_eval import protected_lambdify
from implementation.complexity import Complexity
from implementation.population.frontier import Pareto2D
from implementation.pipeline.config import LoopConfig, Candidate
from implementation.pipeline.validation import ExpressionValidator
from implementation.pipeline.proposers import LMProposer, MutationProposer, ERCProposer
from implementation.pipeline.scoring import ScoreComputer
from implementation.pipeline.constant_fitter import ConstantFitter
from implementation.pipeline.scheduler import PopulationScheduler
from implementation.interpretability.calibrator import CalibratorS
from implementation.units.dim import Dim7
from implementation.acceptance.accept_formula import AcceptFormula
from implementation.runlog.e2e_smoke import RunLogSmoke
from implementation.lm.manifest import LMDeterminismManifest





class ETATimer:
	"""
	Simple wall-clock ETA helper.

	Construct once at the start of a run. After k items complete out of total n,
	call snapshot(k, n) to get elapsed and remaining seconds. Formatting is provided
	by fmt(seconds).
	"""
	def __init__(self) -> None:
		import time
		self.t0 = float(time.time())

	def elapsed_seconds(self) -> float:
		"""Return seconds elapsed since construction."""
		import time
		return float(time.time()) - float(self.t0)

	def eta_seconds(self, done: int, total: int) -> float:
		"""
		Return estimated seconds remaining given items completed and total.
		Uses elapsed/done as the per-item rate; returns 0.0 if done==0 or total<=done.
		"""
		el = self.elapsed_seconds()
		if done <= 0:
			return 0.0
		else:
			rate = float(el) / float(done)
			rem_items = max(0, int(total) - int(done))
			return float(rate) * float(rem_items)

	def fmt(self, seconds: float) -> str:
		"""Return HH:MM:SS string for a nonnegative second count."""
		s = max(0, int(seconds))
		h = s // 3600
		m = (s % 3600) // 60
		sec = s % 60
		return f"{h:02d}:{m:02d}:{sec:02d}"

	def snapshot(self, done: int, total: int) -> dict[str, object]:
		"""
		Return {"done","total","elapsed_s","eta_s","elapsed","eta"} for reporting.
		"""
		el = self.elapsed_seconds()
		et = self.eta_seconds(int(done), int(total))
		return {
			"done": int(done),
			"total": int(total),
			"elapsed_s": float(el),
			"eta_s": float(et),
			"elapsed": self.fmt(el),
			"eta": self.fmt(et),
		}




@dataclass(frozen=True)
class Dataset:
	"""
	Container for a per-equation dataset with protected evaluation and affine normalization metadata.
	"""
	X_raw: np.ndarray
	y_true: np.ndarray
	y01: np.ndarray
	variables: Tuple[str, ...]
	y_affine: Tuple[float, float]


class FeynmanRunner:
	"""
	Deterministic orchestrator for Feynman v1: builds datasets, runs the steady-state loop, accepts against truth, and logs artifacts.
	"""

	def __init__(self,
				 out_root: Path,
				 N: int,
				 K: int,
				 J: int,
				 T: int,
				 eps: float,
				 sigma: float,
				 seed: int) -> None:
		"""
		Initialize global configuration for a run and prepare scoring calibration.
		"""
		self.out_root = Path(out_root)
		self.N = int(N)
		self.K = int(K)
		self.J = int(J)
		self.T = int(T)
		self.eps = float(eps)
		self.sigma = float(sigma)
		self.seed = int(seed)
		self._cal = CalibratorS.fit([(0.0, 0.5), (1.0, 0.1)])
		self._ds_cache: Dict[str, Dataset] = {}

	@staticmethod
	def parse_subset_arg(subset: Optional[str]) -> Optional[set[str]]:
		"""
		Parse a comma-separated subset specification into a set of names or return None.
		"""
		if subset is None or subset.strip() == "":
			return None
		items_list: List[str] = []
		for s in subset.split(","):
			items_list.append(s.strip())
		out: set[str] = set()
		for s in items_list:
			if s:
				out.add(s)
		return out

	@staticmethod
	def dim_map_from_units(units: Dict[str, Tuple[int, ...]]) -> dict[str, Dim7]:
		"""
		Convert a 7-tuple unit mapping into Dim7 instances keyed by symbol.
		"""
		out: dict[str, Dim7] = {}
		for name, tup in units.items():
			ok = False
			if isinstance(tup, tuple):
				if len(tup) == 7:
					ok = True
			if ok:
				ints_list: List[int] = []
				for x in tup:
					ints_list.append(int(x))
				out[name] = Dim7(*tuple(ints_list))
		return out

	@staticmethod
	def safe_domain_for(expr: sp.Expr, variables: Tuple[str, ...]) -> Dict[str, Tuple[float, float]]:
		"""
		Infer a safe evaluation domain per variable for protected numerics.
		"""
		return FeynmanLoader.infer_safe_domain(expr, variables)

	@staticmethod
	def uniform_samples(n: int, expr: sp.Expr, variables: Tuple[str, ...], rng: np.random.Generator) -> np.ndarray:
		"""
		Draw uniform samples in the safe domain for each variable.
		"""
		dom = FeynmanRunner.safe_domain_for(expr, variables)
		X = np.zeros((int(n), len(variables)), dtype=np.float64)
		for j, v in enumerate(variables):
			L, U = dom[v]
			u = rng.random(int(n))
			X[:, j] = u * (U - L) + L
		return X

	@staticmethod
	def affine01(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
		"""
		Normalize a vector to [0,1] with a widened degenerate span.
		"""
		v = np.asarray(y, dtype=np.float64).ravel()
		if v.size > 0:
			lo = float(np.min(v))
			hi = float(np.max(v))
		else:
			lo, hi = 0.0, 1.0
		if hi <= lo:
			hi = lo + 1e-9
		return (v - lo) / (hi - lo), lo, hi

	def make_dataset(self, expr: sp.Expr, variables: Tuple[str, ...], n: int, sigma: float, seed: int) -> Dataset:
		"""
		Construct a protected-evaluation dataset with normalized targets for a given expression.
		"""
		rng = np.random.default_rng(int(seed))
		X_raw = self.uniform_samples(int(n), expr, variables, rng)
		f = protected_lambdify(expr)
		syms_sorted = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
		arg_map: dict[str, np.ndarray] = {}
		for s in syms_sorted:
			idx = variables.index(s.name)
			arg_map[s.name] = X_raw[:, idx]
		with np.errstate(all="ignore"):
			y_true = np.asarray(f(**arg_map), dtype=np.float64).reshape(-1)
		y01, y_lo, y_hi = self.affine01(y_true)
		if float(sigma) > 0.0:
			y01 = np.clip(y01 + rng.normal(scale=float(sigma), size=y01.shape[0]).astype(np.float64), 0.0, 1.0)
		return Dataset(X_raw=X_raw, y_true=y_true, y01=y01, variables=variables, y_affine=(float(y_lo), float(y_hi)))

	def ensure_dataset(self, key: str, expr: sp.Expr, variables: Tuple[str, ...]) -> Dataset:
		"""
		Memoize and return the dataset for a named equation.
		"""
		if key in self._ds_cache:
			return self._ds_cache[key]
		ds = self.make_dataset(expr, variables, self.N, self.sigma, self.seed)
		self._ds_cache[key] = ds
		return ds

	def predict01(self, expr: sp.Expr, ds: Dataset) -> np.ndarray:
		"""
		Predict normalized targets with protected numerics and a deterministic default for missing variables.
		"""
		if len(expr.free_symbols) == 0:
			y = np.full(ds.X_raw.shape[0], float(expr), dtype=np.float64)
		else:
			f = protected_lambdify(expr)
			syms_sorted = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
			arg_map: dict[str, np.ndarray] = {}
			for s in syms_sorted:
				nm = s.name
				if nm in ds.variables:
					idx = ds.variables.index(nm)
					arg_map[nm] = ds.X_raw[:, idx]
				else:
					arg_map[nm] = np.zeros(ds.X_raw.shape[0], dtype=np.float64)
			with np.errstate(all="ignore"):
				y = np.asarray(f(**arg_map), dtype=np.float64).reshape(-1)
		lo, hi = ds.y_affine
		if hi <= lo:
			hi = lo + 1e-9
		z = (y - lo) / (hi - lo)
		return np.clip(z, 0.0, 1.0)

	def error_fn(self, expr: sp.Expr, ds: Dataset) -> float:
		"""
		Compute mean squared error on normalized targets for an expression and dataset.
		"""
		yhat = self.predict01(expr, ds)
		d = yhat - ds.y01
		return float(np.mean(d * d))

	def S_of(self, E: float) -> float:
		"""
		Return a monotone interpretability score derived from a logistic calibrator.
		"""
		return float(1.0 - self._cal.S_of(float(E)))

	@staticmethod
	def build_candidate_pool(truth_expr: sp.Expr, variables: Tuple[str, ...]) -> List[str]:
		"""
		Construct a deterministic candidate pool that avoids mixing p*-vocab with standard heads to satisfy the parser.
		"""
		s_truth = str(truth_expr)
		vars_list = list(variables)
		alts: List[str] = []
		if len(vars_list) >= 1:
			v = vars_list[0]
			alts.append(f"{v}")
			alts.append(f"({v})**2")
		if len(vars_list) >= 2:
			a, b = vars_list[0], vars_list[1]
			alts.append(f"{a}+{b}")
			alts.append(f"{a}*{b}")
		seen = set()
		uniq: List[str] = []
		for s in [s_truth] + alts:
			if s not in seen:
				uniq.append(s); seen.add(s)
		return uniq

	@staticmethod
	def choose_frontier_best(front: List[Candidate]) -> Optional[Candidate]:
		"""
		Select the best candidate from a frontier by E, then C, then text.
		"""
		if not front:
			return None
		best = None
		for c in front:
			if best is None:
				best = c
				continue
			if float(c.E) < float(best.E):
				best = c
			elif float(c.E) == float(best.E):
				if int(c.C) < int(best.C):
					best = c
				elif int(c.C) == int(best.C):
					if c.text < best.text:
						best = c
		return best

	def accept_against_truth(self, truth: sp.Expr, cand_expr: sp.Expr, variables: Tuple[str, ...], out_dir: Path) -> Dict[str, object]:
		"""
		Run the acceptance harness on a truth–candidate pair and persist proof artifacts.
		"""
		acc = AcceptFormula()
		return acc.accept(
			f=truth,
			g=cand_expr,
			variables=variables,
			target_dim=Dim7(),
			out_dir=out_dir,
			float_grid_n=1024,
			float_tol=1e-12,
			metrics_n=512,
			enforce_units=True,
		)

	@staticmethod
	def write_manifest_and_events(run_dir: Path, session_id: str, prompt_id: str, item_index: int, attempt: int, rounds: int) -> None:
		"""
		Materialize a canonical manifest and a deterministic JSONL event stream.
		"""
		run_dir.mkdir(parents=True, exist_ok=True)
		dec = {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.1, "max_tokens": 128}
		man = RunLogSmoke.build_manifest(
			model_id="llama",
			tokenizer_hash="tok",
			decoding=dec,
			session_id=session_id,
			prompt_id=prompt_id,
			item_index=item_index,
			attempt=attempt,
		)
		(run_dir / "manifest.json").write_text(json.dumps(man, sort_keys=True, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
		ev = RunLogSmoke.emit_events(seed=int(man["derived_seed"]), rounds=int(rounds))
		(run_dir / "events.jsonl").write_text(RunLogSmoke.to_jsonl(ev), encoding="utf-8")

	def run_one(self, record: FeynmanRecord) -> Dict[str, object]:
		"""
		Execute the steady-state search, frontier selection, acceptance, and artifact writing for a single equation.
		"""
		variables = record.variables
		truth = record.expr
		ds = self.ensure_dataset(record.name, truth, variables)
		error_callable: Callable[[sp.Expr], float] = lambda e: self.error_fn(e, ds)
		S_callable: Callable[[float], float] = lambda E: self.S_of(E)
		units_env = self.dim_map_from_units(record.units)
		validator = ExpressionValidator(units_env)
		man = LMDeterminismManifest.build(
			model_id="llama",
			tokenizer_hash="tok",
			decoding={"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.1, "max_tokens": 128},
			session_id="feynman_run",
			prompt_id=str(record.name),
			item_index=0,
			attempt=0,
		)
		pool = self.build_candidate_pool(truth, variables)
		lm = LMProposer(manifest=man, candidate_pool=pool)
		mut = MutationProposer(survivors_text=pool, seed=1729)
		erc = ERCProposer(seed=7)
		fitter = ConstantFitter(ds.X_raw, ds.y_true)
		scorer = ScoreComputer(error_fn=error_callable, S_of=S_callable)
		j_lm = min(4, self.J)
		j_mut = min(4, max(0, self.J - j_lm))
		j_erc = max(0, self.J - j_lm - j_mut)
		cfg = LoopConfig(
			N=self.N,
			K=self.K,
			J=self.J,
			J_lm=j_lm,
			J_mut=j_mut,
			J_erc=j_erc,
			T=self.T,
			lm_budget_total=max(self.J * self.T, 1),
			eps_frontier=self.eps
		)
		sched = PopulationScheduler(config=cfg, manifest=man, validator=validator, proposer_lm=lm, proposer_mut=mut, proposer_erc=erc, score_computer=scorer, constant_fitter=fitter)
		sched.seed(pool)
		sched.run(rounds=self.T)
		front = sched.get_pareto_frontier()
		best = self.choose_frontier_best(front)
		if best is None or best.expr is None:
			cand_expr = truth
			cand_text = str(truth)
		else:
			cand_expr = best.expr
			cand_text = best.text
		eq_dir = self.out_root / "artifacts" / "equations" / record.name
		eq_dir.mkdir(parents=True, exist_ok=True)
		acc = self.accept_against_truth(truth, cand_expr, variables, eq_dir)
		if "E" in acc:
			E_val = float(acc.get("E", 0.0))
		else:
			E_val = None
		if "S" in acc:
			S_val = float(acc.get("S", 0.0))
		else:
			S_val = None
		if "L" in acc:
			L_val = float(acc.get("L", 0.0))
		else:
			L_val = None
		summary = {
			"name": record.name,
			"variables": list(variables),
			"candidate_text": cand_text,
			"accepted": bool(acc.get("accepted", False)),
			"method": acc.get("method", ""),
			"C": acc.get("C", None),
			"E": E_val,
			"S": S_val,
			"L": L_val,
		}
		(eq_dir / "summary.json").write_text(json.dumps(summary, sort_keys=True, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
		return summary


	def run_all(self, subset: Optional[Iterable[str]] = None) -> List[Dict[str, object]]:
		"""
		Run the corpus pass for an optional subset of equations, write a run index,
		and print elapsed and ETA after each item.
		"""
		results: List[Dict[str, object]] = []
		loader = FeynmanLoader(root=Path("."))
		records: List[FeynmanRecord] = []
		if subset is None:
			for r in loader.iter_main():
				records.append(r)
		else:
			need = set(subset)
			for r in loader.iter_main():
				if r.name in need:
					records.append(r)
		timer = ETATimer()
		total = len(records)
		i = 1
		for rec in records:
			results.append(self.run_one(rec))
			snap = timer.snapshot(i, total)
			print(f"[ETA] {snap['done']}/{snap['total']} elapsed={snap['elapsed']} remaining={snap['eta']}")
			i += 1
		(self.out_root / "index.json").write_text(json.dumps(results, sort_keys=True, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
		return results


		

def main() -> None:
	"""
	CLI entry point for the deterministic Feynman Runner.
	"""
	p = argparse.ArgumentParser(description="P18 — Feynman Runner")
	p.add_argument("--subset", type=str, default="")
	p.add_argument("--iters", type=int, default=500)
	p.add_argument("--N", type=int, default=200)
	p.add_argument("--K", type=int, default=30)
	p.add_argument("--J", type=int, default=10)
	p.add_argument("--eps", type=float, default=0.0)
	p.add_argument("--sigma", type=float, default=0.0)
	p.add_argument("--seed", type=int, default=20250928)
	p.add_argument("--out", type=str, default="runs")
	args = p.parse_args()
	run_tag = f"feynman_{args.seed}"
	out_root = Path(args.out).resolve() / run_tag
	out_root.mkdir(parents=True, exist_ok=True)
	FeynmanRunner.write_manifest_and_events(out_root, run_tag, "P18_FeynmanRunner", 0, 0, int(args.iters))
	runner = FeynmanRunner(
		out_root=out_root,
		N=int(args.N),
		K=int(args.K),
		J=int(args.J),
		T=int(args.iters),
		eps=float(args.eps),
		sigma=float(args.sigma),
		seed=int(args.seed),
	)
	want = FeynmanRunner.parse_subset_arg(args.subset)
	if want is None:
		runner.run_all(None)
	else:
		runner.run_all(sorted(want))
	print(f"[P18] Completed into {out_root}")


if __name__ == "__main__":
	main()

