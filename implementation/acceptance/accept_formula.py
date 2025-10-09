"""Acceptance harness (class-based).

Provides:
  • AcceptFormula.accept(f, g, variables, target_dim=DIMLESS, out_dir=None,
						 float_grid_n=1024, float_tol=1e-12, metrics_n=512,
						 enforce_units=True) -> dict
  • Symbolic path attaches {"certificate":"simplify(f-g)==0"} in proof
  • Lattice is refutation-only via EqualityChecks.rational_grid_counterexample(...)
  • Float path attaches probabilistic bound
  • When enforce_units=False, unit typing is advisory: no hard reject; inferred dims are recorded.
"""

from __future__ import annotations
from pathlib import Path
from typing import Callable, Dict, List, Tuple
import sympy as sp

from implementation.complexity import Complexity
from implementation.units.system import UnitsChecker
from implementation.units.dim import Dim7, DIMLESS

from .equality import EqualityChecks
from .float_probe import FloatProbe
from .metrics import MetricsComputer
from .persist import ProofPersister


class AcceptFormula:
	"""Acceptance harness that certifies equality of two expressions and reports (C, E, S, L)."""

	def __init__(self, S_of: Callable[[float], float] | None = None) -> None:
		"""Construct the acceptance harness and bind complexity, equality, probe, metrics, and persistence helpers."""
		self._C = Complexity()
		self._S_of = S_of
		self._eq = EqualityChecks()
		self._probe = FloatProbe()
		self._metrics = MetricsComputer(self._C, self._S_of)
		self._persist = ProofPersister()

	def _canonical(self, e: sp.Expr) -> sp.Expr:
		"""Return the canonical normal form of a SymPy expression using the project’s Complexity canonicalizer."""
		return self._C.canonical(e)

	def _units(self, variables: Tuple[str, ...]) -> UnitsChecker:
		"""Build a UnitsChecker environment; single-variable problems default that variable to DIMLESS."""
		if len(variables) == 1:
			return UnitsChecker({variables[0]: DIMLESS})
		else:
			return UnitsChecker()

	def _syms_sorted(self, variables: Tuple[str, ...]) -> Tuple[sp.Symbol, ...]:
		"""Create SymPy symbols from a variable tuple, preserving order and returning a deterministic tuple."""
		out: List[sp.Symbol] = []
		for v in variables:
			out.append(sp.Symbol(v))
		return tuple(out)

	def _check_dim(self, U: UnitsChecker, expr: sp.Expr, target: Dim7) -> tuple[bool, Dim7, str]:
		"""Run a dimension check that never raises: (ok, inferred_dim, message) derived from UnitsChecker."""
		s = str(self._canonical(expr))
		try:
			ok, d, msg = U.check_expr(s, target=target)
			return bool(ok), d, str(msg)
		except Exception as e:
			try:
				d = U.infer(U.sympify_expr(s))
			except Exception:
				d = DIMLESS
			return False, d, f"type_error:{e}"

	def _decide_equality(self, f_can: sp.Expr, g_can: sp.Expr, variables: Tuple[str, ...],	float_grid_n: int,
							 float_tol: float, metrics_n: int) -> tuple[bool, str, Dict[str, float], Dict[str, object]]:
		"""Apply the symbolic → lattice-refutation → float-probe cascade; return (accepted, method, metrics, proof)."""
		if self._eq.symbolic_equal(f_can, g_can):
			metrics = self._metrics.compute(truth=f_can, cand=g_can, variables=variables, n=int(metrics_n), force_E=0.0)
			return True, "symbolic", metrics, {"certificate": "simplify(f-g)==0"}

		syms = self._syms_sorted(variables)
		w = self._eq.rational_grid_counterexample(f_can, g_can, syms)
		if w is not None:
			metrics = self._metrics.compute(truth=f_can, cand=g_can, variables=variables, n=int(metrics_n))
			proof = {"grid": "rational", "point": dict(w["point"]), "f_val": str(w["f_val"]), "g_val": str(w["g_val"]), "diff": str(w["diff"])}
			return False, "reject", metrics, {"reason": "counterexample_rational", **proof}

		ok_float, max_abs, over, m = self._probe.probe(f_can, g_can, variables, int(float_grid_n), float(float_tol))
		if ok_float and max_abs <= float(float_tol):
			delta_up, p_up = self._probe.prob_bound(over=over, m=m, alpha=0.05)
			proof = {
				"grid": "float",
				"n": int(float_grid_n),
				"m": int(m),
				"over": int(over),
				"max_abs_diff": float(max_abs),
				"tol": float(float_tol),
				"bound_form": "exp(-delta * m)",
				"delta_upper_95": float(delta_up),
				"p_miss_upper_95": float(p_up),
			}
			metrics = self._metrics.compute(truth=f_can, cand=g_can, variables=variables, n=int(metrics_n))
			return True, "float_probe", metrics, proof

		metrics = self._metrics.compute(truth=f_can, cand=g_can, variables=variables, n=int(metrics_n))
		return False, "reject", metrics, {}

	def _finalize(self,
				  accepted: bool,
				  method: str,
				  metrics: Dict[str, float],
				  f_can: sp.Expr,
				  g_can: sp.Expr,
				  proof: Dict[str, object],
				  out_dir: Path | None,
				  f_dim_str: str | None = None,
				  g_dim_str: str | None = None) -> Dict[str, object]:
		"""Assemble the final result payload, attach proof and dims, persist if requested, and return the dict."""
		out: Dict[str, object] = {
			"accepted": bool(accepted),
			"method": method,
			"C": int(metrics["C"]),
			"E": float(metrics["E"]),
			"S": float(metrics["S"]),
			"L": float(metrics["L"]),
			"f_can": str(f_can),
			"g_can": str(g_can),
		}
		if f_dim_str is not None:
			out["f_dim"] = str(f_dim_str)
		if g_dim_str is not None:
			out["g_dim"] = str(g_dim_str)
		if method == "float_probe":
			out["proof"] = dict(proof)
		else:
			if method == "symbolic":
				out["proof"] = dict(proof)
			else:
				if method == "reject":
					if proof:
						out["proof"] = dict(proof)
		if not accepted:
			if "reason" in proof:
				out["reason"] = str(proof["reason"])
			else:
				out["reason"] = "not_equal"
		if out_dir is not None:
			self._persist.persist(Path(out_dir), dict(out))
		return out

	def accept(
		self,
		f: sp.Expr,
		g: sp.Expr,
		variables: Tuple[str, ...],
		target_dim: Dim7 = DIMLESS,
		out_dir: Path | None = None,
		float_grid_n: int = 1024,
		float_tol: float = 1e-12,
		metrics_n: int = 512,
		enforce_units: bool = True,
	) -> Dict[str, object]:
		"""
		Run the full acceptance pipeline and return decision, method, metrics, canonical forms, and proof.
		"""
		f_can = self._canonical(f)
		g_can = self._canonical(g)
		U = self._units(variables)

		ok_f, df, _ = self._check_dim(U, f_can, target_dim)
		ok_g, dg, _ = self._check_dim(U, g_can, target_dim)

		f_dim_str = U.pretty_dim(df)
		g_dim_str = U.pretty_dim(dg) if dg is not None else None

		if enforce_units:
			if not ok_f:
				return self._finalize(False, "reject",
									  {"C": 0.0, "E": 1.0, "S": 1e-15, "L": 0.0},
									  f_can, g_can,
									  {"reason": "dimension_mismatch"},
									  out_dir,
									  f_dim_str=f_dim_str, g_dim_str=g_dim_str)
			if not ok_g:
				return self._finalize(False, "reject",
									  {"C": 0.0, "E": 1.0, "S": 1e-15, "L": 0.0},
									  f_can, g_can,
									  {"reason": "dimension_mismatch"},
									  out_dir,
									  f_dim_str=f_dim_str, g_dim_str=g_dim_str)
		else:
			pass

		accepted, method, metrics, proof = self._decide_equality(
			f_can, g_can, variables, int(float_grid_n), float(float_tol), int(metrics_n)
		)
		return self._finalize(accepted, method, metrics, f_can, g_can, proof, out_dir, f_dim_str=f_dim_str, g_dim_str=g_dim_str)

