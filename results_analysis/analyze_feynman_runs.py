"""
Analyze Feynman run artifacts for theory-aligned invariants and produce an aggregate JSON report and CSV table.

This script validates:
- Manifest integrity (required keys, canonical SHA-256 re-derivation).
- Event stream shape and monotone logical timestamps.
- Per-equation artifact triplets (accept_proof.json, summary.json, forms.txt) with cross-checks.
- Acceptance method and metric consistency, plus simple corpus summaries.

CLI:
	python analyze_feynman_run.py <runs/feynman_YYYYMMDD> --out <report_dir>
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import hashlib
import csv

METHOD_SET = {"symbolic", "float_probe", "reject"}


def canonical_json(o: dict) -> str:
	"""
	Return a canonical JSON string for a Python dict using sorted keys and tight separators.

	Parameters
	----------
	o : dict
		The dictionary to serialize.

	Returns
	-------
	str
		Canonical JSON representation.
	"""
	return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256_hex(b: bytes) -> str:
	"""
	Compute the SHA-256 hex digest for a bytes payload.

	Parameters
	----------
	b : bytes
		Input byte sequence.

	Returns
	-------
	str
		Hexadecimal digest string.
	"""
	h = hashlib.sha256()
	h.update(b)
	return h.hexdigest()


def load_json(path: Path) -> dict:
	"""
	Load a JSON file from disk.

	Parameters
	----------
	path : Path
		Filesystem path to JSON file.

	Returns
	-------
	dict
		Parsed JSON object.
	"""
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def load_jsonl(path: Path) -> List[dict]:
	"""
	Load a JSONL file as a list of dicts, skipping empty lines and validating per-line JSON.

	Parameters
	----------
	path : Path
		Filesystem path to JSONL file.

	Returns
	-------
	List[dict]
		Parsed list of JSON objects.

	Raises
	------
	ValueError
		If any line fails to parse as valid JSON.
	"""
	out: List[dict] = []
	with path.open("r", encoding="utf-8") as f:
		for i, line in enumerate(f, 1):
			s = line.strip()
			if not s:
				continue
			try:
				out.append(json.loads(s))
			except Exception as e:
				raise ValueError(f"Bad JSONL at {path}:{i}: {e}")
	return out


def validate_manifest(man_path: Path) -> Tuple[bool, Dict[str, Any]]:
	"""
	Validate a run manifest for required keys and verify its canonical SHA-256 hash.

	Parameters
	----------
	man_path : Path
		Path to manifest.json.

	Returns
	-------
	Tuple[bool, Dict[str, Any]]
		(ok flag, diagnostic notes including 'manifest_hash_ok' and any missing keys).
	"""
	man = load_json(man_path)
	required = [
		"model_id",
		"tokenizer_hash",
		"decoding",
		"derived_seed",
		"session_id",
		"prompt_id",
		"item_index",
		"attempt",
		"env",
		"manifest_hash",
	]
	missing = [k for k in required if k not in man]
	ok = len(missing) == 0
	notes: Dict[str, Any] = {}
	if not ok:
		notes["missing_keys"] = missing
	core = {k: man[k] for k in required if k != "manifest_hash"}
	recomputed = sha256_hex(canonical_json(core).encode("utf-8"))
	notes["manifest_hash_ok"] = (recomputed == man.get("manifest_hash"))
	if recomputed != man.get("manifest_hash"):
		notes["manifest_hash_expected"] = recomputed
		ok = False
	return ok, notes


def validate_events(ev_path: Path) -> Tuple[bool, Dict[str, Any]]:
	"""
	Validate the events.jsonl stream for shape, field types, and monotone timestamps.

	Parameters
	----------
	ev_path : Path
		Path to events.jsonl.

	Returns
	-------
	Tuple[bool, Dict[str, Any]]
		(ok flag, diagnostics including counts, start/stop presence, and monotonicity status).
	"""
	evs = load_jsonl(ev_path)
	ok = True
	notes: Dict[str, Any] = {"count": len(evs)}
	for i, e in enumerate(evs):
		if not isinstance(e, dict) or not all(k in e for k in ("ts", "kind", "payload")):
			ok = False
			notes.setdefault("bad_events", []).append({"index": i, "event": e, "reason": "shape"})
		else:
			if not isinstance(e["ts"], int) or not isinstance(e["kind"], str) or not isinstance(e["payload"], dict):
				ok = False
				notes.setdefault("bad_events", []).append({"index": i, "event": e, "reason": "types"})
	ts = [e.get("ts", -1) for e in evs]
	if ts and any(ts[i] > ts[i + 1] for i in range(len(ts) - 1)):
		ok = False
		notes["non_monotone_ts"] = True
	else:
		notes["non_monotone_ts"] = False
	kinds = [e.get("kind", "") for e in evs]
	notes["has_start"] = ("start" in kinds)
	notes["has_stop"] = ("stop" in kinds)
	return ok, notes


def validate_accept_proof(ap: dict) -> Tuple[bool, Dict[str, Any]]:
	"""
	Validate an accept_proof.json object for required fields and internal consistency.

	Parameters
	----------
	ap : dict
		Parsed accept_proof JSON.

	Returns
	-------
	Tuple[bool, Dict[str, Any]]
		(ok flag, diagnostics including missing keys and method-specific checks).
	"""
	ok = True
	notes: Dict[str, Any] = {}
	core_keys = ["accepted", "method", "C", "E", "S", "L", "f_can", "g_can"]
	missing = [k for k in core_keys if k not in ap]
	if missing:
		ok = False
		notes["missing_keys"] = missing
	m = ap.get("method", "")
	if m not in METHOD_SET:
		ok = False
		notes["bad_method"] = m
	if m == "float_probe":
		req = ["grid", "n", "m", "over", "max_abs_diff", "tol", "bound_form", "delta_upper_95", "p_miss_upper_95"]
		miss2 = [k for k in req if k not in ap.get("proof", ap)]
		if miss2:
			ok = False
			notes["missing_float_fields"] = miss2
		else:
			proof = ap.get("proof", ap)
			delta = float(proof["delta_upper_95"])
			mcount = int(proof["m"])
			p = float(proof["p_miss_upper_95"])
			expect = math.exp(-delta * mcount)
			if not (abs(p - expect) <= 1e-9 or (abs(p - expect) / max(1e-12, expect) < 1e-9)):
				ok = False
				notes["p_bound_mismatch"] = {"p": p, "expected": expect, "delta": delta, "m": mcount}
	if m == "reject":
		reason = ap.get("reason") or ap.get("proof", {}).get("reason")
		if reason is None:
			ok = False
			notes["reject_missing_reason"] = True
	return ok, notes


def validate_forms(forms_path: Path, f_can: str, g_can: str) -> Tuple[bool, Dict[str, Any]]:
	"""
	Validate that forms.txt begins with canonical forms for f and g on the first two lines.

	Parameters
	----------
	forms_path : Path
		Path to forms.txt.
	f_can : str
		Canonical form of truth.
	g_can : str
		Canonical form of candidate.

	Returns
	-------
	Tuple[bool, Dict[str, Any]]
		(ok flag, diagnostics including line count).
	"""
	text = forms_path.read_text(encoding="utf-8")
	lines = text.splitlines()
	ok = len(lines) >= 2 and lines[0].strip() == str(f_can) and lines[1].strip() == str(g_can)
	return ok, {"lines": len(lines)}


def analyze_equation_dir(eq_dir: Path) -> Dict[str, Any]:
	"""
	Analyze a single equation artifact directory and return per-equation diagnostics and metrics.

	Parameters
	----------
	eq_dir : Path
		Directory containing accept_proof.json, summary.json, and forms.txt.

	Returns
	-------
	Dict[str, Any]
		Report dictionary including booleans, metrics, and method.
	"""
	ap_path = eq_dir / "accept_proof.json"
	summ_path = eq_dir / "summary.json"
	forms_path = eq_dir / "forms.txt"
	result: Dict[str, Any] = {"path": str(eq_dir)}
	ap = load_json(ap_path)
	summ = load_json(summ_path)
	ok_ap, ap_notes = validate_accept_proof(ap)
	result["accept_proof_ok"] = ok_ap
	result["accept_proof_notes"] = ap_notes
	fields = ["accepted", "method", "C", "E", "S", "L"]
	mismatches: Dict[str, Any] = {}
	for k in fields:
		v_ap = ap.get(k)
		v_sm = summ.get(k)
		if isinstance(v_ap, float) and isinstance(v_sm, float):
			if not (abs(v_ap - v_sm) <= 1e-12):
				mismatches[k] = {"accept_proof": v_ap, "summary": v_sm}
		else:
			if v_ap != v_sm:
				mismatches[k] = {"accept_proof": v_ap, "summary": v_sm}
	result["summary_match"] = (len(mismatches) == 0)
	if mismatches:
		result["summary_mismatches"] = mismatches
	ok_forms, forms_notes = validate_forms(forms_path, ap.get("f_can", ""), ap.get("g_can", ""))
	result["forms_ok"] = ok_forms
	result["forms_notes"] = forms_notes
	if ap.get("method") == "symbolic":
		e = float(ap.get("E", 1.0))
		result["symbolic_E_zero_ok"] = (abs(e - 0.0) <= 1e-12)
	if ap.get("method") == "float_probe":
		proof = ap.get("proof", {})
		max_abs = float(proof.get("max_abs_diff", 0.0))
		tol = float(proof.get("tol", 0.0))
		result["float_probe_within_tol"] = (max_abs <= tol + 1e-15)
	result["accepted"] = bool(ap.get("accepted", False))
	result["method"] = ap.get("method", "")
	result["C"] = ap.get("C", None)
	result["E"] = ap.get("E", None)
	result["S"] = ap.get("S", None)
	result["L"] = ap.get("L", None)
	return result


def summarize_equations(eq_reports: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""
	Summarize per-equation reports into corpus-level counts and simple distribution summaries.

	Parameters
	----------
	eq_reports : List[Dict[str, Any]]
		List of per-equation report dicts.

	Returns
	-------
	Dict[str, Any]
		Aggregated summary including accept rates, method counts, and metric stats.
	"""
	total = len(eq_reports)
	accepts = sum(1 for r in eq_reports if r.get("accepted"))
	by_method: Dict[str, int] = {}
	core_ok = 0
	for r in eq_reports:
		m = r.get("method", "")
		by_method[m] = by_method.get(m, 0) + 1
		if r.get("accept_proof_ok") and r.get("summary_match") and r.get("forms_ok"):
			core_ok += 1

	def safe_stats(vals: List[float]) -> Dict[str, float]:
		"""
		Compute min/median/mean/max for a list of numeric values, ignoring non-numeric entries.

		Parameters
		----------
		vals : List[float]
			Raw values (possibly with non-numerics).

		Returns
		-------
		Dict[str, float]
			Summary dictionary with count and basic statistics.
		"""
		vals = [float(v) for v in vals if isinstance(v, (int, float))]
		if not vals:
			return {"count": 0}
		vals.sort()
		n = len(vals)
		mean = sum(vals) / n
		p50 = vals[n // 2] if n % 2 == 1 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
		return {"count": n, "min": vals[0], "p50": p50, "max": vals[-1], "mean": mean}

	C_stats = safe_stats([r.get("C") for r in eq_reports])
	E_stats = safe_stats([r.get("E") for r in eq_reports])
	S_stats = safe_stats([r.get("S") for r in eq_reports])
	L_stats = safe_stats([r.get("L") for r in eq_reports])

	return {
		"equations_total": total,
		"accepted_total": accepts,
		"accept_rate": (accepts / total if total else 0.0),
		"by_method": by_method,
		"artifact_core_ok_rate": (core_ok / total if total else 0.0),
		"C_stats": C_stats,
		"E_stats": E_stats,
		"S_stats": S_stats,
		"L_stats": L_stats,
	}


def write_csv(eq_reports: List[Dict[str, Any]], out_csv: Path) -> None:
	"""
	Write a per-equation CSV table for quick inspection of acceptance and metrics.

	Parameters
	----------
	eq_reports : List[Dict[str, Any]]
		List of per-equation report dicts.
	out_csv : Path
		Destination CSV path.
	"""
	cols = [
		"name",
		"accepted",
		"method",
		"C",
		"E",
		"S",
		"L",
		"accept_proof_ok",
		"summary_match",
		"forms_ok",
		"symbolic_E_zero_ok",
		"float_probe_within_tol",
	]
	with out_csv.open("w", newline="", encoding="utf-8") as f:
		w = csv.writer(f)
		w.writerow(cols)
		for r in eq_reports:
			name = Path(r["path"]).name
			row = [
				name,
				r.get("accepted"),
				r.get("method"),
				r.get("C"),
				r.get("E"),
				r.get("S"),
				r.get("L"),
				r.get("accept_proof_ok"),
				r.get("summary_match"),
				r.get("forms_ok"),
				r.get("symbolic_E_zero_ok", None),
				r.get("float_probe_within_tol", None),
			]
			w.writerow(row)


def main() -> None:
	"""
	Entry point: validate run-level artifacts, analyze all equation subdirs, and emit reports.

	Side Effects
	------------
	Creates an output directory with:
	  - aggregate_report.json (canonical JSON)
	  - accept_table.csv (per-equation table)
	Prints a short run summary to stdout.
	"""
	ap = argparse.ArgumentParser(description="Analyze Feynman run artifacts and verify theory-aligned invariants.")
	ap.add_argument("run_dir", type=str, help="Path to runs/feynman_YYYYMMDD or similar directory")
	ap.add_argument("--out", type=str, default="report", help="Output directory (created if missing)")
	args = ap.parse_args()

	run_dir = Path(args.run_dir).expanduser().resolve()
	out_dir = Path(args.out).expanduser().resolve()
	out_dir.mkdir(parents=True, exist_ok=True)

	man_path = run_dir / "manifest.json"
	ev_path = run_dir / "events.jsonl"
	index_path = run_dir / "index.json"

	out: Dict[str, Any] = {"run_dir": str(run_dir)}

	ok_man, man_notes = (False, {"error": "missing"})
	if man_path.exists():
		ok_man, man_notes = validate_manifest(man_path)
	out["manifest_ok"] = ok_man
	out["manifest_notes"] = man_notes

	ok_ev, ev_notes = (False, {"error": "missing"})
	if ev_path.exists():
		ok_ev, ev_notes = validate_events(ev_path)
	out["events_ok"] = ok_ev
	out["events_notes"] = ev_notes

	eq_root = run_dir / "artifacts" / "equations"
	eq_reports: List[Dict[str, Any]] = []
	if eq_root.exists():
		for p in sorted(eq_root.iterdir()):
			if p.is_dir():
				try:
					eq_reports.append(analyze_equation_dir(p))
				except Exception as e:
					eq_reports.append({"path": str(p), "error": str(e)})

	summary = summarize_equations(eq_reports)
	out["equations_summary"] = summary

	if index_path.exists():
		try:
			idx = load_json(index_path)
			out["index_entries"] = len(idx) if isinstance(idx, list) else None
		except Exception as e:
			out["index_error"] = str(e)

	json_path = out_dir / "aggregate_report.json"
	with json_path.open("w", encoding="utf-8") as f:
		f.write(canonical_json(out))

	csv_path = out_dir / "accept_table.csv"
	write_csv(eq_reports, csv_path)

	print("== Feynman Run Analysis ==")
	print(f"Run dir: {run_dir}")
	print(f"Manifest OK: {out['manifest_ok']}  | Events OK: {out['events_ok']}")
	print(f"Equations: {summary['equations_total']} | Accepted: {summary['accepted_total']} ({summary['accept_rate']:.3f})")
	print(f"By method: {summary['by_method']}")
	print(f"Artifact core OK rate (accept_proof + summary + forms): {summary['artifact_core_ok_rate']:.3f}")
	print(f"Report JSON: {json_path}")
	print(f"CSV table:   {csv_path}")


if __name__ == "__main__":
	main()

