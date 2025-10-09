from __future__ import annotations
from typing import Iterable, Tuple, Dict, List
import hashlib
import numpy as np
import sympy as sp

from implementation.complexity import Complexity
from implementation.interpretability.feature_translate import translate_line
from implementation.numeric.protected_eval import protected_lambdify

class PreScoreDeduper:
	"""
	Pre-scoring deduplication using canonical structure, numeric fingerprint, and symbolic certificate.

	Methods
	-------
	canonical_struct_key(text)  -> srepr of canonical(normalized(parse(text)))
	numeric_fingerprint(text)   -> BLAKE2b digest of deterministic evaluations
	symbolic_equal(text1, text2) -> True if simplify(expr1 - expr2) == 0
	dedup(texts)                -> (unique_texts, meta_map[text] = (struct, fp))
	"""

	@staticmethod
	def _safe_lambdify(expr: sp.Expr):
		"""
		Build a safe numeric evaluator: log(|x|+1e-12), sqrt(max(x,0)), Abs -> np.abs.
		"""
		return protected_lambdify(expr)

	@staticmethod
	def _needs_translation(text: str) -> bool:
		"""Check if text contains p*-vocab functions."""
		p_funcs = [
			'padd', 'psub', 'pmul', 'pdiv', 'plog', 'pexp', 'psqrt',
			'psin', 'pcos', 'ptanh', 'pow2', 'pow3', 'powm1', 'powm2', 'powm3'
		]
		for func in p_funcs:
			if func + '(' in text:
				return True
		return False


	@staticmethod
	def canonical_struct_key(text: str) -> str:
		"""Return the normalized structural key (SymPy srepr)."""
		if PreScoreDeduper._needs_translation(text):
			translated = translate_line(text)
		else:
			translated = text
		
		C = Complexity()
		e = C.parse(translated)
		can = C.canonical(e)
		return sp.srepr(can)

	@staticmethod
	def numeric_fingerprint(text: str, n_points: int = 8) -> str:
		"""
		Deterministic numeric fingerprint by evaluating on a fixed nonzero grid.
		"""
		if PreScoreDeduper._needs_translation(text):
			translated = translate_line(text)
		else:
			translated = text

		C = Complexity()
		expr = C.canonical(C.parse(translated))
		syms = tuple(sorted(expr.free_symbols, key=lambda s: s.name))
		f = PreScoreDeduper._safe_lambdify(expr)

		vals = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float64)
		out_codes: List[str] = []
		n_eval = max(1, int(n_points))

		for t in range(n_eval):
			assigns = {}
			for k, s in enumerate(syms):
				assigns[s] = float(vals[(2 * t + k) % len(vals)])

			if not callable(f):
				print("numeric_fingerprint: evaluator is not callable; emitting 'nan' for this point")
				out_codes.append("nan")
				continue

			missing = []
			for s in syms:
				if s not in assigns:
					missing.append(s.name)
			if len(missing) > 0:
				print(f"numeric_fingerprint: missing assignments for {missing}; emitting 'nan'")
				out_codes.append("nan")
				continue

			with np.errstate(all="ignore"):
				v = f(**{s.name: assigns[s] for s in syms})

			arr = np.asarray(v, dtype=np.float64)
			if arr.shape == ():
				v_scalar = float(arr)
			elif arr.size == 1:
				v_scalar = float(arr.reshape(()))
			else:
				print(f"numeric_fingerprint: non-scalar output shape {arr.shape}; emitting 'nan'")
				out_codes.append("nan")
				continue

			if not np.isfinite(v_scalar):
				out_codes.append("nan")
			else:
				out_codes.append(f"{v_scalar:.6f}")

		blob = "|".join(out_codes).encode("utf-8")
		h = hashlib.blake2b(blob, digest_size=8)
		return h.hexdigest()


	@staticmethod
	def symbolic_equal(text1: str, text2: str) -> bool:
		"""
		Check symbolic equality via simplify(expr1 - expr2) == 0.
		Returns False on any error (guards + prints + early returns).
		"""
		if text1 is None or text2 is None:
			print("symbolic_equal: one or both inputs are None")
			return False
		if not isinstance(text1, str) or not isinstance(text2, str):
			print("symbolic_equal: inputs must be strings")
			return False
		if text1.strip() == "" or text2.strip() == "":
			print("symbolic_equal: one or both inputs are empty strings")
			return False

		if PreScoreDeduper._needs_translation(text1):
			try:
				trans1 = translate_line(text1)
			except Exception as e:
				print(f"symbolic_equal: translate error on text1: {e}")
				return False
		else:
			trans1 = text1

		if PreScoreDeduper._needs_translation(text2):
			try:
				trans2 = translate_line(text2)
			except Exception as e:
				print(f"symbolic_equal: translate error on text2: {e}")
				return False
		else:
			trans2 = text2

		if not isinstance(trans1, str) or not isinstance(trans2, str):
			print("symbolic_equal: translated inputs are not strings")
			return False
		if trans1.strip() == "" or trans2.strip() == "":
			print("symbolic_equal: translated inputs are empty")
			return False

		C = Complexity()
		try:
			expr1 = C.parse(trans1)
		except Exception as e:
			print(f"symbolic_equal: parse error on text1: {e}")
			return False

		try:
			expr2 = C.parse(trans2)
		except Exception as e:
			print(f"symbolic_equal: parse error on text2: {e}")
			return False

		try:
			diff = sp.simplify(expr1 - expr2)
		except Exception as e:
			print(f"symbolic_equal: simplify error: {e}")
			return False

		if diff == 0:
			return True
		else:
			e1c = C.canonical(expr1)
			e2c = C.canonical(expr2)
			if sp.srepr(e1c) == sp.srepr(e2c):
				return True
			else:
				return False



	@staticmethod
	def _is_ascii_or_skip(s: str) -> bool:
		"""Return True if ASCII; otherwise print and signal skip."""
		if s is None:
			print("dedup: skipping None entry")
			return False
		if not isinstance(s, str):
			print("dedup: skipping non-string entry")
			return False
		if s.strip() == "":
			print("dedup: skipping empty string")
			return False
		if not s.isascii():
			print("dedup: non-ASCII line; skipping")
			return False
		return True

	@staticmethod
	def _translate_if_needed(s: str) -> str | None:
		"""Translate p*-vocab if needed; return None to skip on empty/invalid post-translation."""
		if PreScoreDeduper._needs_translation(s):
			translated = translate_line(s)
		else:
			translated = s

		if not isinstance(translated, str):
			print("dedup: translated form is not a string; skipping")
			return None

		if translated.strip() == "":
			print("dedup: translated form is empty; skipping")
			return None

		return translated

	@staticmethod
	def _parse_and_canonical(s: str) -> tuple[sp.Expr, str] | None:
		"""
		Parse with Complexity().parse, canonicalize, and return (expr_can, srepr_key).
		Returns None on any parse/canonicalization error so callers can skip safely.
		"""
		C = Complexity()
		try:
			e = C.parse(s)
		except Exception:
			return None
		can = C.canonical(e)
		k = sp.srepr(can)
		return can, k



	@staticmethod
	def _has_unsupported_funcs(expr: sp.Expr, allowed: set[str]) -> bool:
		"""
		Return True if expr contains any function head not in 'allowed'.
		Used to quietly drop unknowns (e.g., pfoo) before lambdify.
		"""
		for fnode in expr.atoms(sp.Function):
			name = getattr(fnode.func, "__name__", "")
			if name not in allowed:
				return True
		return False

	@staticmethod
	def _numeric_fingerprint_expr(expr_can: sp.Expr, n_eval: int = 8) -> str:
		"""
		Compute the numeric fingerprint for a canonical SymPy expression using the
		project's fixed grid and guarded evaluation. Returns a hex digest.
		"""
		syms = tuple(sorted(expr_can.free_symbols, key=lambda s: s.name))
		f = PreScoreDeduper._safe_lambdify(expr_can)

		vals = np.array([-2.0, -1.0, -0.5, 0.5, 1.0, 2.0], dtype=np.float64)
		out_codes: List[str] = []
		n_eval2 = max(1, int(n_eval))

		for t in range(n_eval2):
			assigns = {}

			for idx, sym in enumerate(syms):
				assigns[sym] = float(vals[(2 * t + idx) % len(vals)])

			if not callable(f):
				print("dedup: evaluator is not callable; emit 'nan'")
				out_codes.append("nan")
				continue

			missing = []
			for sym in syms:
				if sym not in assigns:
					missing.append(sym.name)
			if len(missing) > 0:
				print(f"dedup: missing kwargs {missing}; emit 'nan'")
				out_codes.append("nan")
				continue

			with np.errstate(all="ignore"):
				v = f(**{sym.name: assigns[sym] for sym in syms})

			arr = np.asarray(v, dtype=np.float64)
			if arr.shape == ():
				v_scalar = float(arr)
			elif arr.size == 1:
				v_scalar = float(arr.reshape(()))
			else:
				print(f"dedup: non-scalar output shape {arr.shape}; emit 'nan'")
				out_codes.append("nan")
				continue

			if not np.isfinite(v_scalar):
				out_codes.append("nan")
			else:
				out_codes.append(f"{v_scalar:.6f}")

		blob = "|".join(out_codes).encode("utf-8")
		h = hashlib.blake2b(blob, digest_size=8)
		return h.hexdigest()


	@staticmethod
	def _replace_in_lists(
		uniq: List[str],
		fp_map: Dict[str, List[str]],
		fp: str,
		old_text: str,
		new_text: str,
	) -> None:
		"""Replace 'old_text' with 'new_text' in both uniq and the fp bucket."""
		new_list = []
		for x in uniq:
			if x == old_text:
				new_list.append(new_text)
			else:
				new_list.append(x)
		uniq[:] = new_list

		new_fp_list = []
		for x in fp_map.get(fp, []):
			if x == old_text:
				new_fp_list.append(new_text)
			else:
				new_fp_list.append(x)
		fp_map[fp] = new_fp_list

	@staticmethod
	def dedup(texts: Iterable[str]) -> Tuple[List[str], Dict[str, Tuple[str, str]]]:
		"""
		Deduplicate translated feature expressions by:
		1. (canonical_struct_key, numeric_fingerprint) pairs
		2. Symbolic certificate checking when fingerprints match but structures differ
		"""
		uniq: List[str] = []
		meta: Dict[str, Tuple[str, str]] = {}
		seen: set[Tuple[str, str]] = set()
		fp_to_texts: Dict[str, List[str]] = {}

		allowed_funcs = {"log", "sqrt", "Abs", "sin", "cos", "tanh", "exp"}

		for s in texts:
			if not PreScoreDeduper._is_ascii_or_skip(s):
				continue

			translated = PreScoreDeduper._translate_if_needed(s)
			if translated is None:
				continue

			pc = PreScoreDeduper._parse_and_canonical(translated)
			if pc is None:
				print("dedup: parse/canonicalize failed; skipping")
				continue
			can, k = pc

			if PreScoreDeduper._has_unsupported_funcs(can, allowed_funcs):
				print("dedup: unsupported function head detected; skipping")
				continue

			fp = PreScoreDeduper._numeric_fingerprint_expr(can, n_eval=8)

			pair = (k, fp)
			meta[s] = pair

			if pair in seen:
				continue

			is_duplicate = False
			if fp in fp_to_texts:
				for existing in fp_to_texts[fp]:
					if PreScoreDeduper.symbolic_equal(s, existing):
						Cm = Complexity()
						c_new = Cm.C_min(s)
						c_existing = Cm.C_min(existing)

						if c_new < c_existing:
							PreScoreDeduper._replace_in_lists(uniq, fp_to_texts, fp, existing, s)

						is_duplicate = True
						break

			if not is_duplicate:
				seen.add(pair)
				uniq.append(s)
				if fp not in fp_to_texts:
					fp_to_texts[fp] = []
				fp_to_texts[fp].append(s)

		return uniq, meta

