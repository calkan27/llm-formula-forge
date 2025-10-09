"""Deterministic Feynman v1 (+ bonus) loader with canonical SymPy parsing, SI-unit mapping, and safe domain inference.

This class wires together:
  • schema adaptation for Feynman CSVs,
  • SymPy parsing and canonicalization with row-aware symbol binding,
  • unit-table parsing,
  • safe-domain heuristics,
  • deterministic record iteration and digesting.
"""

from __future__ import annotations
import csv
import hashlib
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import pandas as pd
import sympy as sp

from .sympy_utils import SympyUtils
from .feynman_meta import FeynmanMeta


@dataclass(frozen=True)
class FeynmanRecord:
	"""Immutable record containing canonical fields for a Feynman equation/problem."""
	name: str
	expr: sp.Expr
	variables: Tuple[str, ...]
	units: Dict[str, Tuple[int, int, int, int, int, int, int]]
	safe_domain: Dict[str, Tuple[float, float]]
	notes: str
	digest: str


class FeynmanLoader:
	"""Deterministic loader for Feynman v1 (+ bonus) that unpacks archives once and yields canonical records."""


	_DIM_KEYS = ("M", "L", "T", "I", "Theta", "N", "J")

	tmp_dim_suffixes = []
	for k in _DIM_KEYS:
		tmp_dim_suffixes.append(f"_{k}")
	_DIM_SUFFIXES = tuple(tmp_dim_suffixes)

	MAIN_TAR = "Feynman_with_units.tar.gz"
	BONUS_TAR = "bonus_without_units.tar.gz"
	MAIN_META = "FeynmanEquations.csv"
	BONUS_META = "BonusEquations.csv"
	UNITS_CSV = "units.csv"
	SI_DIM_COUNT = 7

	def __init__(self, root: Path | str = "."):
		"""Create a loader and prepare archives/metadata."""
		self.root = Path(root)
		self.raw_dir = self.root / "data" / "feynman-v1" / "raw"
		self.meta_dir = self.root / "data" / "feynman-v1" / "meta"
		self._unpack_if_needed(self.raw_dir / self.MAIN_TAR)
		self._unpack_if_needed(self.raw_dir / self.BONUS_TAR)
		main_path = self.meta_dir / self.MAIN_META
		bonus_path = self.meta_dir / self.BONUS_META
		self.df_main = self._read_csv(main_path)
		self.df_bonus = self._read_csv(bonus_path)
		df_main_norm = self._filter_core_main(self.df_main)
		if len(df_main_norm) != len(self.df_main) or not df_main_norm.equals(self.df_main):
			name_col, _, _ = FeynmanMeta.detect_columns(df_main_norm)
			df_out = df_main_norm.sort_values(by=name_col, kind="mergesort", ascending=True)
			df_out.to_csv(main_path, index=False)
			self.df_main = df_out.reset_index(drop=True)
		else:
			self.df_main = self.df_main.reset_index(drop=True)
		self.df_bonus = self.df_bonus.reset_index(drop=True)
		self.units = self._load_units(self.meta_dir / self.UNITS_CSV)


	def iter_main(self) -> Iterator[FeynmanRecord]:
		"""Yield main Feynman v1 problems deterministically ordered by name."""
		yield from self._iter_from_df(self.df_main)

	def iter_bonus(self) -> Iterator[FeynmanRecord]:
		"""Yield bonus Feynman problems deterministically ordered by name."""
		yield from self._iter_from_df(self.df_bonus)

	def iter_all(self) -> Iterator[FeynmanRecord]:
		"""Yield main then bonus records, each deterministically ordered by name."""
		yield from self.iter_main()
		yield from self.iter_bonus()

	@staticmethod
	def infer_safe_domain(expr: sp.Expr, variables: Iterable[str]) -> Dict[str, Tuple[float, float]]:
		"""Infer per-variable intervals that avoid singularities under protected numerics.
		"""
		out: Dict[str, Tuple[float, float]] = {}
		for v in variables:
			sym = sp.Symbol(v)
			L, U = -1.0, 1.0
			if SympyUtils.inside_function(expr, sym, sp.log):
				L, U = -1.0, 1.0
			elif (SympyUtils.appears_in_denominator(expr, sym)
				  or SympyUtils.has_negative_power(expr, sym)
				  or SympyUtils.under_sqrt(expr, sym)):
				L, U = 0.5, 2.0
			out[v] = (float(L), float(U))
		return out

	def _iter_from_df(self, df: pd.DataFrame) -> Iterator[FeynmanRecord]:
		"""Yield records from a metadata dataframe in deterministic order.

		Rows with missing/NaN name or equation are skipped to enforce a valid corpus.
		"""
		name_col, eq_col, notes_col = FeynmanMeta.detect_columns(df)
		df_sorted = df.sort_values(by=name_col, kind="mergesort", ascending=True).reset_index(drop=True)

		for _, row in df_sorted.iterrows():
			name_raw = row[name_col]
			expr_raw = row[eq_col]
			name = self._clean_str(name_raw)
			expr_text = self._clean_str(expr_raw)
			if name is None or expr_text is None:
				continue
			vars_from_cols = FeynmanMeta.variables_from_vname_columns(row)
			if vars_from_cols:
				extra_symbols = vars_from_cols
			else:
				extra_symbols = None
			expr = SympyUtils.to_sympy(expr_text, extra_symbols=extra_symbols)
			if vars_from_cols:
				variables = vars_from_cols
			else:
				variables = FeynmanMeta.variables_from_expr(expr)
			units_map = self._units_for_name(name, variables)
			safe_domain = self.infer_safe_domain(expr, variables)
			if notes_col:
				if pd.notna(row[notes_col]):
					notes_val = row[notes_col]
				else:
					notes_val = ""
			else:
				notes_val = ""
			cleaned_notes = self._clean_str(notes_val)
			if cleaned_notes:
				notes = cleaned_notes
			else:
				notes = ""
			digest = record_digest(name, expr, variables, units_map, safe_domain, notes)
			yield FeynmanRecord(
				name=name,
				expr=expr,
				variables=variables,
				units=units_map,
				safe_domain=safe_domain,
				notes=notes,
				digest=digest,
			)

	def _load_units(self, path: Path) -> Dict[str, Dict[str, Tuple[int, ...]]]:
		"""Parse units.csv into a mapping: name -> symbol/base -> 7-tuple SI exponents.
		"""
		if not path.exists():
			raise FileNotFoundError(f"Missing units file: {path}")

		with path.open("r", newline="", encoding="utf-8") as f:
			reader, fieldnames = self._sniff_and_reader(f)
			lower = [c.lower() for c in fieldnames]
			if self._has_long_shape(lower):
				return self._parse_units_long(reader, fieldnames)
			return self._parse_units_suffix(reader, fieldnames)

	@staticmethod
	def _sniff_and_reader(f) -> tuple[csv.DictReader, list[str]]:
		"""Build a DictReader with a sniffed dialect and normalized fieldnames (trimmed)."""
		sample = f.read(4096)
		f.seek(0)
		if sample:
			dialect = csv.Sniffer().sniff(sample)
		else:
			dialect = csv.excel
		reader = csv.DictReader(f, dialect=dialect)
		fieldnames_raw = reader.fieldnames or []
		fieldnames: list[str] = []
		for c in fieldnames_raw:
			fieldnames.append(c.strip())
		reader.fieldnames = fieldnames  
		return reader, fieldnames

	@staticmethod
	def _key_of(fieldnames: list[str], requested: str) -> str:
		"""Case-insensitive lookup for a column name; preserves original spelling."""
		r = requested.lower()
		for c in fieldnames:
			if c.lower().lstrip("\ufeff").strip() == r:
				return c
		return requested  

	def _parse_units_long(self, reader: csv.DictReader, fieldnames: list[str]) -> Dict[str, Dict[str, Tuple[int, ...]]]:
		"""Parse the long/vertical schema: Name, Symbol, M, L, T, I, Theta, N, J."""
		lk = lambda k: self._key_of(fieldnames, k)
		name_col = lk("name")
		sym_col  = lk("symbol")
		dim_cols_list: List[str] = []
		for k in self._DIM_KEYS:
			dim_cols_list.append(lk(k))
		dim_cols = dim_cols_list
		units_map: Dict[str, Dict[str, Tuple[int, ...]]] = {}
		for row in reader:
			nm  = str(row[name_col]).strip()
			sym = str(row[sym_col]).strip()
			if not nm or not sym:
				continue
			dims_list: List[int] = []
			for c in dim_cols:
				dims_list.append(int(row[c]))
			dims: Tuple[int, ...] = tuple(dims_list)
			units_map.setdefault(nm, {})[sym] = dims
		return units_map



	def _parse_units_suffix(self, reader: csv.DictReader, fieldnames: list[str]) -> Dict[str, Dict[str, Tuple[int, ...]]]:
		"""Parse the wide/suffix schema: Name, then for each base: base_{M,L,T,I,Theta,N,J}."""
		lk = lambda k: self._key_of(fieldnames, k)
		name_col = lk("name")

		units_map: Dict[str, Dict[str, Tuple[int, ...]]] = {}
		for row in reader:
			nm = str(row.get(name_col, "")).strip()
			if not nm:
				continue
			bases = self._deduce_bases(fieldnames, name_col)
			symbol_map: Dict[str, Tuple[int, ...]] = {}
			for base in bases:
				dims = self._dims_tuple_from_row(row, base)
				if dims is not None:
					symbol_map[base] = dims
			units_map[nm] = symbol_map
		return units_map

	def _deduce_bases(self, fieldnames: list[str], name_col: str) -> list[str]:
		"""Return unique base names that have all or some of the _DIM_SUFFIXES present."""
		bases: set[str] = set()
		for col in fieldnames:
			if col == name_col:
				continue
			for suf in self._DIM_SUFFIXES:
				if col.endswith(suf):
					bases.add(col[: -len(suf)])
					break
		return sorted(bases)

	@staticmethod
	def _has_long_shape(lower_fields: list[str]) -> bool:
		need = {"m", "l", "t", "i", "theta", "n", "j"}
		return need.issubset(set(lower_fields))

	def _dims_tuple_from_row(self, row: dict, base: str) -> Tuple[int, ...] | None:
		"""Pull a 7-tuple of ints for a given base from a row; missing cells are treated as zeros."""
		dims: list[int] = []
		found_any = False
		for suf in self._DIM_SUFFIXES:
			key = base + suf
			val = row.get(key, "")
			s = str(val).strip()
			if s != "":
				dims.append(int(val))
				found_any = True
			else:
				dims.append(0)
		if found_any or bool(dims):
			return tuple(dims)
		else:
			return None

	@staticmethod
	def _read_csv(path: Path) -> pd.DataFrame:
		"""Read a CSV file using pandas with UTF-8 decoding."""
		if not path.exists():
			raise FileNotFoundError(f"Missing CSV: {path}")
		return pd.read_csv(path)

	def _unpack_if_needed(self, tgz_path: Path) -> None:
		"""Unpack a .tar.gz archive into a sibling directory if not already unpacked."""
		if not tgz_path.exists():
			raise FileNotFoundError(f"Missing required archive: {tgz_path}")
		out_dir = tgz_path.with_suffix("").with_suffix("")
		if out_dir.exists():
			return
		out_dir.mkdir(parents=True, exist_ok=True)
		with tarfile.open(tgz_path, "r:gz") as tf:
			for m in tf.getmembers():
				member_path = out_dir / m.name
				if not is_within_dir(out_dir, member_path):
					raise RuntimeError(f"Unsafe member path in archive: {m.name}")
			tf.extractall(out_dir)

	def _units_for_name(self, name: str, variables: Tuple[str, ...]) -> Dict[str, Tuple[int, ...]]:
		"""Return units mapping for all variables present and optional 'target' if available."""
		m = self.units.get(name, {})
		out: Dict[str, Tuple[int, ...]] = {}

		for v in variables:
			if v in m and isinstance(m[v], tuple) and len(m[v]) == self.SI_DIM_COUNT:
				vals_list: List[int] = []
				for x in m[v]:
					vals_list.append(int(x))
				out[v] = tuple(vals_list)

		if "target" in m and len(m["target"]) == self.SI_DIM_COUNT:
			t_vals_list: List[int] = []
			for x in m["target"]:
				t_vals_list.append(int(x))
			out["target"] = tuple(t_vals_list)

		return out


	@staticmethod
	def _clean_str(val: object) -> str | None:
		"""Convert to trimmed string; return None for empty/NaN-like."""
		s = str(val).strip()
		if s == "" or s.lower() == "nan":
			return None
		return s

	def _filter_core_main(self, df: pd.DataFrame) -> pd.DataFrame:
		"""
		Drop rows with missing/empty name or equation, preserve all remaining columns verbatim.
		"""
		name_col, eq_col, _ = FeynmanMeta.detect_columns(df)

		name_mask_list = []
		for v in df[name_col]:
			if self._clean_str(v) is not None:
				name_mask_list.append(True)
			else:
				name_mask_list.append(False)
		mask_name_ok = pd.Series(name_mask_list, index=df.index)

		eq_mask_list = []
		for v in df[eq_col]:
			if self._clean_str(v) is not None:
				eq_mask_list.append(True)
			else:
				eq_mask_list.append(False)
		mask_eq_ok = pd.Series(eq_mask_list, index=df.index)

		clean = df[mask_name_ok & mask_eq_ok].copy()

		mask_core = clean[name_col].astype(str).apply(self._is_core)
		core = clean[mask_core].copy()
		return core.reset_index(drop=True)

	@staticmethod
	def _is_core(nm: str) -> bool:
		"""
		Return True iff a name looks like a Feynman v1 *core* ID.
		"""
		s = nm.strip()
		if "." not in s:
			return False
		left, right = s.split(".", 1)
		if not left or not right:
			return False
		romans = set("IVXLCDM")
		any_non_roman = False
		for ch in left:
			if ch not in romans:
				any_non_roman = True
				break
		if any_non_roman:
			return False
		return right[:1].isdigit()

def record_digest(
	name: str,
	expr: sp.Expr,
	variables: Tuple[str, ...],
	units: Dict[str, Tuple[int, ...]],
	safe_domain: Dict[str, Tuple[float, float]],
	notes: str,
) -> str:
	"""Compute a deterministic BLAKE2b digest across canonical serialization of record fields."""
	h = hashlib.blake2b(digest_size=16)

	h.update(name.encode("utf-8"))
	h.update(sp.srepr(expr).encode("utf-8"))
	h.update(",".join(variables).encode("utf-8"))

	for k in sorted(units.keys()):
		h.update(k.encode("utf-8"))

		unit_parts_list: List[str] = []
		for x in units[k]:
			unit_parts_list.append(str(int(x)))
		units_joined = ",".join(unit_parts_list)

		h.update(units_joined.encode("utf-8"))

	for k in sorted(safe_domain.keys()):
		L, U = safe_domain[k]
		h.update(k.encode("utf-8"))
		h.update(f"{L:.12g},{U:.12g}".encode("utf-8"))

	h.update(notes.encode("utf-8"))
	return h.hexdigest()





def is_within_dir(directory: Path, target: Path) -> bool:
	"""Return True if target resolves within directory to prevent path traversal."""
	directory = directory.resolve()
	target = target.resolve()
	dir_parts = directory.parts
	tgt_parts = target.parts
	if len(tgt_parts) < len(dir_parts):
		return False
	i = 0
	while i < len(dir_parts) and tgt_parts[i] == dir_parts[i]:
		i += 1
	if i == len(dir_parts):
		return True
	else:
		return False



def key_of(fieldnames: List[str], requested: str) -> str:
	"""Return the canonical field name from fieldnames matching requested, case-insensitively and trimming BOMs."""
	r = requested.lower()
	for c in fieldnames:
		cc = c.lower().lstrip("\ufeff").strip()
		if cc == r:
			return c
	return requested

