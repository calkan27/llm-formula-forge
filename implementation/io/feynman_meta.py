"""CSV schema adapter for Feynman v1 metadata.

Class:
  • FeynmanMeta: static helpers to detect columns and extract variable lists.

Schema notes:
  • Main tables use 'Formula' for the expression and 'v1_name'..'v10_name' for variables.
"""

from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import sympy as sp


class FeynmanMeta:
	"""Static namespace for Feynman CSV schema helpers."""

	@staticmethod
	def _colmap(df: pd.DataFrame) -> dict:
		"""Return a case-insensitive mapping from logical keys to actual dataframe columns."""
		lc = {}
		for c in df.columns:
			lc[c.lower().strip()] = c
		m = {}
		if "name" in lc:
			m["name"] = lc["name"]
		elif "filename" in lc:
			m["name"] = lc["filename"]
		elif "id" in lc:
			m["name"] = lc["id"]
		else:
			m["name"] = list(df.columns)[0]
		if "equation" in lc:
			m["equation"] = lc["equation"]
		elif "formula" in lc:
			m["equation"] = lc["formula"]
		elif "expr" in lc:
			m["equation"] = lc["expr"]
		elif "y" in lc:
			m["equation"] = lc["y"]
		else:
			m["equation"] = "Formula"
		if "notes" in lc:
			m["notes"] = lc["notes"]
		elif "comment" in lc:
			m["notes"] = lc["comment"]
		else:
			m["notes"] = None
		return m


	@staticmethod
	def variables_from_vname_columns(row: pd.Series) -> Tuple[str, ...]:
		"""Extract variables from v1_name..v10_name columns if present; returns empty tuple if none."""
		names: List[str] = []
		for k in row.index:
			kl = k.lower().strip()
			if kl.endswith("_name") and kl.startswith("v"):
				val = str(row[k]).strip()
				if val and val.lower() != "nan":
					names.append(val)
		seen = set()
		out: List[str] = []
		for v in names:
			if v not in seen:
				out.append(v)
				seen.add(v)
		return tuple(out)

	@staticmethod
	def variables_from_expr(expr: sp.Expr) -> Tuple[str, ...]:
		"""Infer variables from a SymPy expression deterministically (alphabetical)."""
		names_set: set[str] = set()
		for s in expr.free_symbols:
			names_set.add(str(s))

		names_list = list(names_set)
		names_list.sort()
		return tuple(names_list)


	@staticmethod
	def detect_columns(df: pd.DataFrame) -> Tuple[str, str, str | None]:
		"""Return the actual column names for (name, equation, notes)."""
		m = FeynmanMeta._colmap(df)
		return m["name"], m["equation"], m["notes"]

