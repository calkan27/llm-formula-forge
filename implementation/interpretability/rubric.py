"""
Five-criterion rubric container with stable JSON schema and idempotent
serialization. Raw total x ∈ [-100, 100].
"""

from __future__ import annotations
from typing import Dict, List
import json

from .criterion import Criterion


class Rubric:
	"""
	Fixed five-criterion rubric with deterministic behavior.

	The canonical criteria (fixed order):
	  1) structural_simplicity
	  2) variable_semantics_units
	  3) modularity
	  4) parameter_interpretability
	  5) domain_behavior
	"""

	_CANON_ORDER = (
		"structural_simplicity",
		"variable_semantics_units",
		"modularity",
		"parameter_interpretability",
		"domain_behavior",
	)

	def __init__(self, criteria: List[Criterion] | None = None) -> None:
		"""
		Initialize a rubric with a list of Criterion. If omitted, construct
		score=0, note="" for all canonical criteria.
		"""
		if criteria is None:
			self.criteria: List[Criterion] = []
			for n in self._CANON_ORDER:
				self.criteria.append(Criterion(name=n, score=0.0, note=""))
		else:
			m: Dict[str, Criterion] = {}
			for c in criteria:
				m[c.name] = c
			canon: List[Criterion] = []
			for name in self._CANON_ORDER:
				if name in m:
					c = m[name]
				else:
					c = Criterion(name=name, score=0.0, note="")
				score = float(c.score)
				if score < -20.0:
					score = -20.0
				elif score > 20.0:
					score = 20.0
				canon.append(Criterion(name=name, score=score, note=str(c.note)))
			self.criteria = canon


	@staticmethod
	def schema() -> Dict[str, object]:
		"""
		Return an idempotent, deterministic JSON-schema-like dict describing
		the rubric layout and per-criterion bounds.
		"""
		properties: Dict[str, object] = {}
		for name in Rubric._CANON_ORDER:
			properties[name] = {
				"type": "object",
				"properties": {
					"score": {"type": "number", "minimum": -20.0, "maximum": 20.0},
					"note": {"type": "string"},
				},
				"required": ["score"],
				"additionalProperties": False,
			}

		return {
			"type": "object",
			"title": "Interpretability Rubric (5 criteria, each in [-20,20])",
			"properties": properties,
			"required": list(Rubric._CANON_ORDER),
			"additionalProperties": False,
		}


	@classmethod
	def from_mapping(cls, mapping: Dict[str, Dict[str, object]]) -> "Rubric":
		"""
		Construct a Rubric from {name: {"score": s, "note": "..."}}, ignoring
		unknown keys and filling missing ones with score=0, note="".
		"""
		crits: List[Criterion] = []
		for name in cls._CANON_ORDER:
			item = mapping.get(name, {})
			score = float(item.get("score", 0.0))
			note = str(item.get("note", ""))
			if score < -20.0:
				score = -20.0
			if score > 20.0:
				score = 20.0
			crits.append(Criterion(name=name, score=score, note=note))
		return cls(criteria=crits)

	def raw_total(self) -> float:
		"""Return raw total x clipped to [-100, 100]."""
		total = 0.0
		for c in self.criteria:
			total += c.score
		x = total

		if x < -100.0:
			return -100.0
		if x > 100.0:
			return 100.0
		return float(x)

	def to_json(self) -> str:
		"""Stable JSON representation in canonical order (idempotent across runs)."""
		obj: Dict[str, dict] = {}
		for name in self._CANON_ORDER:
			found_score = None
			for c in self.criteria:
				if c.name == name:
					found_score = c.score
					break
			found_note = None
			for c in self.criteria:
				if c.name == name:
					found_note = c.note
					break
			obj[name] = {"score": found_score, "note": found_note}
		return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))



