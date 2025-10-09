"""
Interpretability S & Calibration:

Defines a single rubric criterion used by the five-criterion rubric.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Criterion:
	"""
	A single rubric criterion with a bounded score and an optional note.

	Attributes
	----------
	name : str
		Human-readable identifier for the criterion (e.g., "modularity").
	score : float
		Raw score in [-20, 20]. Any consumer (e.g., Rubric) enforces bounds
		deterministically when constructing or aggregating.
	note : str
		Optional text note for qualitative context.
	"""
	name: str
	score: float
	note: str = ""

