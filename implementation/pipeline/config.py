"""
Loop configuration and typed containers for the population scheduler.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import sympy as sp



@dataclass(frozen=True)
class Candidate:
	"""
	Immutable container for a proposed expression and its evaluation channels.
	"""
	text: str
	expr: Optional[sp.Expr]
	ok: bool
	provenance: str
	reason: str
	C: Optional[int] = None
	E: Optional[float] = None
	S: Optional[float] = None
	L: Optional[float] = None


@dataclass
class LoopConfig:
	"""
	Steady-state schedule and budgets.
	"""
	N: int = 200
	K: int = 30
	J: int = 10
	J_lm: int = 4
	J_mut: int = 4
	J_erc: int = 2
	T: int = 500
	lm_budget_total: int = 5000
	eps_frontier: float = 0.0


@dataclass
class LogEvent:
	"""
	Structured event for run-time logging.
	"""
	kind: str
	payload: Dict[str, object]


@dataclass
class LoopState:
	"""
	Mutable loop state tracked across rounds.
	"""
	round_idx: int = 0
	survivors: List[Candidate] = field(default_factory=list)
	log: List[LogEvent] = field(default_factory=list)
	lm_calls_used: int = 0


ScoreFn = Callable[[sp.Expr], Tuple[float, int, float, float]]

