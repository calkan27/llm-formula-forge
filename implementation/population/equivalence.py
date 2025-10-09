"""
EquivalenceOrder defines a partial order over expression texts using a three-stage
equivalence check: (1) canonical structural key equality, (2) numeric fingerprint
match on a fixed grid, and (3) a sound symbolic certificate (simplify(f-g)==0)
with tie-breaking by minimal canonical complexity. The class also exposes basic
axiom checks (reflexivity, antisymmetry on canonical forms).
"""

from __future__ import annotations
from typing import List, Tuple
from implementation.complexity import Complexity
from implementation.feature_selection.dedup import PreScoreDeduper

class EquivalenceOrder:
	def __init__(self) -> None:
		"""
		Initialize the order with a Complexity instance used for canonical
		minimal complexity C_min when comparing certificate-equivalent forms.
		"""
		self._C = Complexity()

	def leq(self, f: str, g: str) -> bool:
		"""
		Return True iff f ≤ g under the robust equivalence order:
		  • If canonical structural keys match, accept immediately.
		  • Else, if numeric fingerprints match and the symbolic certificate
			proves equality, prefer the form with smaller C_min; accept when
			C_min(f) ≤ C_min(g).
		"""
		kf = PreScoreDeduper.canonical_struct_key(f)
		kg = PreScoreDeduper.canonical_struct_key(g)
		if kf == kg:
			return True
		ff = PreScoreDeduper.numeric_fingerprint(f)
		fg = PreScoreDeduper.numeric_fingerprint(g)
		if ff == fg:
			if PreScoreDeduper.symbolic_equal(f, g):
				cf = self._C.C_min(f)
				cg = self._C.C_min(g)
				if cf <= cg:
					return True
		return False

	def eq(self, f: str, g: str) -> bool:
		"""
		Return True iff f and g are equivalent in both directions (f ≤ g and g ≤ f).
		This collapses to structural identity or certified semantic equality with
		matching minimal complexity.
		"""
		if self.leq(f, g):
			if self.leq(g, f):
				return True
		return False

	def check_axioms(self, items: List[str]) -> Tuple[bool, bool]:
		"""
		Check two partial-order properties on a set of items:
		  • Reflexivity: ∀s, s ≤ s.
		  • Antisymmetry (on canonical/certified equals): if a ≤ b and b ≤ a,
			then either canonical structures match or the symbolic certificate
			confirms a == b. Returns (reflexive_ok, antisymmetric_ok).
		"""
		reflexive = True
		for s in items:
			if not self.leq(s, s):
				reflexive = False
				break

		antisym = True
		n = len(items)
		for i in range(n):
			for j in range(i + 1, n):
				a = items[i]
				b = items[j]
				if self.leq(a, b):
					if self.leq(b, a):
						ka = PreScoreDeduper.canonical_struct_key(a)
						kb = PreScoreDeduper.canonical_struct_key(b)
						if ka != kb:
							if not PreScoreDeduper.symbolic_equal(a, b):
								antisym = False
								break
			if not antisym:
				break

		return reflexive, antisym





