"""Persistence helpers for acceptance artifacts (class-based)."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict


class ProofPersister:
	"""Encapsulates writing of JSON and form artifacts for acceptance proofs."""

	def __init__(self) -> None:
		"""Initialize stateless persister."""

	def _write_json(self, path: Path, payload: Dict[str, object]) -> None:
		"""Write a compact, stable JSON payload to 'path'."""
		with path.open("w", encoding="utf-8") as f:
			f.write(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")))

	def _write_forms(self, path: Path, f_can: str, g_can: str) -> None:
		"""Write canonical forms (f_can then g_can) to 'path', one per line."""
		with path.open("w", encoding="utf-8") as f:
			f.write(str(f_can) + "\n")
			f.write(str(g_can) + "\n")

	def persist(self, out_dir: Path, payload: Dict[str, object]) -> None:
		"""
		Write accept_proof.json and forms.txt with canonical strings.

		accept_proof.json: full JSON payload of the acceptance result.
		forms.txt: two lines with f_can then g_can (exact strings), to ease offline checks.
		"""
		out_dir.mkdir(parents=True, exist_ok=True)
		self._write_json(out_dir / "accept_proof.json", payload)
		self._write_forms(out_dir / "forms.txt", str(payload.get("f_can", "")), str(payload.get("g_can", "")))

