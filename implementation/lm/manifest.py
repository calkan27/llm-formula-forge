from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import hashlib
import json


def _derive_seed(session: object, prompt_id: object, item_index: object, attempt: object = 0) -> int:
	"""
	Return a uint64 derived from SHA256(session||prompt_id||item_index||attempt)
	using the first 8 bytes in big-endian order.
	"""
	h = hashlib.sha256()
	h.update(str(session).encode("utf-8"))
	h.update(str(prompt_id).encode("utf-8"))
	h.update(str(item_index).encode("utf-8"))
	h.update(str(attempt).encode("utf-8"))
	val = int.from_bytes(h.digest()[:8], "big")
	return val & 0xFFFFFFFFFFFFFFFF


@dataclass(frozen=True)
class LMDeterminismManifest:
	"""
	Determinism manifest for proposal components.

	Fields
	------
	model_id        : model identifier string
	tokenizer_hash  : opaque tokenizer/version hash
	decoding        : decoding knobs (temperature, top_p, top_k, max_tokens, etc.)
	derived_seed    : uint64 seed derived from (session_id, prompt_id, item_index, attempt)
	session_id      : session/run identifier
	prompt_id       : prompt identifier (string tag)
	item_index      : integer index
	attempt         : retry attempt index
	"""
	model_id: str
	tokenizer_hash: str
	decoding: Dict[str, Any]
	derived_seed: int
	session_id: str
	prompt_id: str
	item_index: int
	attempt: int

	@staticmethod
	def build(
		model_id: str,
		tokenizer_hash: str,
		decoding: Dict[str, Any],
		session_id: Optional[str] = None,
		prompt_id: str = "PROMPT_ID",
		item_index: int = 0,
		attempt: int = 0,
		run_id: Optional[str] = None,
		round_index: Optional[int] = None,
	) -> "LMDeterminismManifest":
		"""
		Build a manifest. Accepts both the new API (session_id, item_index, attempt)
		and the legacy API (run_id, round_index). If both are provided, legacy
		arguments take precedence for those fields.
		"""
		if run_id is not None:
			sid = run_id
		else:
			sid = session_id
		if sid is None:
			sid = "session"

		if round_index is not None:
			idx = round_index
		else:
			idx = item_index
		if idx is None:
			idx = 0

		knobs_unsorted: Dict[str, Any] = {}
		for k, v in decoding.items():
			ks = str(k)
			if isinstance(v, (int, float)):
				vs = float(v)
			else:
				vs = v
			knobs_unsorted[ks] = vs
		knobs = dict(sorted(knobs_unsorted.items(), key=lambda kv: kv[0]))

		seed = _derive_seed(sid, prompt_id, idx, attempt)

		return LMDeterminismManifest(
			model_id=str(model_id),
			tokenizer_hash=str(tokenizer_hash),
			decoding=knobs,
			derived_seed=int(seed),
			session_id=str(sid),
			prompt_id=str(prompt_id),
			item_index=int(idx),
			attempt=int(attempt),
		)

	def to_json(self) -> str:
		"""
		Return a stable JSON representation with decoding knobs sorted by key.
		"""
		obj = asdict(self)
		obj["decoding"] = dict(sorted(self.decoding.items(), key=lambda kv: kv[0]))
		return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

	@staticmethod
	def build_legacy(
		model_id: str,
		tokenizer_hash: str,
		decoding: Dict[str, Any],
		run_id: str,
		round_index: int
	) -> "LMDeterminismManifest":
		"""
		Convenience wrapper for legacy call sites.
		"""
		return LMDeterminismManifest.build(
			model_id=model_id,
			tokenizer_hash=tokenizer_hash,
			decoding=decoding,
			run_id=run_id,
			round_index=round_index,
		)





