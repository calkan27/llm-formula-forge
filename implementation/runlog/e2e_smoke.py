"""
This module builds a deterministic run manifest and emits canonical JSONL events.
Determinism is guaranteed by: seed = uint64(SHA256(s||p||i||a)), canonical JSON
serialization, and logical-time event counters. The manifest hash is the SHA-256
of the canonicalized core (excluding the hash itself).
"""

from __future__ import annotations
import json
import hashlib
import platform
import sys
import numpy as np
from typing import Dict, List, Tuple
from implementation.lm.manifest import LMDeterminismManifest

class RunLogSmoke:
	"""
	Class facade for building manifests, emitting events, and JSONL canonicalization.
	"""

	@staticmethod
	def canonical_json(o: dict) -> str:
		"""
		Return a canonical JSON string with sorted keys and fixed separators.
		"""
		return json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

	@staticmethod
	def sha256_hex(b: bytes) -> str:
		"""
		Return the hexadecimal SHA-256 digest of a bytes buffer.
		"""
		h = hashlib.sha256()
		h.update(b)
		return h.hexdigest()

	@staticmethod
	def env_block() -> Dict[str, str]:
		"""
		Return a stable environment descriptor with fixed keys and string values.
		"""
		pyver = ".".join(str(x) for x in sys.version_info[:3])
		npver = np.__version__
		sysname = platform.system()
		machine = platform.machine()
		processor = platform.processor()
		impl = platform.python_implementation()
		return {
			"python_version": pyver,
			"numpy_version": npver,
			"system": sysname,
			"machine": machine,
			"processor": processor,
			"python_impl": impl,
		}

	@staticmethod
	def stable_decoding(knobs: Dict[str, float]) -> Dict[str, float]:
		"""
		Return decoding knobs with keys sorted and numerics coerced to float.
		"""
		items = sorted([(str(k), float(v) if isinstance(v, (int, float)) else v) for k, v in knobs.items()], key=lambda kv: kv[0])
		out: Dict[str, float] = {}
		for k, v in items:
			out[k] = v
		return out

	@staticmethod
	def build_manifest(
		model_id: str,
		tokenizer_hash: str,
		decoding: Dict[str, float],
		session_id: str,
		prompt_id: str,
		item_index: int,
		attempt: int
	) -> Dict[str, object]:
		"""
		Build a deterministic run manifest dict with a stable manifest hash.
		"""
		dec = RunLogSmoke.stable_decoding(decoding)
		man = LMDeterminismManifest.build(
			model_id=model_id,
			tokenizer_hash=tokenizer_hash,
			decoding=dec,
			session_id=session_id,
			prompt_id=prompt_id,
			item_index=int(item_index),
			attempt=int(attempt),
		)
		env = RunLogSmoke.env_block()
		core = {
			"model_id": man.model_id,
			"tokenizer_hash": man.tokenizer_hash,
			"decoding": man.decoding,
			"derived_seed": int(man.derived_seed),
			"session_id": man.session_id,
			"prompt_id": man.prompt_id,
			"item_index": int(man.item_index),
			"attempt": int(man.attempt),
			"env": env,
		}
		h = RunLogSmoke.sha256_hex(RunLogSmoke.canonical_json(core).encode("utf-8"))
		out = dict(core)
		out["manifest_hash"] = h
		return out

	@staticmethod
	def emit_events(seed: int, rounds: int) -> List[Dict[str, object]]:
		"""
		Emit a deterministic list of events with logical-time counters.
		"""
		r = max(0, int(rounds))
		ts = 0
		out: List[Dict[str, object]] = []
		ev1 = {"ts": ts, "kind": "start", "payload": {"seed": int(seed)}}
		out.append(ev1)
		ts = ts + 1
		for i in range(r):
			ev = {"ts": ts, "kind": "round", "payload": {"index": int(i)}}
			out.append(ev)
			ts = ts + 1
		ev2 = {"ts": ts, "kind": "stop", "payload": {"rounds": r}}
		out.append(ev2)
		return out

	@staticmethod
	def run_once(session_id: str, prompt_id: str, item_index: int, attempt: int) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
		"""
		Build a manifest and a small deterministic event stream for smoke testing.
		"""
		dec = {"temperature": 0.7, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.1, "max_tokens": 128}
		man = RunLogSmoke.build_manifest(
			model_id="llama",
			tokenizer_hash="tok",
			decoding=dec,
			session_id=session_id,
			prompt_id=prompt_id,
			item_index=item_index,
			attempt=attempt,
		)
		seed = int(man["derived_seed"] & 0xFFFFFFFFFFFFFFFF)
		events = RunLogSmoke.emit_events(seed=seed, rounds=3)
		return man, events

	@staticmethod
	def to_jsonl(events: List[Dict[str, object]]) -> str:
		"""
		Serialize events to canonical JSONL (one canonical object per line).
		"""
		lines: List[str] = []
		for e in events:
			lines.append(RunLogSmoke.canonical_json(e))
		return "\n".join(lines)

	@staticmethod
	def validate_event_shape(event: Dict[str, object]) -> bool:
		"""
		Return True iff event has exactly {ts:int, kind:str, payload:dict}.
		"""
		if not isinstance(event, dict):
			return False
		if "ts" not in event:
			return False
		if "kind" not in event:
			return False
		if "payload" not in event:
			return False
		if not isinstance(event["ts"], int):
			return False
		if not isinstance(event["kind"], str):
			return False
		if not isinstance(event["payload"], dict):
			return False
		return True



