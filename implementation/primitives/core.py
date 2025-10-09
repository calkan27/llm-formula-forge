"""
Protected Primitives & Stability (production)

This module provides a hardened numeric core with float64 semantics and guards:
  • Binary: padd, psub, pmul, pdiv
  • Unary:  plog, pexp (clipped), psqrt, invsqrt, psin, pcos, ptanh
  • Fixed small-power maps: pow2, pow3, powm1, powm2, powm3
  • ERC sampler with mixture: 30% in {0, ±1/2, ±1, ±2, ±π, ±e}, 35% Uniform[-3,3], 35% signed LogUniform[1e-4,1e4]
All functions operate on and return numpy.float64 arrays or scalars with guards to avoid NaN/Inf.

Public API:
  • class Primitives: static methods implementing all primitives
  • top-level proxies with the same names for ergonomic imports
"""

from __future__ import annotations
import math
import numpy as np


class Primitives:
	"""Protected numeric primitives with float64 semantics and stability guards."""

	EPS: float = 1e-12
	EXP_CLIP: float = 80.0

	@staticmethod
	def _as_array(x):
		"""Convert input to np.float64 ndarray."""
		return np.asarray(x, dtype=np.float64)

	@staticmethod
	def padd(x, y):
		"""Protected addition."""
		return Primitives._as_array(x) + Primitives._as_array(y)

	@staticmethod
	def psub(x, y):
		"""Protected subtraction."""
		return Primitives._as_array(x) - Primitives._as_array(y)

	@staticmethod
	def pmul(x, y):
		"""Protected multiplication."""
		return Primitives._as_array(x) * Primitives._as_array(y)

	@staticmethod
	def pdiv(x, y):
		"""Protected division with denominator guarding."""
		x64 = Primitives._as_array(x)
		y64 = Primitives._as_array(y)
		den = np.where(np.abs(y64) > Primitives.EPS, y64, np.sign(y64) * Primitives.EPS + (y64 == 0) * Primitives.EPS)
		return x64 / den

	@staticmethod
	def plog(x):
		"""Protected natural logarithm log(|x|+eps)."""
		return np.log(np.abs(Primitives._as_array(x)) + Primitives.EPS)

	@staticmethod
	def pexp(x):
		"""Protected exponential with input clipping."""
		return np.exp(np.clip(Primitives._as_array(x), -Primitives.EXP_CLIP, Primitives.EXP_CLIP))

	@staticmethod
	def psqrt(x):
		"""Protected square root on max(x,0)."""
		return np.sqrt(np.clip(Primitives._as_array(x), 0.0, None))

	@staticmethod
	def invsqrt(x):
		"""Protected inverse square root using psqrt and lower bound."""
		root = Primitives.psqrt(x)
		root_safe = np.where(root > 0, root, Primitives.EPS)
		return 1.0 / root_safe

	@staticmethod
	def psin(x):
		"""Sine."""
		return np.sin(Primitives._as_array(x))

	@staticmethod
	def pcos(x):
		"""Cosine."""
		return np.cos(Primitives._as_array(x))

	@staticmethod
	def ptanh(x):
		"""Hyperbolic tangent."""
		return np.tanh(Primitives._as_array(x))

	@staticmethod
	def pow2(x):
		"""Fixed power: x^2."""
		z = Primitives._as_array(x)
		return z * z

	@staticmethod
	def pow3(x):
		"""Fixed power: x^3."""
		z = Primitives._as_array(x)
		return z * z * z

	@staticmethod
	def powm1(x):
		"""Fixed power: x^-1 with protection."""
		z = Primitives._as_array(x)
		zs = np.where(np.abs(z) > Primitives.EPS, z, np.sign(z) * Primitives.EPS + (z == 0) * Primitives.EPS)
		return 1.0 / zs

	@staticmethod
	def powm2(x):
		"""Fixed power: x^-2 with protection."""
		z = Primitives._as_array(x)
		zs = np.where(np.abs(z) > Primitives.EPS, z, np.sign(z) * Primitives.EPS + (z == 0) * Primitives.EPS)
		return 1.0 / (zs * zs)

	@staticmethod
	def powm3(x):
		"""Fixed power: x^-3 with protection."""
		z = Primitives._as_array(x)
		zs = np.where(np.abs(z) > Primitives.EPS, z, np.sign(z) * Primitives.EPS + (z == 0) * Primitives.EPS)
		return 1.0 / (zs * zs * zs)

	@staticmethod
	def erc(rng: np.random.Generator | None = None, size: int | None = None):
		"""Sample ephemeral random constants with the specified 30/35/35 mixture in float64."""
		if rng is None:
			rng = np.random.default_rng(20250928)

		if size is None:
			n = 1
		else:
			n = int(size)

		u = rng.random(n)
		out = np.empty(n, dtype=np.float64)

		disc = np.array(
			[0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, math.pi, -math.pi, math.e, -math.e],
			dtype=np.float64,
		)
		mix_disc = (u < 0.30)
		mix_uni = (u >= 0.30) & (u < 0.65)
		mix_log = (u >= 0.65)

		k = int(np.sum(mix_disc))
		if k > 0:
			out[mix_disc] = rng.choice(disc, size=k)

		k = int(np.sum(mix_uni))
		if k > 0:
			out[mix_uni] = rng.uniform(-3.0, 3.0, size=k)

		k = int(np.sum(mix_log))
		if k > 0:
			mag = np.exp(rng.uniform(np.log(1e-4), np.log(1e4), size=k))
			sign = np.where(rng.random(k) < 0.5, -1.0, 1.0)
			out[mix_log] = sign * mag

		if size is None:
			return float(out[0])
		else:
			return out



def padd(x, y): return Primitives.padd(x, y)
def psub(x, y): return Primitives.psub(x, y)
def pmul(x, y): return Primitives.pmul(x, y)
def pdiv(x, y): return Primitives.pdiv(x, y)

def plog(x): return Primitives.plog(x)
def pexp(x): return Primitives.pexp(x)
def psqrt(x): return Primitives.psqrt(x)
def invsqrt(x): return Primitives.invsqrt(x)

def psin(x): return Primitives.psin(x)
def pcos(x): return Primitives.pcos(x)
def ptanh(x): return Primitives.ptanh(x)

def pow2(x): return Primitives.pow2(x)
def pow3(x): return Primitives.pow3(x)
def powm1(x): return Primitives.powm1(x)
def powm2(x): return Primitives.powm2(x)
def powm3(x): return Primitives.powm3(x)

def erc(rng: np.random.Generator | None = None, size: int | None = None):
	"""Top-level ERC sampler proxy."""
	return Primitives.erc(rng=rng, size=size)

