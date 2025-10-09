"""
MCTS for Explanations (UCB1; r ∈ [0,1], distinct from S)

Implements tree search with UCB1 selection using c=√2, fixed branching cap W, max depth Dmax,
bounded rollout horizon Broll, and a fixed simulation budget Nsims. All randomness is controlled
by a deterministic NumPy PCG64 generator. Rewards are assumed bounded in [0,1].
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass


@dataclass
class NodeStats:
	N: int
	W: float

class MCTSExplain:
	"""
	MCTS for explanation search using UCB1 at selection time.
	"""

	def __init__(self, c: float = math.sqrt(2.0), W: int = 6, Dmax: int = 5, Broll: int = 20, Nsims: int = 4000, seed: int = 0) -> None:
		"""Initialize hyperparameters and deterministic RNG."""
		self.c = float(c)
		self.W = int(W)
		self.Dmax = int(Dmax)
		self.Broll = int(Broll)
		self.Nsims = int(Nsims)
		self.rng = np.random.Generator(np.random.PCG64(int(seed)))
		self.N: dict[tuple[int, ...], NodeStats] = {}
		self.children: dict[tuple[int, ...], list[int]] = {}
		self.parent: dict[tuple[int, ...], tuple[int, ...] | None] = {}
		self.expanded: set[tuple[int, ...]] = set()

	def _ensure(self, s: tuple[int, ...]) -> None:
		"""Ensure node bookkeeping exists."""
		if s not in self.N:
			self.N[s] = NodeStats(0, 0.0)
			self.parent[s] = None

	def _ucb(self, s: tuple[int, ...], a: int, actions_fn) -> float:
		"""Return UCB1 score for taking action a at state s."""
		child = s + (a,)
		ns = self.N.get(s, NodeStats(0, 0.0)).N
		nc = self.N.get(child, NodeStats(0, 0.0)).N
		if nc <= 0:
			return float("inf")
		wc = self.N[child].W
		mc = wc / max(1, nc)
		if ns > 0:
			bonus = self.c * math.sqrt(math.log(ns) / nc)
		else:
			bonus = float("inf")
		return mc + bonus

	def _select(self, s: tuple[int, ...], actions_fn) -> tuple[tuple[int, ...], list[tuple[int, ...]]]:
		"""Select a path by iteratively choosing UCB1-best children until a leaf or expansion frontier."""
		path: list[tuple[int, ...]] = [s]
		cur = s
		for _ in range(self.Dmax + 1):
			if len(cur) >= self.Dmax:
				return cur, path
			acts = actions_fn(cur, self.W)
			if len(acts) == 0:
				return cur, path

			if cur not in self.children:
				self.children[cur] = list(acts[: self.W])
				j = int(self.rng.integers(0, len(self.children[cur])))
				a0 = int(self.children[cur][j])
				nxt = cur + (a0,)
				self._ensure(nxt)
				self.parent[nxt] = cur
				path.append(nxt)
				cur = nxt
				continue  

			best_a = None
			best_v = None
			for a in self.children[cur]:
				v = self._ucb(cur, a, actions_fn)
				if best_v is None:
					best_v = v; best_a = a
				else:
					if v > best_v:
						best_v = v; best_a = a
					elif v == best_v:
						if int(self.rng.integers(0, 2)) == 1:
							best_v = v; best_a = a
			if best_a is None:
				return cur, path
			nxt = cur + (int(best_a),)
			self._ensure(nxt)
			self.parent[nxt] = cur
			path.append(nxt)
			cur = nxt
		return cur, path

	def _expand(self, s: tuple[int, ...], actions_fn) -> None:
		"""Expand a leaf by materializing its action set, up to W children (no reordering)."""
		if s in self.expanded:
			return
		acts = actions_fn(s, self.W)
		self.children[s] = list(acts[: self.W])
		self.expanded.add(s)


	def _rollout_policy(self, s: tuple[int, ...], actions_fn) -> list[int]:
		"""Return a sequence of actions drawn uniformly from available actions up to Broll steps or depth limit."""
		out: list[int] = []
		cur = s
		for _ in range(self.Broll):
			depth = len(cur)
			if depth >= self.Dmax:
				break
			acts = actions_fn(cur, self.W)
			if len(acts) == 0:
				break
			j = int(self.rng.integers(0, len(acts)))
			a = int(acts[j])
			out.append(a)
			cur = cur + (a,)
		return out


	def _simulate(self, s: tuple[int, ...], actions_fn, reward_fn) -> float:
		"""Simulate from s by expanding once and performing a default-policy rollout; return bounded reward in [0,1]."""
		self._expand(s, actions_fn)
		roll = self._rollout_policy(s, actions_fn)
		path = s + tuple(roll)
		r = float(reward_fn(path))
		if r < 0.0:
			return 0.0
		if r > 1.0:
			return 1.0
		return r

	def _backprop(self, path: list[tuple[int, ...]], r: float) -> None:
		"""Backpropagate reward r along the selected path."""
		for node in path:
			st = self.N.get(node, None)
			if st is None:
				self.N[node] = NodeStats(1, r)
			else:
				self.N[node] = NodeStats(st.N + 1, st.W + r)

	def run(self, actions_fn, reward_fn) -> None:
		"""
		Run Nsims simulations from the root state () using actions_fn and reward_fn.
		"""
		root: tuple[int, ...] = tuple()
		self._ensure(root)
		self.children[root] = list(actions_fn(root, self.W)[: self.W])
		self.expanded.add(root)
		for _ in range(self.Nsims):
			leaf, path_nodes = self._select(root, actions_fn)
			r = self._simulate(leaf, actions_fn, reward_fn)
			self._backprop(path_nodes, r)



	def best_action(self) -> int | None:
		"""
		Return the most-visited root action after run(), or None if no children.
		"""
		root = tuple()
		if root not in self.children:
			return None
		best_a = None
		best_n = None
		for a in self.children[root]:
			child = root + (a,)
			nc = self.N.get(child, NodeStats(0, 0.0)).N
			if best_n is None:
				best_n = nc
				best_a = a
			else:
				if nc > best_n:
					best_n = nc
					best_a = a
		return None if best_a is None else int(best_a)

	def visit_counts(self) -> dict[tuple[int, ...], int]:
		"""
		Return a mapping from node to visit count after run().
		"""
		out: dict[tuple[int, ...], int] = {}
		for k, st in self.N.items():
			out[k] = int(st.N)
		return out

	def mean_rewards(self) -> dict[tuple[int, ...], float]:
		"""
		Return a mapping from node to mean reward after run().
		"""
		out: dict[tuple[int, ...], float] = {}
		for k, st in self.N.items():
			if st.N > 0:
				out[k] = float(st.W / st.N)
			else:
				out[k] = 0.0
		return out

