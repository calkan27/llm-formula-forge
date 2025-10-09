# P16 — UCB1 in MCTS for Explanations: Sketch of Logarithmic Regret and the Role of `c`

We use UCB1 for child selection:
\[
\text{UCB1}(i) \;=\; \hat\mu_i \;+\; c\,\sqrt{\frac{\ln N}{n_i}}\!,
\]
where \( \hat\mu_i \) is the empirical mean reward of child \(i\), \(n_i\) is its visit count, and \(N\) is the visit count of the parent.

**Bounded rewards.** We clamp rewards to \([0,1]\), satisfying the bounded setting required by UCB1 analyses.

**Logarithmic regret (sketch).** In the i.i.d. K‑armed bandit setting with suboptimality gaps \(\Delta_i = \mu_* - \mu_i > 0\), the expected number of pulls of suboptimal arm \(i\) under UCB1 is \(O\!\left(\frac{\ln N}{\Delta_i^2}\right)\). Summed over arms, the expected regret is \(O\!\left(\sum_i \frac{\ln N}{\Delta_i} \right)\). In MCTS, UCB1 is applied *locally* to each parent’s children, so the same counting argument applies **per node** under the standard independence/rollout assumptions (see Kocsis & Szepesvári, 2006; Auer et al., 2002). Therefore visit allocation at each node concentrates on the optimal child with only logarithmic exploration cost in the number of simulations routed through that node.

**Role of \(c\).** The exploration weight \(c\) scales the confidence radius. Larger \(c\) increases exploration, raising early visitation of poorly sampled children; smaller \(c\) reduces exploration and becomes greedier. In the limit \(c\to 0\) the policy becomes purely exploitative based on current means; for large \(c\) the policy becomes more uniform until sufficient evidence accumulates.

**Depth and rollout truncation.** We cap search depth by \(D_{\max}\) and rollout horizon by \(B_{\mathrm{roll}}\), so the regret decomposes along the search path. The UCB1 allocation per node is unaffected by depth caps; it only changes how often nodes are visited.

**Determinism.** We use a fixed PRNG (PCG64) to make tie‑breaking reproducible, which does not change the asymptotic allocation guarantees.

