# Recurrent core acceptance

The deterministic causal and complete-state checks passed. This is an isolated mechanism check, not a navigation or AGI result.

- Real A → B → C transitions produce A, B, C, then silence without further sensory input or task reward.
- At the same state and weights, disabling disinhibition or recurrence stops the weak next-pattern activation.
- Multiple observed successors activate together. A learned two-pattern cycle persists for 12 blank steps; resetting activity stops it.
- External and remembered current have separately checked effects. Internal replay does not strengthen weights.
- A safe NPZ round trip continues 12 steps with exact arrays, flags, diagnostics and decay state.
- 16 versus 512 active sources give the same bounded recurrent-current maximum within floating-point tolerance.

8192 cells, 500 active sources, 1,500,000 excitatory edges and 39,000 disinhibitory edges: combined observe/advance median 12.52 ms, p95 13.85 ms. Allocated synapse storage 100.8 MB. This excludes the rest of the controller.

Numerical changes from the original raw-sum class are explicit: saturating Hebb efficacy, mean presynaptic current, normalized local site activity, leaky membrane, cell adaptation and bounded overall inhibition. Every nonzero learned edge is retained until uniform decay places it below the common disappearance threshold; there is no top-K connection deletion.

Default decay gives a weight half-life of about 27,657 actual observation steps without reinforcement. Fixed-width interference, mixture ambiguity, finite memory and actual embodied integration remain to be evaluated.
