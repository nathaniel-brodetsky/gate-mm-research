# Topological features: the lab first (stage 3, step 3b)

Status: lab written 2026-11-25; not yet run. This is the step of the stage 3 plan that was skipped
after the quote pauses: "do topological features detect regime changes better and earlier than
rolling volatility?" It is a question about a method, so it is answered on synthetic data with known
regimes before any feature is computed on the archive (rule 1 of the stage 3 plan).

## What to expect

- Published evidence is about slow, daily data. Gidea and Katz (Physica A, 2018;
  https://arxiv.org/abs/1703.04385) report that L^p norms of persistence landscapes of four US index
  returns rose ahead of the 2000 and 2008 crashes; a survey (https://arxiv.org/abs/1809.10745) notes
  that such norms grow with variance and cross-correlation, i.e. with what simple statistics already
  measure. Nothing published shows an edge at the scale of seconds that matters to a maker.
- On the spot pairs a regime detector has little to win: dropping each pair's three worst days, an
  ex-post bound, raises theta by 3.36 bp against −10.45 bp (step 3a). Regime features matter more
  where the fee is small (perpetuals: maker 0.02%, stage 4).

## Lab (`python -m mmlab.topology.regime_lab`)

A mid on a tick grid (latent 1-s changes of 0.35 ticks), one value per second, switches between two
regimes (exponential spells, mean 900 s); 80 paths of 4 hours per scenario.

| scenario | regime 1 differs from regime 0 by | role |
|---|---|---|
| volatility | 1.5 times the volatility | positive control for the baselines |
| reversion | changes load +0.3 instead of −0.3 on the EMA (30 s) of past changes; same 1-s variance | what a maker needs: reverting against continuing moves |
| jumps | 30% of the 1-s variance in jumps (0.5% of seconds); same 1-s variance | sweeps by informed takers |
| cycles | 10% of the 1-s variance in a 40-s cycle; same 1-s variance | positive control for topology |

Every 20 s after a 300-s warm-up, two trailing windows: 60 s and 300 s (61 values; the 300-s window
is embedded at 5-s steps). Label: the regime now.

- Baselines (`mmlab.feature_engineering.window_stats`): realized volatility, range, variance ratio and
  autocorrelation of the window's 1/12-length changes, efficiency ratio, largest change's share of
  the variance, share of seconds with a change.
- Topology (`mmlab.topology.persistence`, GUDHI 3.13): the swing diagram (H0 of the sublevel
  filtrations of the path and of its negative: each bar is a swing that reversed) gives the number of
  swings of at least 2 ticks, their entropy and the L1 norm of their landscape; the Rips diagram of
  the delay embedding (dimension 3) of the detrended, standardized window gives the total H0 length
  and the total, longest and landscape L1 norm of H1 (loops). The total swing length is left out: for
  a path it equals the total variation minus the range (tested), which the baselines already carry.
- Even paths fit ridge logistic models (baselines, topology, both); odd paths give the AUC. The
  bootstrap over test paths (1,000 draws) gives a 99% interval for the gain of both over baselines.
  AUC is also reported by the time since the last switch (how early a regime is seen).

Difficulty was set on the baselines alone before any topological feature was computed (seed 999, not
the seed of the run): with 70% of the variance in jumps and 50% in the cycle the baselines reached an
AUC of 0.93–0.98, leaving nothing to gain; with 30% and 10% they reach 0.76/0.83 (jumps) and
0.76/0.80 (cycles) at 60/300 s, reversion 0.58/0.70 and volatility 0.94/0.91 [синтетика, 40 paths].

## Rule (written before the run)

Topology goes on to the archive only if, in the reversion or the jumps scenario, at either window,
adding the topological features to the baselines raises the test AUC by at least 0.02 and the 99%
interval of that gain lies above 0. The volatility and cycles scenarios are controls and decide
nothing; if topology fails even in cycles, the features are not doing what they are built for.

- Passes: the passing features are computed as of 1 s before each trade on the archive and explored
  like step 4a (spot exploration period), always against the same baselines, and later on perpetuals.
- Fails: topological features of single price windows are closed for regime detection at these
  scales. Other topology (Mapper, correlation networks across pairs) only with a concrete hypothesis,
  e.g. together with lead-lag from liquid markets.
