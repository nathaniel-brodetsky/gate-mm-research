# H-003: On illiquid Gate spot pairs, trades at a quoted half-spread of at least 80 bp earn makers a realized spread above the maker fee

- **Status:** registered
- **Registered:** 2026-11-24 (after exploring 2026-08-01 – 2026-10-31 of the research pairs; before any
  November data or any data of the U-001 holdout pairs was examined)

## Economic rationale

H-001 found that the liquidity traded on these pairs, taken as a whole, earns about nothing before
fees. A maker does not have to quote all the time: the quoted spread is visible before a trade, and a
maker can quote only when it is wide. In the exploration (`results/h003_explore/explore.csv`, commit
d9c5356) the cross-pair median θ is positive only in the widest bucket, q ≥ 80 bp: +35.8 bp in the
development period (12 of 16 pairs positive) and +27.9 bp in the validation period (11 of 15). Every
narrower bucket of q and every bucket of the maker-side share ρ is negative in both periods. The bucket
was picked after looking at 22 buckets, so the exploration is not evidence; the hypothesis is tested
only on data that was not used to pick it.

This is a necessary condition, not a strategy:
- a new maker is filled only after the queue ahead of it at that price, and those fills may be more
  adversely selected than the average trade (to be measured with a queue-aware simulator);
- when the spread is wide the mid is not a price one can trade at, so inventory must be unwound,
  slowly as a maker or by paying the spread again as a taker (the exit markout below bounds this);
- capacity is small: in the exploration such trades came to about 32,000 USDT a day over all 24 pairs
  together.

## Mathematical formulation

For trade i: taker direction d_i, price p_i, quote value v_i and mid m(t_i) as in H-001, and the quoted
half-spread q_i = (a − b) / (2 m) of the last valid top of book known at t_i (a batch labelled τ is
known from τ + 100 ms). A trade is eligible if q_i ≥ 80 bp, with q in bp rounded to 6 decimals before
the comparison, as in the exploration.

Per pair P, over eligible trades with RS_i(60 s) defined:
θ_P = Σ v_i RS_i(60 s) / Σ v_i − f_m, with f_m = 0.10% (global spot VIP 0).

A pair counts if it has at least 30 such trades on at least 5 days of the test period.

- H-003: the cross-pair median of θ_P over counted pairs is positive.
- H0: it is at most 0.

## Data

Gate global spot archive (`orderbooks`, `deals`), built with `mmlab.market_data.build_dataset` under
the validity rules of H-001. Decision: U-001 research pairs. One-time estimate: U-001 holdout pairs.

## Features

q_i, with the threshold fixed at 80 bp.

## Target variable

RS_i(60 s).

## Forecast horizon

60 s. Reported, not tested: 10 s and 300 s.

## Baseline

θ = 0 at f_m = 0.10%. The unconditional θ of H-001 (all trades) on the same pairs and period is
reported alongside.

## Metric

Cross-pair median of θ_P with a 95% interval from a bootstrap over counted pairs (10,000 resamples,
seed 20261125). Per pair: θ_P with a 95% interval from a stationary block bootstrap over days (mean
block 3 days, 10,000 resamples, seed 20261126) and a one-sided p-value; Benjamini–Hochberg at
q = 0.10 across counted pairs, reported. Capacity: eligible quote value per calendar day, per pair and
summed, and θ_P times that value.

Sensitivity (reported, not tested): thresholds 40 and 120 bp; microprice instead of the mid;
per-trade median instead of the volume-weighted mean; exit markout
RS_exit,i(60 s) = d_i (p_i − x(t_i + 60 s)) / m(t_i), where x is the best ask for d_i = +1 and the best
bid for d_i = −1, i.e. the price at which the maker could close the position as a taker after 60 s
(before the taker fee).

## Success criterion (fixed before the experiment)

Test period 2026-11-01 – 2026-11-30 on the 24 research pairs.
- Accepted if at least 8 pairs count and the lower end of the 95% interval of the cross-pair median
  θ_P is above 0.
- Rejected if at least 8 pairs count and that lower end is at or below 0.
- If fewer than 8 pairs count, the test period is extended to 2026-11-01 – 2026-12-31 (December
  trades are published on 2027-01-01) and the same rule applies; if still fewer than 8, undecided.

After the decision the holdout pairs are opened once, over 2026-08-01 to the end of the test period,
to estimate θ; they do not revise the decision.

## Periods

- Exploration (used to pick the threshold): 2026-08-01 – 2026-10-31, research pairs
- Test: 2026-11-01 – 2026-11-30, research pairs (extension as above)
- Untouched holdout (opened once): U-001 holdout pairs, 2026-08-01 to the end of the test period

## Variants tested (all, including failed ones)

| # | Date | Variant | Result | Commit |
|---|---|---|---|---|
| 0 | 2026-11-24 | Exploration, not a test: 6 buckets of q and 5 of ρ, development, validation and both | Only q ≥ 80 bp has a positive median θ in both periods | d9c5356 |

## Results

### Power, from the synthetic lab (2026-11-24, before November was opened)

The lab (`results/lab/checks.json`) ran the registered rule on 800 synthetic Novembers calibrated per
pair on August–October. It accepts in 50.7% of months if every pair's August–October theta persists,
44.9% if it halves and 4.0% if it is absent (nominal 2.5%), with about 16 pairs counted. The spread of
theta across pairs, not the number of days, limits the power: a rejection says little against an
effect of the size seen in exploration.

The per-pair p-values reported with this test come from the registered day bootstrap, which the lab
found too liberal on these trades (11.7% rejections at a nominal 5% over 30 days). They are reported,
not used in the decision.

### Lab with real days, 2026-11-24 (before November was opened)

With synthetic Novembers rebuilt from real days of August–October (the only generator of the lab that
reproduces the real daily statistics; `docs/stage3-plan.md`, steps 2d–2g):

- The registered rule accepts in 45.3% of Novembers if every pair's exploration theta persists, 23.6%
  if it halves and 1.9% if it is absent (`results/lab/checks_days.json`).
- At the registered level of 2.5% it accepts in 2.4% of periods with every pair at 0 and 0.9% with
  pairs drawn at random around a median of 0 (`results/lab/levels_days.json`). 2.5% is the level
  [D-002](../docs/decisions/D-002-testing-standard.md) calibrates for this subset and window; the rule
  is unchanged.
- On wide-spread trades the days are skewed to the gain side in 12 of 17 pairs: the three heaviest days
  carry 54–63% of the absolute daily deviations for PEOPLEFB_USDT, NAVX_USDT and PINGO_USDT
  (`results/lab/tails_days.json`). An effect, if accepted, would come in a few days per pair; that
  matters for capacity and risk, not for the decision.

### How a result will be read: the execution simulator, 2026-11-25 (before November was opened)

H-003 is about the average trade's realized spread against the mid 60 s later. The queue-aware simulator
(`docs/stage3-plan.md`, step 6) shows what a new maker would get on the same trades in August–October:

- Joining the back of the queue at half-spreads of at least 80 bp: fills earn −8 to −24 bp, 40–49 bp below
  the average trade, under both queue models and at latencies of 100–1,000 ms.
- Stepping one tick inside: fills earn +4 to +11 bp, still 11–27 bp below the average trade.
- Full round trips, closing passively where possible and across the spread otherwise: −48 to −82 bp net,
  −28 to −62 bp before fees; 1–3 of 14–15 pairs positive.

So an acceptance of H-003 will mean that the liquidity in place at wide spreads earned its fee on average,
not that a new maker can earn it. The decision rule is unchanged.
