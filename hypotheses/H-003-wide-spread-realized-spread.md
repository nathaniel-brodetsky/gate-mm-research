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

None yet.
