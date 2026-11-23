# H-001: Liquidity providers on illiquid Gate spot pairs earn a realized spread above the maker fee

- **Status:** registered
- **Registered:** 2026-11-23 (before any order book or trade data of the U-001 pairs was examined)

## Economic rationale

Makers are paid by takers who want immediacy. On illiquid pairs takers are plausibly mostly
uninformed (retail, portfolio moves, slow cross-venue arbitrage), so resting quotes should earn part of
the spread after adverse selection. If the liquidity that trades today does not earn more than the
VIP 0 maker fee, a new maker without queue priority has no edge. This is a necessary condition, not a
sufficient one: it says nothing yet about our own fill probability or queue position.

## Mathematical formulation

For trade i at time t_i: price p_i, taker direction d_i (+1 buy, −1 sell), quote value v_i, and mid
m(t) of the reconstructed book as known at t (a batch labelled τ is known from τ + 100 ms):

- effective half-spread ES_i = d_i (p_i − m(t_i)) / m(t_i)
- realized spread RS_i(h) = d_i (p_i − m(t_i + h)) / m(t_i)
- price impact PI_i(h) = d_i (m(t_i + h) − m(t_i)) / m(t_i), so ES_i = RS_i(h) + PI_i(h)

Per pair, RS̄(h) = Σ v_i RS_i(h) / Σ v_i. Maker fee f_m = 0.10% (global spot VIP 0, from 2026-04-09).

- H-001: RS̄(60 s) > f_m.
- H0: RS̄(60 s) ≤ f_m.

## Data

Gate global spot archive (`orderbooks`, `deals`), research pairs of U-001 (D-001). Only hours that are
internally clean and chained (stage 1 rules). A trade is dropped if t_i or t_i + h falls outside clean
data, or if the book is one-sided at either time.

## Features

None; this is a descriptive test. Inputs: trade price, quantity, taker side, reconstructed mid.

## Target variable

RS_i(h); ES_i and PI_i(h) are reported alongside.

## Forecast horizon

Primary h = 60 s. Reported, not tested: 1 s, 10 s, 300 s.

## Baseline

f_m = 0.10%. Sensitivity (reported, not tested): RS̄ with the microprice instead of the mid.

## Metric

Per pair: RS̄(60 s) − f_m with a 95% interval from a stationary block bootstrap over days (mean block
3 days, 10,000 resamples, seed 20261123) and a one-sided bootstrap p-value. Across pairs: number of
pairs rejecting H0 after Benjamini–Hochberg at q = 0.10, and the cross-pair median of RS̄(60 s) − f_m.

## Success criterion (fixed before the experiment)

Accepted if in the development period at least 12 of the 24 research pairs reject H0 after BH and the
cross-pair median of RS̄(60 s) − f_m is positive, and the same holds in the validation period.
Otherwise rejected. The holdout is then opened once to estimate the effect, not to revise the decision.

## Periods

- Development: 2026-08-01 – 2026-09-30
- Validation: 2026-10-01 – 2026-10-31
- Untouched holdout: 2026-11-01 – 2026-11-30 (trades published 2026-12-01), and the U-001 holdout pairs
  over all periods

## Variants tested (all, including failed ones)

| # | Date | Variant | Result | Commit |
|---|---|---|---|---|

## Results

None yet.
