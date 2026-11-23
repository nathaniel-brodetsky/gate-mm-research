# H-001: Liquidity providers on illiquid Gate spot pairs earn a realized spread above the maker fee

- **Status:** rejected
- **Registered:** 2026-11-23 (before any order book or trade data of the U-001 pairs was examined)
- **Decided:** 2026-11-23, on the development period (see Results)

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
| 1 | 2026-11-23 | Development, 24 research pairs, as registered | 2 of 24 reject after BH (12 required); median θ −9.63 bp; criterion not met | 35449bd |
| 2 | 2026-11-23 | Development, 23 pairs without SPYX_USDT (D-001 amendment 2) | 2 of 23 reject after BH; median θ −9.38 bp; criterion not met | 35449bd |

## Results

### Development period, 2026-08-01 – 2026-09-30

Full tables: [docs/results/h001-h002-development.md](../docs/results/h001-h002-development.md), generated
from `results/h001_h002/development.json` and `development_no_spyx.json`.

- 2 of 24 pairs reject H0 after BH at q = 0.10: NAVX_USDT (θ = +11.1 bp, p = 0.0013) and BOS_USDT
  (θ = +16.4 bp, p = 0.0042). 12 were required.
- The point estimate of θ is positive in 5 of 24 pairs. Cross-pair median θ = −9.63 bp: the median
  pair's RS̄(60 s) is +0.37 bp against a 10 bp fee.
- Without SPYX_USDT: 2 of 23 reject, median θ = −9.38 bp.

**Decision: rejected** under the registered criterion. Its development half fails, so the hypothesis
cannot be accepted whatever the validation period shows. The validation period is still run and
reported here; it does not change the decision. The holdout is opened once, together with H-002's,
because both hypotheses use the same markouts and H-002 is not decided yet (November trades are
published on 2026-12-01).

Scope: the liquidity that traded on these pairs, volume-weighted as a whole, did not earn the VIP 0
maker fee at 60 s. This does not test selective quoting (conditioning on book state or order flow),
lower fee tiers, or the two pairs that passed. Each of these needs its own hypothesis, registered
before it is tested on data not used here.
