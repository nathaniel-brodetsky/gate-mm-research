# H-002: On illiquid Gate spot pairs most price impact after a trade arrives later than 1 s

- **Status:** rejected
- **Registered:** 2026-11-23 (before any order book or trade data of the U-001 pairs was examined)
- **Decided:** 2026-11-23, on the development and validation periods (see Results)

## Economic rationale

If the adverse move after a trade happens within milliseconds, the edge belongs to the fastest
participants and quotes managed from 10–400 ms away are picked off. If it builds over seconds to
minutes, latency is secondary and the strategy does not require co-location. The answer decides where
the recorder and any later execution must run, and which latency range results must be reported for.

## Mathematical formulation

With PI_i(h) and v_i as in H-001, the volume-weighted impact curve of a pair is
Λ(h) = Σ v_i PI_i(h) / Σ v_i, and the share of 5-minute impact reached after 1 s is
S = Λ(1 s) / Λ(300 s), defined for pairs with Λ(300 s) > 0.

- H-002: the cross-pair median of S is below 0.5.
- H0: the cross-pair median of S is at least 0.5.

Pairs with Λ(300 s) ≤ 0 are reported and excluded from the test.

## Data

As H-001: Gate global spot archive, U-001 research pairs, clean and chained hours only.

## Features

None; descriptive.

## Target variable

PI_i(h) for h in {0.1, 0.2, 0.5, 1, 2, 5, 10, 30, 60, 300} s.

## Forecast horizon

Test at h = 1 s against h = 300 s; the whole curve Λ(h) is reported.

## Baseline

Λ(h) = 0 for all h (no information in trades). Resolution limit: book batches are 100 ms and a batch
labelled τ is known from τ + 100 ms, so h = 0.1 s means "the next batch"; nothing below 100 ms can be
measured with this data.

## Metric

Per pair: S with a 95% interval from a stationary block bootstrap over days (mean block 3 days, 10,000
resamples, seed 20261124). Across pairs: median S with a 95% interval from a bootstrap over pairs.

## Success criterion (fixed before the experiment)

Accepted if the upper end of the 95% interval of the cross-pair median S is below 0.5 in the
development period and in the validation period. Rejected if the lower end is above 0.5 in both.
Otherwise undecided, and latency stays an open question for the recorder plan.

## Periods

- Development: 2026-08-01 – 2026-09-30
- Validation: 2026-10-01 – 2026-10-31
- Untouched holdout: 2026-11-01 – 2026-11-30, and the U-001 holdout pairs over all periods

## Variants tested (all, including failed ones)

| # | Date | Variant | Result | Commit |
|---|---|---|---|---|
| 1 | 2026-11-23 | Development, 24 research pairs, as registered | S defined for 24; median S 0.867, 95% CI [0.724, 0.939] | 35449bd |
| 2 | 2026-11-23 | Development, 23 pairs without SPYX_USDT (D-001 amendment 2) | median S 0.895, 95% CI [0.717, 0.945] | 35449bd |
| 3 | 2026-11-23 | Validation, 24 research pairs, as registered | S defined for 23; median S 0.758, 95% CI [0.700, 0.889] | b0881fb |
| 4 | 2026-11-23 | Validation, 23 pairs without SPYX_USDT | S defined for 22; median S 0.752, 95% CI [0.700, 0.889] | b0881fb |

## Results

### Development period, 2026-08-01 – 2026-09-30

Full tables, including the curve Λ(h) of every pair:
[docs/results/h001-h002-development.md](../docs/results/h001-h002-development.md).

- S is defined for all 24 pairs (Λ(300 s) > 0 in each). Cross-pair median S = 0.867, 95% CI
  [0.724, 0.939]. Without SPYX_USDT: 0.895, 95% CI [0.717, 0.945].
- Per pair, S ranges from 0.15 (SAITO_USDT) to 1.33 (ORCA_USDT). S > 1 in 3 pairs (BAND_USDT,
  XT_USDT, ORCA_USDT): there the impact at 300 s is smaller than at 1 s.

The lower end of the interval is above 0.5: the development half of the rejection criterion holds.

Median over the 24 pairs of Λ(h) / Λ(300 s): 0.70 at 0.1 s, 0.77 at 0.2 s, 0.86 at 0.5 s, 0.87 at
1 s, 0.86 at 10 s, 0.93 at 60 s. Median Λ(300 s) = 46.5 bp.

### Validation period, 2026-10-01 – 2026-10-31

Full tables: [docs/results/h001-h002-validation.md](../docs/results/h001-h002-validation.md).

- S is defined for 23 pairs; PEOPLEFB_USDT has Λ(300 s) ≤ 0 and is excluded, as registered.
  Cross-pair median S = 0.758, 95% CI [0.700, 0.889]. Without SPYX_USDT: 22 pairs, 0.752,
  95% CI [0.700, 0.889].
- Per pair, S ranges from 0.35 (SAITO_USDT) to 5.92 (ANON_USDT). Ratios above 1 come from a small
  Λ(300 s); the cross-pair median is not sensitive to them.

**Decision: rejected.** The lower end of the interval is above 0.5 in both periods. On these pairs
most of the 5-minute price impact of a trade is in the mid within 1 s.

Consequences, as set out in the economic rationale:
- About 70% of the 5-minute impact is already in the next 100 ms batch (development median). The
  archive cannot resolve anything faster, so the latency range 1–100 ms is not measurable from it.
  A recorder of finer data, with exchange timestamps, is needed before results can be reported for
  1 and 10 ms.
- A participant reacting 200–500 ms after a trade acts when about 77–86% of the impact is already in
  the price (development medians at 0.2 s and 0.5 s). From the owner's network (round trip about
  400 ms to the global API, stage 1) a reaction to a trade comes after most of its impact; execution
  that has to react to trades must run close to the matching engine.
- Part of the share at 0.1 s is the trade's own effect on the book (consumed levels). No speed avoids
  that for the order that was filled; S does not measure how often a fast maker cancels before being
  hit.
