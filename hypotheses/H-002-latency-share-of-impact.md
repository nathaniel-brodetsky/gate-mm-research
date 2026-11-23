# H-002: On illiquid Gate spot pairs most price impact after a trade arrives later than 1 s

- **Status:** registered
- **Registered:** 2026-11-23 (before any order book or trade data of the U-001 pairs was examined)

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

## Results

None yet.
