# Stage 2 — Research dataset and first microstructure facts

Status: started 2026-11-23. No trading strategy in this stage.

1. Pre-register the research universe (D-001) and hypotheses H-001, H-002. *(done)*
2. Run the selection; commit `configs/universe_U001.json`. *(done)*
3. Asset metadata for research pairs; top of book per batch with availability time and validity.
   *(done; SPYX_USDT flagged, D-001 amendment 2)*
4. Dataset builder: download and validate archives for the research pairs, store top of book and
   trades in Parquet with gap sentinels. *(done: 24 pairs, 2026-08-01 – 2026-10-31, 2208 hours each,
   none missing; see [results/README.md](../results/README.md))*
5. Descriptive facts and the tests of H-001 and H-002 on the development period, then validation.
   *(this step; development done: H-001 rejected, H-002 waits for validation;
   [report](results/h001-h002-development.md))*
6. Plan for a WebSocket recorder; where it runs depends on H-002.

Results are reported as a function of latency (1, 10, 100, 400 ms) wherever latency matters.

Implementation of the tests (step 5): `mmlab.microstructure.markouts` (as-of joins on availability
time, validity enforced) and `mmlab.evaluation.h001_h002` (block bootstrap over days, BH across pairs).
H-002's ratio S is computed on trades usable at both 1 s and 300 s, so that both horizons describe the
same trades.
