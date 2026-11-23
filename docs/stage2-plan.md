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
   *(done: H-001 rejected, H-002 rejected; reports for [development](results/h001-h002-development.md)
   and [validation](results/h001-h002-validation.md))*
6. Plan for a WebSocket recorder; where it runs depends on H-002. *(next. H-002 is rejected: most
   impact arrives within 1 s, about 70% within the first 100 ms batch, so the recorder must capture
   order book and trade events finer than 100 ms, with exchange timestamps)*
7. Holdout for H-001 and H-002, opened once after 2026-12-01 (November trades): the 12 holdout pairs
   over 2026-08-01 – 2026-11-30 and the research pairs in November. It estimates the effects; the
   decisions stand.

Results are reported as a function of latency (1, 10, 100, 400 ms) wherever latency matters.

Implementation of the tests (step 5): `mmlab.microstructure.markouts` (as-of joins on availability
time, validity enforced) and `mmlab.evaluation.h001_h002` (block bootstrap over days, BH across pairs).
H-002's ratio S is computed on trades usable at both 1 s and 300 s, so that both horizons describe the
same trades.
