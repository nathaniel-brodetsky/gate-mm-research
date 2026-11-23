# Stage 2 — Research dataset and first microstructure facts

Status: started 2026-11-23. No trading strategy in this stage.

1. Pre-register the research universe (D-001) and hypotheses H-001, H-002. *(this step)*
2. Run the selection; commit `configs/universe_U001.json`.
3. Dataset builder: download and validate archives for the research pairs, store events in Parquet
   with availability time (book: label + 100 ms), and exclusion masks from the coverage scan.
4. Descriptive facts and the tests of H-001 and H-002 on the development period, then validation.
5. Plan for a WebSocket recorder; where it runs depends on H-002.

Results are reported as a function of latency (1, 10, 100, 400 ms) wherever latency matters.
