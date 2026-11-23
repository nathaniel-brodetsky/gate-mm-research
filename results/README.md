# Results

Outputs of registered tests, committed unchanged, so that every number in the hypothesis files and
reports can be traced to a file and a code commit. Reports in `docs/results/` are generated from these
files with `python -m mmlab.reporting.h001_h002_report`.

Evaluation code (`mmlab.evaluation`, `mmlab.microstructure`) is the same at 35449bd and b0881fb.

| File | Command | Code commit |
|---|---|---|
| `h001_h002/development.json` | `python -m mmlab.evaluation.h001_h002 data/processed/U001 --period development` | 35449bd |
| `h001_h002/development_no_spyx.json` | the same with `--exclude SPYX_USDT` (D-001 amendment 2) | 35449bd |
| `h001_h002/validation.json` | `python -m mmlab.evaluation.h001_h002 data/processed/U001 --period validation` | b0881fb |
| `h001_h002/validation_no_spyx.json` | the same with `--exclude SPYX_USDT` | b0881fb |

`ws_probe/probe-20261123T220212Z.txt` is the output of `python -m mmlab.data_ingestion.gate_ws_probe
summarize` on the 10-minute recording of 2026-11-23 (code at 58e6296); the raw recording stays in
`data/raw/ws_probe/`.

The dataset `data/processed/U001` is not committed. It was built on 2026-11-23 with
`python -m mmlab.market_data.build_dataset configs/universe_U001.json data/processed/U001 --start
2026-08-01 --end 2026-11-01`: 15 pairs before 35449bd and 9 with it (that commit changed retries and
memory use, not the output). SUKU_USDT rebuilt at 35449bd is identical to its first build (top of
book, trades and hourly checks).
