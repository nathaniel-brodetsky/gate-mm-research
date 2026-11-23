# Results

Outputs of registered tests, committed unchanged, so that every number in the hypothesis files and
reports can be traced to a file and a code commit. Reports in `docs/results/` are generated from these
files with `python -m mmlab.reporting.h001_h002_report`.

| File | Command | Code commit |
|---|---|---|
| `h001_h002/development.json` | `python -m mmlab.evaluation.h001_h002 data/processed/U001 --period development` | 35449bd |
| `h001_h002/development_no_spyx.json` | the same with `--exclude SPYX_USDT` (D-001 amendment 2) | 35449bd |

The dataset `data/processed/U001` is not committed. It was built on 2026-11-23 with
`python -m mmlab.market_data.build_dataset configs/universe_U001.json data/processed/U001 --start
2026-08-01 --end 2026-11-01`: 15 pairs before 35449bd and 9 with it (that commit changed retries and
memory use, not the output). SUKU_USDT rebuilt at 35449bd is identical to its first build (top of
book, trades and hourly checks).
