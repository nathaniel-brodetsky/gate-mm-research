# gate-mm-research

Market making research on Gate. Private repository.

## Stages

Offline research → historical backtesting → forward data collection → shadow mode → paper trading →
small live size (only with separate written approval) → scaling (same).

## Rules

- Every hypothesis is registered in `hypotheses/` before the experiment. Failed experiments are never deleted.
- Data under `data/` is not committed.
- Secrets come from environment variables only. No secrets in the repository.
- No trading APIs are connected. No orders are placed.

## Setup and checks

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock && .venv/bin/python -m pip install -e . --no-deps
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest
```

## Layout

| Path | Purpose |
|---|---|
| `src/mmlab/` | code; one subpackage per component (see each `__init__.py` docstring) |
| `tests/` | pytest |
| `hypotheses/` | hypothesis registry (`TEMPLATE.md`) |
| `configs/` | experiment configurations |
| `docs/` | stage reports and technical decisions |
| `scripts/` | CLI entry points |
| `data/` | local data, not in git |
| `requirements.lock` | exact versions of all dependencies (`pip freeze`) |
