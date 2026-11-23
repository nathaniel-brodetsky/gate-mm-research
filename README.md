# gate-mm-research

Исследование маркет-мейкинга на Gate. Приватный репозиторий.

## Стадии

Offline research → historical backtesting → forward data collection → shadow mode → paper trading →
малый реальный размер (только с отдельным письменным разрешением) → масштабирование (так же).

## Правила

- Гипотеза регистрируется в `hypotheses/` до эксперимента. Неудачные эксперименты не удаляются.
- Данные в `data/` не коммитятся.
- Секреты только из переменных окружения. В репозитории секретов нет.
- Торговые API не подключены. Ордера не размещаются.

## Установка и проверки

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy && .venv/bin/pytest
```

## Структура

| Путь | Назначение |
|---|---|
| `src/mmlab/` | код; по подпакету на компонент (см. docstring каждого `__init__.py`) |
| `tests/` | pytest |
| `hypotheses/` | реестр гипотез (`TEMPLATE.md`) |
| `configs/` | конфигурации экспериментов |
| `docs/` | отчёты этапов и технические решения |
| `scripts/` | CLI-точки входа |
| `data/` | локальные данные, вне git |
