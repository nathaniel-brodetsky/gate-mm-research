"""Reference grid: months, the as-of price by hand, a build from cached deals files, the guards."""

from __future__ import annotations

import gzip
import math
from decimal import Decimal
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from mmlab.data_ingestion.gate_archive import archive_ref
from mmlab.data_ingestion.gate_download import cache_path
from mmlab.data_ingestion.gate_trades_file import Trade
from mmlab.market_data import reference_grid as rg


def test_months() -> None:
    assert rg.months("2026-08-01", "2026-11-01") == ["202608", "202609", "202610"]
    assert rg.months("2026-08-15", "2026-08-16") == ["202608"]
    assert rg.months("2026-12-15", "2027-01-02") == ["202612", "202701"]
    assert rg.months("2026-12-01", "2027-01-01") == ["202612"]


def _t(sec: float, px: float) -> Trade:
    return Trade(round(sec * 1_000_000), 0, Decimal(str(px)), Decimal(1), "buy")


def test_grid_from_trades_by_hand() -> None:
    trades = [
        _t(98.0, 9),
        _t(99.5, 10),  # the seed: the last trade before second 100
        _t(97.0, 8),  # older, out of order: not the seed
        _t(100.2, 11),
        _t(100.7, 12),
        _t(102.1, 13),
        _t(102.05, 14),  # out of order inside second 102: not the last
        _t(105.0, 15),  # outside the grid
    ]
    price, count = rg.grid_from_trades(trades, 100, 5)
    assert price.tolist() == [10, 12, 12, 13, 13]
    assert count.tolist() == [2, 0, 2, 0, 0]
    price, _ = rg.grid_from_trades([_t(101.5, 7)], 100, 4)
    assert math.isnan(price[0])
    assert math.isnan(price[1])  # the trade in second 101 is known from second 102
    assert price[2:].tolist() == [7, 7]


def _deals(root: Path, pair: str, ym: str, rows: list[str]) -> None:
    path = cache_path(root, archive_ref("spot", "deals", pair, ym))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(("\n".join(rows) + "\n").encode()))


def test_build_from_cached_files_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    raw = tmp_path / "raw"
    g0 = 1_785_542_400  # 2026-08-01
    _deals(raw, "BTC_USDT", "202608", [f"{g0 + 5}.250000,1,100.5,0.1,1", f"{g0 + 86_400 + 7}.5,2,101,0.2,2"])
    out = tmp_path / "ref"
    argv = ["BTC_USDT", str(out), "--start", "2026-08-01", "--end", "2026-08-03", "--raw-dir", str(raw)]
    assert rg.main([*argv, "--base", "http://127.0.0.1:9"]) == 0  # cached: no request is made
    df = pl.read_parquet(out / "BTC_USDT.parquet")
    assert df.height == 2 * 86_400
    assert df["second"][0] == g0
    p = df["price_before"].to_numpy()
    assert np.isnan(p[:6]).all()
    assert p[6] == 100.5
    assert p[86_407] == 100.5
    assert p[86_408] == 101.0
    assert int(df["trades"].sum()) == 2
    assert "2 trades" in capsys.readouterr().out
    assert rg.main([*argv[:3], "2026-10-31", "--end", "2026-11-02"]) == 2  # time holdout
    assert "refused" in capsys.readouterr().out


def test_a_missing_month_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rg, "fetch", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="refusing to bridge"):
        rg.build("BTC_USDT", "2026-08-01", "2026-08-02", tmp_path)
