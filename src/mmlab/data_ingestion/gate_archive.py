"""Availability probe for Gate public historical archives (download.gatedata.org).

Read-only: HEAD requests with a ranged-GET fallback. No credentials, no trading endpoints.

URL scheme (https://www.gate.com/developer/historical_quotes):
  monthly: {base}/{biz}/{type}/{YYYYMM}/{market}-{YYYYMM}.csv.gz
  hourly:  {base}/{biz}/orderbooks/{YYYYMM}/{market}-{YYYYMMDDHH}.csv.gz
           {base}/{biz}/orderbooks_slice/{YYYYMM}/{market}-{YYYYMMDDHH}.gz
All periods are UTC.

Usage:
  python -m mmlab.data_ingestion.gate_archive --biz spot --markets BTC_USDT,ABC_USDT \\
      --types deals,orderbooks --months 2023-01,2026-10 --days 15 --hours 0,12 --out probe.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from typing import Final

BASE_URL: Final = "https://download.gatedata.org"
BIZ: Final = ("spot", "futures_usdt", "futures_btc")
HOURLY_SUFFIX: Final = {"orderbooks": ".csv.gz", "orderbooks_slice": ".gz"}
USER_AGENT: Final = "mmlab-archive-probe/0.1"
_MARKET_RE: Final = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")
_TYPE_RE: Final = re.compile(r"^[a-z0-9_]+$")

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArchiveRef:
    biz: str
    type: str
    market: str
    period: str  # YYYYMM (monthly) or YYYYMMDDHH (hourly), UTC
    url: str


@dataclass(frozen=True)
class ProbeResult:
    biz: str
    type: str
    market: str
    period: str
    url: str
    status: int  # HTTP status; 0 = network error
    size_bytes: int  # -1 if unknown
    last_modified: str  # raw Last-Modified header, "" if absent
    method: str
    error: str

    @property
    def available(self) -> bool:
        return self.status in (200, 206)


def archive_ref(
    biz: str, type_: str, market: str, ym: str, day: int = 1, hour: int = 0, base: str = BASE_URL
) -> ArchiveRef:
    """Build the archive URL. `ym` is 'YYYYMM'; `day`/`hour` matter for hourly types only."""
    if biz not in BIZ:
        raise ValueError(f"unknown biz: {biz!r}")
    if not _TYPE_RE.match(type_):
        raise ValueError(f"bad type: {type_!r}")
    if not _MARKET_RE.match(market):
        raise ValueError(f"bad market: {market!r}")
    if len(ym) != 6 or not ym.isdigit():
        raise ValueError(f"bad month: {ym!r}")
    year, month = int(ym[:4]), int(ym[4:])
    base = base.rstrip("/")
    if type_ in HOURLY_SUFFIX:
        dt = datetime(year, month, day, hour)  # raises ValueError on impossible dates
        period = dt.strftime("%Y%m%d%H")
        url = f"{base}/{biz}/{type_}/{ym}/{market}-{period}{HOURLY_SUFFIX[type_]}"
        return ArchiveRef(biz, type_, market, period, url)
    datetime(year, month, 1)
    return ArchiveRef(biz, type_, market, ym, f"{base}/{biz}/{type_}/{ym}/{market}-{ym}.csv.gz")


def expand(
    biz: str,
    types: Sequence[str],
    markets: Sequence[str],
    months: Sequence[str],
    days: Sequence[int],
    hours: Sequence[int],
    base: str = BASE_URL,
) -> Iterator[ArchiveRef]:
    """All refs for the grid. Monthly types ignore days/hours."""
    for market in markets:
        for type_ in types:
            for ym in months:
                if type_ in HOURLY_SUFFIX:
                    for day in days:
                        for hour in hours:
                            yield archive_ref(biz, type_, market, ym, day, hour, base)
                else:
                    yield archive_ref(biz, type_, market, ym, base=base)


def _request(
    opener: urllib.request.OpenerDirector, url: str, method: str, timeout: float
) -> tuple[int, int, str]:
    headers = {"User-Agent": USER_AGENT}
    if method == "GET":
        headers["Range"] = "bytes=0-0"
    req = urllib.request.Request(url, method=method, headers=headers)
    try:
        with opener.open(req, timeout=timeout) as resp:
            status = int(resp.status)
            content_range = str(resp.headers.get("Content-Range") or "")
            content_length = str(resp.headers.get("Content-Length") or "")
            last_modified = str(resp.headers.get("Last-Modified") or "")
    except urllib.error.HTTPError as e:
        return int(e.code), -1, ""
    size = -1
    total = content_range.rsplit("/", 1)[-1] if "/" in content_range else ""
    if total.isdigit():
        size = int(total)
    elif method == "HEAD" and content_length.isdigit():
        size = int(content_length)
    return status, size, last_modified


def probe(
    ref: ArchiveRef, opener: urllib.request.OpenerDirector | None = None, timeout: float = 20.0
) -> ProbeResult:
    """HEAD first; ranged GET if HEAD is refused. Network errors are reported, not raised."""
    opener = opener or urllib.request.build_opener()
    method = "HEAD"
    try:
        status, size, last_modified = _request(opener, ref.url, method, timeout)
        if status in (403, 405, 501):
            method = "GET"
            status, size, last_modified = _request(opener, ref.url, method, timeout)
        error = ""
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        status, size, last_modified, error = 0, -1, "", f"{type(e).__name__}: {e}"
    return ProbeResult(
        **asdict(ref), status=status, size_bytes=size, last_modified=last_modified, method=method, error=error
    )


def _csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Probe Gate historical archive availability (read-only).")
    p.add_argument("--base", default=BASE_URL)
    p.add_argument("--biz", required=True, choices=BIZ)
    p.add_argument("--markets", required=True, help="comma-separated, e.g. BTC_USDT,ETH_USDT")
    p.add_argument(
        "--types",
        required=True,
        help="comma-separated: deals|trades|orderbooks|orderbooks_slice|candlesticks_1m|"
        "mark_prices|funding_applies|funding_updates",
    )
    p.add_argument("--months", required=True, help="comma-separated YYYY-MM")
    p.add_argument("--days", default="15", help="comma-separated UTC days, hourly types only")
    p.add_argument("--hours", default="0", help="comma-separated UTC hours, hourly types only")
    p.add_argument("--sleep", type=float, default=0.3, help="pause between requests, seconds")
    p.add_argument("--timeout", type=float, default=20.0)
    p.add_argument("--out", default="-", help="CSV path or '-' for stdout")
    return p


def main(argv: Sequence[str] | None = None, opener: urllib.request.OpenerDirector | None = None) -> int:
    """Returns 0 if at least one request reached the server, 1 if every request failed at network level."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    a = _parser().parse_args(argv)
    refs = list(
        expand(
            a.biz,
            _csv_list(a.types),
            _csv_list(a.markets),
            [m.replace("-", "") for m in _csv_list(a.months)],
            [int(d) for d in _csv_list(a.days)],
            [int(h) for h in _csv_list(a.hours)],
            a.base,
        )
    )
    results: list[ProbeResult] = []
    for i, ref in enumerate(refs):
        if i:
            time.sleep(a.sleep)
        results.append(probe(ref, opener, a.timeout))

    out = sys.stdout if a.out == "-" else open(a.out, "w", newline="")  # noqa: SIM115
    try:
        w = csv.DictWriter(out, fieldnames=[f.name for f in fields(ProbeResult)])
        w.writeheader()
        w.writerows(asdict(r) for r in results)
    finally:
        if out is not sys.stdout:
            out.close()

    n_ok = sum(r.available for r in results)
    n_net = sum(r.status == 0 for r in results)
    log.info("available %d/%d, network errors %d", n_ok, len(results), n_net)
    return 1 if results and n_net == len(results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
