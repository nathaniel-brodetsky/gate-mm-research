"""Probe of Gate's public spot WebSocket: record raw messages, then summarise timing and sequencing.

Documented behaviour to verify (Spot WebSocket v4, wss://api.gateio.ws/ws/v4/, read 2026-11-23):
- spot.trades: "realtime", taker side only; create_time_ms is a string that may carry decimals below 1 ms.
- spot.book_ticker: best bid and ask, update speed "10ms"; t = update time in ms, u = update id.
- spot.obu with ob.{pair}.50: 50 levels every 20 ms. The first push is a full book (full = true); an
  increment carries U..u and continues the book only if U = previous u + 1.
The documentation gives no connection or subscription limits and no server location.

Only public channels are used and no keys are read. Pairs of the research universe (research and
holdout) are refused, so that the probe cannot leak data of the untouched holdout.

Usage:
  python -m mmlab.data_ingestion.gate_ws_probe where
  python -m mmlab.data_ingestion.gate_ws_probe record --pairs BTC_USDT,LUNC_USDT --seconds 600 \\
      --out data/raw/ws_probe/probe.jsonl.gz
  python -m mmlab.data_ingestion.gate_ws_probe summarize data/raw/ws_probe/probe.jsonl.gz

File format (gzip, one JSON object per line):
  {"k": "msg", "r": <local receive time, ns since epoch>, "c": <connection number>, "d": <raw text>}
  {"k": "meta", "r": <ns>, "c": <connection number>, "event": "...", ...}
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import ipaddress
import itertools
import json
import math
import os
import socket
import sys
import time
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

WS_URL = "wss://api.gateio.ws/ws/v4/"
API_HOST = "api.gateio.ws"
AWS_RANGES_URL = "https://ip-ranges.amazonaws.com/ip-ranges.json"
USER_AGENT = "mmlab-ws-probe/0.1"
OBU_LEVELS = 50  # 20 ms stream; 400 levels come every 100 ms


def subscriptions(pairs: Sequence[str], now_s: int) -> list[dict[str, Any]]:
    reqs: list[dict[str, Any]] = [
        {"time": now_s, "channel": "spot.trades", "event": "subscribe", "payload": list(pairs)},
        {"time": now_s, "channel": "spot.book_ticker", "event": "subscribe", "payload": list(pairs)},
    ]
    reqs += [
        {"time": now_s, "channel": "spot.obu", "event": "subscribe", "payload": [f"ob.{p}.{OBU_LEVELS}"]}
        for p in pairs
    ]
    return reqs


def _write(fh: IO[str], obj: dict[str, Any]) -> None:
    fh.write(json.dumps(obj, separators=(",", ":")) + "\n")


async def _pinger(ws: ClientConnection, fh: IO[str], conn: int, every_s: float, timeout_s: float) -> None:
    while True:
        await asyncio.sleep(every_s)
        try:
            waiter = await ws.ping()
            rtt = await asyncio.wait_for(waiter, timeout_s)
        except TimeoutError:
            _write(fh, {"k": "meta", "r": time.time_ns(), "c": conn, "event": "ping_timeout"})
            await ws.close()
            return
        _write(fh, {"k": "meta", "r": time.time_ns(), "c": conn, "event": "rtt", "ms": round(rtt * 1e3, 3)})


async def record(
    url: str,
    pairs: Sequence[str],
    seconds: float,
    out: Path,
    ping_every_s: float = 5.0,
    ping_timeout_s: float = 20.0,
    backoff_s: float = 1.0,
) -> int:
    """Record raw messages until `seconds` have passed, reconnecting after failures. Returns connections."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    deadline = time.monotonic() + seconds
    conn = 0
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        while time.monotonic() < deadline:
            conn += 1
            try:
                async with connect(
                    url, proxy=None, ping_interval=None, max_size=2**22, open_timeout=10
                ) as ws:
                    _write(fh, {"k": "meta", "r": time.time_ns(), "c": conn, "event": "connect"})
                    for req in subscriptions(pairs, int(time.time())):
                        await ws.send(json.dumps(req))
                    pinger = asyncio.create_task(_pinger(ws, fh, conn, ping_every_s, ping_timeout_s))
                    try:
                        while (left := deadline - time.monotonic()) > 0:
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=left)
                            except TimeoutError:
                                break
                            text = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                            _write(fh, {"k": "msg", "r": time.time_ns(), "c": conn, "d": text})
                    finally:
                        pinger.cancel()
                _write(fh, {"k": "meta", "r": time.time_ns(), "c": conn, "event": "close"})
            except (OSError, TimeoutError, WebSocketException) as e:
                err = f"{type(e).__name__}: {e}"
                _write(fh, {"k": "meta", "r": time.time_ns(), "c": conn, "event": "disconnect", "error": err})
                await asyncio.sleep(max(0.0, min(backoff_s, deadline - time.monotonic())))
    os.replace(tmp, out)
    return conn


@dataclass
class Stream:
    """One channel for one pair."""

    channel: str
    pair: str
    messages: int = 0
    t_ms: list[float] = field(default_factory=list)  # exchange time of each message
    delay_ms: list[float] = field(default_factory=list)  # local receive time - exchange time
    snapshots: int = 0  # spot.obu full books
    seq_breaks: int = 0  # spot.obu: U != previous u + 1; book_ticker: u decreasing
    before_snapshot: int = 0  # spot.obu increments before the first full book on a connection
    submillisecond: int = 0  # spot.trades with decimals in create_time_ms
    levels: list[int] = field(default_factory=list)  # spot.obu levels changed per increment


@dataclass
class Summary:
    streams: dict[tuple[str, str], Stream] = field(default_factory=dict)
    connects: int = 0
    disconnects: list[str] = field(default_factory=list)
    rtt_ms: list[float] = field(default_factory=list)
    ping_timeouts: int = 0
    sub_ok: int = 0
    sub_errors: list[str] = field(default_factory=list)
    other: int = 0
    first_r_ns: int | None = None
    last_r_ns: int | None = None


def _stream(s: Summary, channel: str, pair: str) -> Stream:
    return s.streams.setdefault((channel, pair), Stream(channel, pair))


def summarize(path: Path) -> Summary:
    s = Summary()
    last_u: dict[tuple[int, str], int] = {}  # (connection, stream) -> last update id
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            r_ns: int = rec["r"]
            s.first_r_ns = r_ns if s.first_r_ns is None else s.first_r_ns
            s.last_r_ns = r_ns
            if rec["k"] == "meta":
                ev = rec["event"]
                if ev == "connect":
                    s.connects += 1
                elif ev == "disconnect":
                    s.disconnects.append(rec.get("error", ""))
                elif ev == "rtt":
                    s.rtt_ms.append(float(rec["ms"]))
                elif ev == "ping_timeout":
                    s.ping_timeouts += 1
                continue
            _message(s, last_u, rec["c"], r_ns / 1e6, json.loads(rec["d"]))
    return s


def _message(s: Summary, last_u: dict[tuple[int, str], int], conn: int, recv_ms: float, m: Any) -> None:
    channel, event, res = m.get("channel"), m.get("event"), m.get("result")
    if event == "subscribe":
        if m.get("error") or not isinstance(res, dict) or res.get("status") != "success":
            s.sub_errors.append(f"{channel} {m.get('payload')}: {m.get('error')}")
        else:
            s.sub_ok += 1
        return
    if not isinstance(res, dict):
        s.other += 1
        return
    if channel == "spot.trades":
        st = _stream(s, channel, res["currency_pair"])
        t = float(res["create_time_ms"])
        st.submillisecond += "." in str(res["create_time_ms"])
    elif channel == "spot.book_ticker":
        st = _stream(s, channel, res["s"])
        t = float(res["t"])
        key = (conn, channel + res["s"])
        if key in last_u and int(res["u"]) < last_u[key]:
            st.seq_breaks += 1
        last_u[key] = int(res["u"])
    elif channel == "spot.obu":
        pair = res["s"].split(".")[1]
        st = _stream(s, channel, pair)
        t = float(res["t"])
        key = (conn, res["s"])
        if res.get("full"):
            st.snapshots += 1
        elif key not in last_u:
            st.before_snapshot += 1
        else:
            if int(res["U"]) != last_u[key] + 1:
                st.seq_breaks += 1
            st.levels.append(len(res.get("b") or []) + len(res.get("a") or []))
        last_u[key] = int(res["u"])
    else:
        s.other += 1
        return
    st.messages += 1
    st.t_ms.append(t)
    st.delay_ms.append(recv_ms - t)


def pct(xs: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile, q in [0, 100] (rank q/100 * (n - 1), halves rounded up)."""
    if not xs:
        return None
    v = sorted(xs)
    return v[min(len(v) - 1, max(0, math.floor(q / 100 * (len(v) - 1) + 0.5)))]


def gaps(t_ms: Sequence[float]) -> list[float]:
    """Exchange-time steps between consecutive messages; 0 = same millisecond, backward steps dropped."""
    return [b - a for a, b in itertools.pairwise(t_ms) if b >= a]


def _f(x: float | None, nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def report(s: Summary) -> str:
    span = 0.0 if s.first_r_ns is None or s.last_r_ns is None else (s.last_r_ns - s.first_r_ns) / 1e9
    lines = [
        f"recorded {span:.0f} s; connections {s.connects}, disconnects {len(s.disconnects)}, "
        f"ping timeouts {s.ping_timeouts}; subscriptions ok {s.sub_ok}, failed {len(s.sub_errors)}; "
        f"other messages {s.other}",
        f"ws ping rtt ms: min {_f(min(s.rtt_ms) if s.rtt_ms else None)} p50 {_f(pct(s.rtt_ms, 50))} "
        f"max {_f(max(s.rtt_ms) if s.rtt_ms else None)} (n {len(s.rtt_ms)})",
        *[f"  disconnect: {e}" for e in s.disconnects[:5]],
        *[f"  subscription failed: {e}" for e in s.sub_errors[:5]],
        "delay = local receive time - exchange time (includes the local clock offset); "
        "gap = exchange-time step between consecutive messages",
        f"{'channel':17} {'pair':12} {'msgs':>7} {'gap p10':>8} {'gap p50':>8} {'delay p5':>9} "
        f"{'p50':>7} {'p95':>7}  notes",
    ]
    for (_, _), st in sorted(s.streams.items()):
        g = gaps(st.t_ms)
        notes = []
        if st.channel == "spot.obu":
            notes.append(f"full {st.snapshots}, seq breaks {st.seq_breaks}, before full {st.before_snapshot}")
            notes.append(f"levels/update p50 {_f(pct(st.levels, 50), 0)}")
        elif st.channel == "spot.trades":
            notes.append(f"sub-ms timestamps {st.submillisecond}/{st.messages}")
        else:
            notes.append(f"u decreasing {st.seq_breaks}")
        lines.append(
            f"{st.channel:17} {st.pair:12} {st.messages:7d} {_f(pct(g, 10)):>8} {_f(pct(g, 50)):>8} "
            f"{_f(pct(st.delay_ms, 5)):>9} {_f(pct(st.delay_ms, 50)):>7} {_f(pct(st.delay_ms, 95)):>7}  "
            + "; ".join(notes)
        )
    return "\n".join(lines)


def aws_regions(ips: Sequence[str], ranges: dict[str, Any]) -> dict[str, str]:
    """Longest AWS prefix per IP, as 'region service border-group', or 'not in AWS ranges'."""
    nets = [(ipaddress.ip_network(p["ip_prefix"]), p) for p in ranges.get("prefixes", [])]
    nets += [(ipaddress.ip_network(p["ipv6_prefix"]), p) for p in ranges.get("ipv6_prefixes", [])]
    out = {}
    for ip in ips:
        addr = ipaddress.ip_address(ip)
        hits = [(n, p) for n, p in nets if n.version == addr.version and addr in n]
        if not hits:
            out[ip] = "not in AWS ranges"
            continue
        best = max(hits, key=lambda h: h[0].prefixlen)[1]
        out[ip] = f"{best['region']} {best['service']} {best.get('network_border_group', '')}".strip()
        out[ip] += f" (+{len({p['service'] for _, p in hits}) - 1} more services)" if len(hits) > 1 else ""
    return out


def where(host: str = API_HOST) -> str:
    infos = socket.getaddrinfo(host, 443, proto=socket.IPPROTO_TCP)
    ips = sorted({str(i[4][0]) for i in infos})
    req = urllib.request.Request(AWS_RANGES_URL, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        ranges = json.load(r)
    regions = aws_regions(ips, ranges)
    lines = [f"{host}: {len(ips)} addresses; AWS ip-ranges createDate {ranges.get('createDate')}"]
    for ip in ips:
        try:
            rdns = socket.gethostbyaddr(ip)[0]
        except OSError:
            rdns = "no reverse DNS"
        rtts = []
        for _ in range(3):
            t0 = time.perf_counter()
            try:
                with socket.create_connection((ip, 443), timeout=5):
                    rtts.append((time.perf_counter() - t0) * 1e3)
            except OSError:
                pass
        rtt = f"{sorted(rtts)[len(rtts) // 2]:.0f} ms" if rtts else "no TCP connect"
        lines.append(f"  {ip:40} {regions[ip]:40} {rdns}  tcp connect {rtt}")
    return "\n".join(lines)


def universe_pairs(path: Path) -> set[str]:
    u = json.loads(path.read_text())
    return {p for part in ("research", "holdout") for ps in u.get(part, {}).values() for p in ps}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Gate public spot WebSocket probe.")
    sub = p.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("where", help="resolve the API host and look its addresses up in AWS ip-ranges")
    w.add_argument("--host", default=API_HOST)
    r = sub.add_parser("record", help="record raw public messages")
    r.add_argument("--pairs", required=True, help="comma-separated, e.g. BTC_USDT,LUNC_USDT")
    r.add_argument("--seconds", type=float, default=600.0)
    r.add_argument("--out", required=True)
    r.add_argument("--url", default=WS_URL)
    r.add_argument("--universe", default="configs/universe_U001.json")
    s = sub.add_parser("summarize", help="timing and sequencing summary of a recording")
    s.add_argument("path")
    a = p.parse_args(argv)
    if a.cmd == "where":
        print(where(a.host))
        return 0
    if a.cmd == "summarize":
        print(report(summarize(Path(a.path))))
        return 0
    pairs = [x.strip() for x in a.pairs.split(",") if x.strip()]
    blocked = sorted(set(pairs) & universe_pairs(Path(a.universe)))
    if blocked:
        print(f"refused: {blocked} belong to the research universe ({a.universe})")
        return 2
    conns = asyncio.run(record(a.url, pairs, a.seconds, Path(a.out)))
    print(f"recorded {a.out} over {conns} connection(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
