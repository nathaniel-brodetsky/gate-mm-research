"""WebSocket probe against a local server that speaks the documented Gate message shapes."""

from __future__ import annotations

import asyncio
import gzip
import json
import socket
import time
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.server import ServerConnection, serve

from mmlab.data_ingestion import gate_ws_probe as wp

LAG_MS = 50.0  # exchange time stamped this long before the push


def _env(channel: str, result: dict[str, Any]) -> str:
    return json.dumps({"time": 1, "time_ms": 1000, "channel": channel, "event": "update", "result": result})


def _pushes() -> list[str]:
    t0 = time.time() * 1e3 - LAG_MS
    out = [
        _env("spot.book_ticker", {"t": t0 + 10 * i, "u": 100 + i, "s": "AAA_USDT", "b": "1", "a": "2"})
        for i in range(5)
    ]
    out += [
        _env(
            "spot.obu", {"t": t0, "full": True, "s": "ob.AAA_USDT.50", "u": 500, "b": [["1", "1"]], "a": []}
        ),
        _env(
            "spot.obu", {"t": t0 + 20, "s": "ob.AAA_USDT.50", "U": 501, "u": 503, "b": [["1", "2"]], "a": []}
        ),
        _env(  # U should be 504: one sequence break
            "spot.obu",
            {"t": t0 + 40, "s": "ob.AAA_USDT.50", "U": 505, "u": 506, "b": [], "a": [["2", "0"], ["3", "1"]]},
        ),
        _env("spot.trades", {"id": 1, "create_time_ms": f"{t0 + 5:.0f}.25", "currency_pair": "AAA_USDT"}),
        _env("spot.trades", {"id": 2, "create_time_ms": f"{t0 + 6:.0f}.5", "currency_pair": "AAA_USDT"}),
        _env("spot.trades", {"id": 3, "create_time_ms": f"{t0 + 7:.0f}", "currency_pair": "AAA_USDT"}),
        json.dumps({"time": 1, "channel": "spot.pong", "event": "", "error": None, "result": None}),
    ]
    return out


async def _run_probe(tmp: Path, seconds: float) -> tuple[Path, int, list[dict[str, Any]]]:
    connections = 0
    requests: list[dict[str, Any]] = []

    async def handler(ws: ServerConnection) -> None:
        nonlocal connections
        connections += 1
        for _ in range(3):  # trades, book_ticker, one obu stream
            req = json.loads(await ws.recv())
            requests.append(req)
            ok = {"time": 1, "channel": req["channel"], "event": "subscribe", "result": {"status": "success"}}
            await ws.send(json.dumps(ok))
        for m in _pushes():
            await ws.send(m)
        if connections == 1:
            await ws.close()  # the probe must reconnect
        else:
            await ws.wait_closed()

    out = tmp / "probe.jsonl.gz"
    async with serve(handler, "127.0.0.1", 0) as server:
        port = next(iter(server.sockets)).getsockname()[1]
        n = await wp.record(
            f"ws://127.0.0.1:{port}", ["AAA_USDT"], seconds, out, ping_every_s=0.1, backoff_s=0.05
        )
    return out, n, requests


def test_record_and_summarize(tmp_path: Path) -> None:
    out, n, requests = asyncio.run(_run_probe(tmp_path, 1.0))
    assert n == 2
    assert not (tmp_path / "probe.jsonl.gz.part").exists()
    assert [r["channel"] for r in requests[:3]] == ["spot.trades", "spot.book_ticker", "spot.obu"]
    assert requests[2]["payload"] == ["ob.AAA_USDT.50"]
    assert abs(requests[0]["time"] - time.time()) < 60  # Gate rejects requests more than 60 s off

    s = wp.summarize(out)
    assert (s.connects, len(s.disconnects), s.sub_ok, s.sub_errors, s.other) == (2, 1, 6, [], 2)
    assert "ConnectionClosed" in s.disconnects[0]
    assert len(s.rtt_ms) >= 3
    assert all(0 < x < 1000 for x in s.rtt_ms)  # milliseconds
    assert s.ping_timeouts == 0

    bt = s.streams[("spot.book_ticker", "AAA_USDT")]
    assert (bt.messages, bt.seq_breaks) == (10, 0)  # u restarts on the new connection: not a break
    assert wp.pct(wp.gaps(bt.t_ms), 50) == pytest.approx(10)
    # stamped 50, 40, ..., 10 ms before the push: delay = receive time - exchange time
    assert min(bt.delay_ms) > LAG_MS - 40 - 1
    assert max(bt.delay_ms) < LAG_MS + 2000
    assert wp.pct(bt.delay_ms, 50) == pytest.approx(sorted(bt.delay_ms)[5])

    ob = s.streams[("spot.obu", "AAA_USDT")]
    assert (ob.messages, ob.snapshots, ob.seq_breaks, ob.before_snapshot) == (6, 2, 2, 0)
    assert sorted(ob.levels) == [1, 1, 2, 2]

    tr = s.streams[("spot.trades", "AAA_USDT")]
    assert (tr.messages, tr.submillisecond) == (6, 4)

    text = wp.report(s)
    assert "connections 2, disconnects 1" in text
    assert "full 2, seq breaks 2, before full 0" in text
    assert "sub-ms timestamps 4/6" in text


def test_summary_flags_increments_before_snapshot_and_failed_subscriptions(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl.gz"
    msgs = [
        {
            "time": 1,
            "channel": "spot.obu",
            "event": "subscribe",
            "error": {"code": 2, "message": "Alert sub"},
            "result": {"status": "fail"},
            "payload": ["ob.AAA_USDT.50"],
        },
        {
            "channel": "spot.obu",
            "result": {"t": 1000.0, "s": "ob.AAA_USDT.50", "U": 7, "u": 9, "b": [], "a": []},
        },
        {
            "channel": "spot.obu",
            "result": {"t": 1020.0, "s": "ob.AAA_USDT.50", "U": 10, "u": 11, "b": [], "a": []},
        },
        {"channel": "spot.book_ticker", "event": "update", "result": {"t": 1000, "u": 5, "s": "AAA_USDT"}},
        {"channel": "spot.book_ticker", "event": "update", "result": {"t": 1010, "u": 4, "s": "AAA_USDT"}},
        {"channel": "spot.book_ticker", "event": "update", "result": {"t": 1010, "u": 4, "s": "AAA_USDT"}},
    ]
    with gzip.open(path, "wt") as fh:
        fh.write(json.dumps({"k": "meta", "r": 0, "c": 1, "event": "connect"}) + "\n")
        fh.write(json.dumps({"k": "meta", "r": 0, "c": 1, "event": "ping_timeout"}) + "\n")
        for m in msgs:
            fh.write(json.dumps({"k": "msg", "r": 2_000_000_000, "c": 1, "d": json.dumps(m)}) + "\n")
    s = wp.summarize(path)
    ob = s.streams[("spot.obu", "AAA_USDT")]
    assert (ob.before_snapshot, ob.seq_breaks, ob.snapshots) == (1, 0, 0)  # 10 = 9 + 1 continues
    assert s.streams[("spot.book_ticker", "AAA_USDT")].seq_breaks == 1  # 5 -> 4; a repeated 4 is not
    assert s.sub_errors == ["spot.obu ['ob.AAA_USDT.50']: {'code': 2, 'message': 'Alert sub'}"]
    assert s.ping_timeouts == 1
    assert s.streams[("spot.book_ticker", "AAA_USDT")].delay_ms == [1000.0, 990.0, 990.0]


def test_unreachable_server_is_recorded_not_raised(tmp_path: Path) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]  # closed again before the probe connects
    out = tmp_path / "down.jsonl.gz"
    n = asyncio.run(wp.record(f"ws://127.0.0.1:{port}", ["AAA_USDT"], 0.3, out, backoff_s=0.1))
    s = wp.summarize(out)
    assert 2 <= n <= 5  # one attempt per 0.1 s backoff
    assert (s.connects, len(s.disconnects)) == (0, n)
    assert all("Error" in e for e in s.disconnects)


def test_aws_regions_longest_prefix() -> None:
    ranges = {
        "prefixes": [
            {
                "ip_prefix": "52.0.0.0/8",
                "region": "GLOBAL",
                "service": "AMAZON",
                "network_border_group": "GLOBAL",
            },
            {
                "ip_prefix": "52.68.0.0/15",
                "region": "ap-northeast-1",
                "service": "EC2",
                "network_border_group": "ap-northeast-1",
            },
        ],
        "ipv6_prefixes": [
            {
                "ipv6_prefix": "2406:da14::/32",
                "region": "ap-northeast-1",
                "service": "EC2",
                "network_border_group": "ap-northeast-1",
            },
        ],
    }
    out = wp.aws_regions(["52.69.1.2", "52.1.1.1", "8.8.8.8", "2406:da14::1"], ranges)
    assert out["52.69.1.2"] == "ap-northeast-1 EC2 ap-northeast-1 (+1 more services)"
    assert out["52.1.1.1"] == "GLOBAL AMAZON GLOBAL"
    assert out["8.8.8.8"] == "not in AWS ranges"
    assert out["2406:da14::1"] == "ap-northeast-1 EC2 ap-northeast-1"


def test_gaps() -> None:
    assert wp.gaps([1000.0, 1010.0, 1010.0, 990.0, 1030.0]) == [10.0, 0.0, 40.0]
    assert wp.gaps([5.0]) == []


def test_pct() -> None:
    assert wp.pct([], 50) is None
    assert wp.pct([5.0, 1.0, 3.0], 50) == 3.0
    assert wp.pct([1.0, 2.0, 3.0, 4.0, 5.0], 0) == 1.0
    assert wp.pct([1.0, 2.0, 3.0, 4.0, 5.0], 100) == 5.0
    assert wp.pct([1.0, 2.0, 3.0, 4.0, 5.0], 25) == 2.0
    assert wp.pct([1.0, 2.0, 3.0, 4.0], 50) == 3.0  # rank 1.5 rounds up


def test_universe_pairs_are_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    uni = tmp_path / "u.json"
    uni.write_text(json.dumps({"research": {"0": ["AAA_USDT"]}, "holdout": {"0": ["HHH_USDT"]}}))
    assert wp.universe_pairs(uni) == {"AAA_USDT", "HHH_USDT"}
    out = tmp_path / "x.jsonl.gz"
    for pair in ("AAA_USDT", "BTC_USDT,HHH_USDT"):
        argv = ["record", "--pairs", pair, "--seconds", "0", "--out", str(out), "--universe", str(uni)]
        assert wp.main(argv) == 2
    assert "refused" in capsys.readouterr().out
    assert not out.exists()


def test_subscriptions_follow_the_documented_shape() -> None:
    reqs = wp.subscriptions(["AAA_USDT", "BBB_USDT"], 1_790_000_000)
    assert reqs[0] == {
        "time": 1_790_000_000,
        "channel": "spot.trades",
        "event": "subscribe",
        "payload": ["AAA_USDT", "BBB_USDT"],
    }
    assert reqs[1]["channel"] == "spot.book_ticker"
    assert [r["payload"] for r in reqs[2:]] == [["ob.AAA_USDT.50"], ["ob.BBB_USDT.50"]]
