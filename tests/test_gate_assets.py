from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from mmlab.data_ingestion import gate_assets as ga

SEEN: list[str] = []


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        SEEN.append(self.path)
        name = self.path.rsplit("/", 1)[-1]
        if "/spot/currencies/" in self.path and name != "BAD":
            body: object = {"currency": name, "name": f"{name} token", "delisted": False}
            code = 200
        elif "/spot/currency_pairs/" in self.path:
            body, code = {"id": name, "trade_status": "tradable"}, 200
        else:
            body, code = {"label": "INVALID_CURRENCY"}, 400
        b = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


@pytest.fixture(scope="module")
def base() -> Iterator[str]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/api/v4"
    srv.shutdown()
    srv.server_close()


def test_only_research_pairs_are_queried(base: str, tmp_path: Path) -> None:
    uni = {"research": {"1": ["BBB_USDT"], "0": ["AAA_USDT"]}, "holdout": {"0": ["SECRET_USDT"]}}
    (tmp_path / "u.json").write_text(json.dumps(uni))
    SEEN.clear()
    assert ga.main([str(tmp_path / "u.json"), str(tmp_path / "a.json"), "--base", base]) == 0
    out = json.loads((tmp_path / "a.json").read_text())
    assert list(out) == ["AAA_USDT", "BBB_USDT"]
    assert out["AAA_USDT"]["currency"]["name"] == "AAA token"
    assert not any("SECRET" in p for p in SEEN)


def test_errors_are_recorded_and_fail_the_run(base: str, tmp_path: Path) -> None:
    (tmp_path / "u.json").write_text(json.dumps({"research": {"0": ["BAD_USDT"]}, "holdout": {}}))
    assert ga.main([str(tmp_path / "u.json"), str(tmp_path / "a.json"), "--base", base]) == 1
    assert "error" in json.loads((tmp_path / "a.json").read_text())["BAD_USDT"]
