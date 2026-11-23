"""Local cache of Gate public archive files (download.gatedata.org). Read-only, no credentials.

Layout: {root}/{biz}/{type}/{YYYYMM}/{file name as on the server}
Writes are atomic (unique .part file + rename, safe with concurrent processes) and gzip integrity is
checked before a file enters the cache.
"""

from __future__ import annotations

import gzip
import http.client
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

from mmlab.data_ingestion.gate_archive import USER_AGENT, ArchiveRef


class DownloadError(OSError):
    """The server answered, but the file could not be stored intact."""


def cache_path(root: Path, ref: ArchiveRef) -> Path:
    return root / ref.biz / ref.type / ref.period[:6] / ref.url.rsplit("/", 1)[1]


def _check_gzip(path: Path) -> None:
    try:
        with gzip.open(path, "rb") as fh:
            while fh.read(1 << 20):
                pass
    except (OSError, EOFError, zlib.error) as e:
        raise DownloadError(f"corrupt gzip {path.name}: {e}") from None


def fetch(
    ref: ArchiveRef,
    root: Path,
    opener: urllib.request.OpenerDirector | None = None,
    timeout: float = 120.0,
    retries: int = 5,
    backoff_s: float = 2.0,
) -> Path | None:
    """Return the cached path, downloading if needed. None if the server has no such file (404).

    Transient failures (connection reset or refused, timeouts, truncated or corrupt transfers, HTTP 5xx
    and 429) are retried up to `retries` times with exponential backoff; other errors are raised.
    """
    dest = cache_path(root, ref)
    if dest.exists():
        return dest
    for attempt in range(retries + 1):
        try:
            return _download(ref, dest, opener, timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (OSError, http.client.HTTPException):
            if attempt == retries:
                raise
        time.sleep(backoff_s * 2**attempt)
    raise AssertionError("unreachable")


def _download(
    ref: ArchiveRef, dest: Path, opener: urllib.request.OpenerDirector | None, timeout: float
) -> Path | None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    opener = opener or urllib.request.build_opener()
    req = urllib.request.Request(ref.url, headers={"User-Agent": USER_AGENT})
    try:
        with opener.open(req, timeout=timeout) as resp, open(tmp, "wb") as fh:
            shutil.copyfileobj(resp, fh)
            length = str(resp.headers.get("Content-Length") or "")
        if length.isdigit() and int(length) != tmp.stat().st_size:
            raise DownloadError(f"{ref.url}: got {tmp.stat().st_size} bytes, expected {length}")
        if ref.url.endswith(".gz"):
            _check_gzip(tmp)
        os.replace(tmp, dest)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404:
            return None
        raise
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return dest
