#!/usr/bin/env python
"""Fetch the IC-ThermBench assets this project needs, downloading as little as possible.

The benchmark publishes two Google Drive archives: datasets.zip (~0.5 GB, S2-S5) and
checkpoints.zip (~43 GB, 112 checkpoints). We only need level4_steady/, level5_steady/
and checkpoints/level4_UNet/. When the download URL honours HTTP Range requests, the
archives are opened remotely (zipfile over a seekable HTTP file) and only the wanted
members are extracted. Otherwise the small datasets archive is downloaded whole with
resume support; the checkpoints archive is refused because it does not fit on disk.

Google Drive enforces a daily download quota per public file. When the quota is
exhausted the script exits with code 3 so a retry loop can try again later.

Usage:
  python tools/fetch_assets.py --what datasets   --out external/thermbench_data
  python tools/fetch_assets.py --what checkpoint --out external/thermbench_data
  python tools/fetch_assets.py --what all        --out external/thermbench_data --manifest docs/results/data_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests

DATASETS_FILE_ID = "15Do8Raf070VseV9cn44j1hdVpD3Rz-Un"
CHECKPOINTS_FILE_ID = "1UE4bOYff7flo1IDXXL8tmyW6WL0jcLVa"
DATASET_MEMBERS = ("level4_steady/", "level5_steady/")
CHECKPOINT_MEMBERS = ("level4_UNet/",)
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/128.0 Safari/537.36"
READAHEAD = 8 * 1024 * 1024  # bytes fetched per Range request when reading remote zips

EXIT_OK, EXIT_ERROR, EXIT_NO_RANGE, EXIT_QUOTA = 0, 1, 2, 3


class QuotaExceeded(RuntimeError):
    pass


@dataclass
class Resolved:
    url: str
    size: int | None
    range_ok: bool


def resolve_drive_url(session: requests.Session, file_id: str) -> str:
    """Return the real download URL behind Drive's virus-scan confirmation page."""
    base = f"https://drive.usercontent.google.com/download?id={file_id}&export=download"
    r = session.get(base, timeout=60)
    r.raise_for_status()
    ctype = r.headers.get("content-type", "")
    if "text/html" not in ctype:
        return base  # small file served directly
    html = r.text
    title = re.search(r"<title>(.*?)</title>", html, re.S)
    title = title.group(1).strip() if title else ""
    if "Quota exceeded" in title:
        raise QuotaExceeded(f"Drive quota exceeded for {file_id}")
    action = re.search(r'action="([^"]+)"', html)
    inputs = re.findall(r'name="([^"]+)" value="([^"]*)"', html)
    if not action or not inputs:
        raise RuntimeError(f"unexpected Drive page for {file_id}: {title!r}")
    return action.group(1) + "?" + "&".join(f"{k}={v}" for k, v in inputs)


def probe(session: requests.Session, url: str) -> Resolved:
    """Ask for the first byte with a Range header and classify the response."""
    r = session.get(url, headers={"Range": "bytes=0-0"}, stream=True, timeout=60)
    ctype = r.headers.get("content-type", "")
    if "text/html" in ctype:
        text = r.text
        title = re.search(r"<title>(.*?)</title>", text, re.S)
        title = title.group(1).strip() if title else ""
        r.close()
        if "Quota exceeded" in title:
            raise QuotaExceeded(title)
        raise RuntimeError(f"HTML instead of file: {title!r}")
    if r.status_code == 206:
        cr = r.headers.get("content-range", "")
        m = re.search(r"/(\d+)$", cr)
        size = int(m.group(1)) if m else None
        r.close()
        return Resolved(url, size, True)
    size = r.headers.get("content-length")
    r.close()
    return Resolved(url, int(size) if size else None, False)


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a simple read-ahead buffer."""

    def __init__(self, session: requests.Session, url: str, size: int):
        self._s, self._url, self._size, self._pos = session, url, size, 0
        self._buf_start, self._buf = 0, b""
        self.requests_made = 0
        self.bytes_fetched = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self._pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self._pos = offset
        elif whence == io.SEEK_CUR: self._pos += offset
        elif whence == io.SEEK_END: self._pos = self._size + offset
        return self._pos

    def _fetch(self, start: int, length: int) -> bytes:
        end = min(start + length, self._size) - 1
        for attempt in range(5):
            try:
                r = self._s.get(self._url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code != 206:
                    raise RuntimeError(f"expected 206, got {r.status_code}")
                self.requests_made += 1
                self.bytes_fetched += len(r.content)
                return r.content
            except (requests.RequestException, RuntimeError) as e:
                if attempt == 4: raise
                time.sleep(2 ** attempt)
        return b""

    def read(self, n=-1):
        if n is None or n < 0:
            n = self._size - self._pos
        out = bytearray()
        while n > 0 and self._pos < self._size:
            if not (self._buf_start <= self._pos < self._buf_start + len(self._buf)):
                self._buf_start = self._pos
                self._buf = self._fetch(self._pos, max(READAHEAD, n))
            off = self._pos - self._buf_start
            chunk = self._buf[off:off + n]
            out += chunk
            self._pos += len(chunk)
            n -= len(chunk)
            if not chunk: break
        return bytes(out)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract_members(zf: zipfile.ZipFile, wanted: tuple[str, ...], out: Path) -> list[Path]:
    written = []
    for info in zf.infolist():
        if info.is_dir() or not any(w in info.filename for w in wanted):
            continue
        # strip any leading archive folder so level4_steady/ lands directly under out/
        rel = info.filename
        for w in wanted:
            i = rel.find(w)
            if i >= 0:
                rel = rel[i:]
                break
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  extracting {info.filename} -> {dest} ({info.file_size / 1e6:.1f} MB)", flush=True)
        with zf.open(info) as src, open(dest, "wb") as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                dst.write(block)
        written.append(dest)
    return written


def download_whole(session: requests.Session, url: str, dest: Path, size: int | None):
    """Stream the archive to disk with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size and have >= size:
        return
    headers = {"Range": f"bytes={have}-"} if have else {}
    with session.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        mode = "ab" if have and r.status_code == 206 else "wb"
        with open(dest, mode) as f:
            for block in r.iter_content(1 << 20):
                f.write(block)


def fetch(what: str, out: Path, manifest: dict) -> int:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    file_id, wanted = (DATASETS_FILE_ID, DATASET_MEMBERS) if what == "datasets" else (CHECKPOINTS_FILE_ID, CHECKPOINT_MEMBERS)

    url = resolve_drive_url(session, file_id)
    res = probe(session, url)
    print(f"[{what}] size={res.size} range_supported={res.range_ok}", flush=True)

    if res.range_ok and res.size:
        rf = HttpRangeFile(session, res.url, res.size)
        with zipfile.ZipFile(rf) as zf:
            names = [i.filename for i in zf.infolist()]
            print(f"[{what}] archive has {len(names)} members; extracting {wanted}", flush=True)
            written = extract_members(zf, wanted, out)
        print(f"[{what}] done: {rf.requests_made} range requests, {rf.bytes_fetched / 1e6:.1f} MB fetched", flush=True)
    else:
        if what != "datasets":
            print(f"[{what}] Range not supported; refusing to download {res.size} bytes whole", flush=True)
            return EXIT_NO_RANGE
        archive = out / "datasets.zip"
        print(f"[{what}] Range not supported; downloading whole archive to {archive}", flush=True)
        download_whole(session, res.url, archive, res.size)
        with zipfile.ZipFile(archive) as zf:
            written = extract_members(zf, wanted, out)
        archive.unlink()

    for p in written:
        manifest["files"][str(p.relative_to(out))] = {"bytes": p.stat().st_size, "sha256": sha256_of(p)}
    return EXIT_OK


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--what", choices=["datasets", "checkpoint", "all"], default="all")
    ap.add_argument("--out", type=Path, required=True, help="directory that will hold level4_steady/, level5_steady/, level4_UNet/")
    ap.add_argument("--manifest", type=Path, default=None, help="json file recording sizes and SHA-256 of fetched files")
    args = ap.parse_args()

    manifest = {"source": "IC-ThermBench (Day333/ThermalBench), data CC BY 4.0, code MIT",
                "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "files": {}}
    if args.manifest and args.manifest.exists():
        manifest["files"] = json.load(open(args.manifest)).get("files", {})

    todo = ["datasets", "checkpoint"] if args.what == "all" else [args.what]
    rc = EXIT_OK
    for what in todo:
        try:
            rc = max(rc, fetch(what, args.out, manifest))
        except QuotaExceeded as e:
            print(f"[{what}] {e}", flush=True)
            rc = max(rc, EXIT_QUOTA)
        except Exception as e:  # noqa: BLE001 - report and keep going with the next asset
            print(f"[{what}] failed: {type(e).__name__}: {e}", flush=True)
            rc = max(rc, EXIT_ERROR)

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        json.dump(manifest, open(args.manifest, "w"), indent=2)
    return rc


if __name__ == "__main__":
    sys.exit(main())
