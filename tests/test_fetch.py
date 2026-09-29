"""Download/caching logic with a fake opener: the test suite never uses the network."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import urllib.request
from email.message import Message
from pathlib import Path

import pytest

import neurodna.fetch as fetch
from neurodna.fetch import ChecksumError, DownloadError, fetch_pdb

REAL = Path(__file__).parent / "data" / "3C2I.pdb"


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body, self.status = body, status
        self.headers = Message()
        self.headers["Content-Type"] = "text/plain"
        self.headers["Last-Modified"] = "Wed, 20 Nov 2024 00:00:00 GMT"

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


@pytest.fixture
def opener(monkeypatch):
    """Serve the local 3C2I copy instead of contacting RCSB; count the calls."""
    calls: list[str] = []
    state = {"body": REAL.read_bytes()}

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        return FakeResponse(state["body"])

    monkeypatch.setattr(fetch, "_urlopen", fake_urlopen)
    return calls, state


def test_download_records_metadata_and_checksum(tmp_path, opener):
    calls, _ = opener
    got = fetch_pdb("3c2i", cache_dir=tmp_path)
    assert calls == ["https://files.rcsb.org/download/3C2I.pdb"]
    assert not got.from_cache and got.path == tmp_path / "pdb" / "3C2I.pdb"
    assert got.path.read_bytes() == REAL.read_bytes()
    meta = json.loads((tmp_path / "pdb" / "3C2I.pdb.json").read_text())
    assert meta["sha256"] == hashlib.sha256(REAL.read_bytes()).hexdigest() == got.sha256
    assert meta["source_url"] == calls[0] and meta["entry_id"] == "3C2I"
    assert meta["size_bytes"] == REAL.stat().st_size
    assert meta["http_last_modified"] == "Wed, 20 Nov 2024 00:00:00 GMT"
    assert meta["retrieved_at_utc"].endswith("+00:00")


def test_cache_is_reused_and_verified(tmp_path, opener):
    calls, _ = opener
    first = fetch_pdb("3C2I", cache_dir=tmp_path)
    again = fetch_pdb("3C2I", cache_dir=tmp_path, expected_sha256=first.sha256)
    assert again.from_cache and len(calls) == 1
    fetch_pdb("3C2I", cache_dir=tmp_path, refresh=True)
    assert len(calls) == 2
    first.path.write_bytes(first.path.read_bytes() + b"tampered\n")
    with pytest.raises(ChecksumError, match="was modified"):
        fetch_pdb("3C2I", cache_dir=tmp_path)


def test_pinned_checksum_mismatch_is_not_installed(tmp_path, opener):
    with pytest.raises(ChecksumError, match="remediated"):
        fetch_pdb("3C2I", cache_dir=tmp_path, expected_sha256="0" * 64)
    assert not (tmp_path / "pdb" / "3C2I.pdb").exists()


def test_wrong_content_is_rejected(tmp_path, opener):
    _, state = opener
    state["body"] = b"<html>not found</html>"
    with pytest.raises(DownloadError, match="did not return a PDB-format file"):
        fetch_pdb("3C2I", cache_dir=tmp_path)


def test_network_errors_are_wrapped(tmp_path, monkeypatch):
    def broken(request, timeout):
        raise OSError("no route to host")

    monkeypatch.setattr(fetch, "_urlopen", broken)
    with pytest.raises(DownloadError, match="no route to host"):
        fetch_pdb("3C2I", cache_dir=tmp_path)


def test_bad_arguments():
    with pytest.raises(ValueError, match="not a PDB entry ID"):
        fetch_pdb("../etc")
    with pytest.raises(ValueError, match="file_format"):
        fetch_pdb("3C2I", file_format="xml")


def test_cache_dir_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("NEURODNA_CACHE_DIR", str(tmp_path / "c"))
    assert fetch.default_cache_dir() == tmp_path / "c"


def test_cli(tmp_path, opener, capsys):
    assert fetch.main(["3C2I", "--cache-dir", str(tmp_path)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["from_cache"] is False and out["entry_id"] == "3C2I"
    assert fetch.main(["3C2I", "--cache-dir", str(tmp_path), "--sha256", "0" * 64]) == 1


def test_importing_neurodna_never_downloads(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network access during import")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    for name in [m for m in sys.modules if m == "neurodna" or m.startswith("neurodna.")]:
        monkeypatch.delitem(sys.modules, name)
    importlib.import_module("neurodna")
    importlib.import_module("neurodna.fetch")
