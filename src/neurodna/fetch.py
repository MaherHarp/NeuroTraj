"""Explicit, cached downloads of PDB entries from RCSB.

Nothing here runs on import. Files are downloaded only when :func:`fetch_pdb`
(or the ``neurodna-fetch`` command / ``python -m neurodna.fetch``) is called.

Each download is stored as ``<cache>/pdb/<ID>.<ext>`` with a JSON sidecar
``<ID>.<ext>.json`` recording the source URL, retrieval time (UTC), SHA-256
checksum, size and HTTP headers. Later calls reuse the cached file after
checking it still matches its recorded checksum.

RCSB does not publish checksums, so the SHA-256 is computed on retrieval. Pass
``expected_sha256`` to pin a known file. RCSB occasionally remediates entries
(see the entry's revision history), so a pinned checksum can legitimately stop
matching; the error then says so, and the new file is not installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from neurodna.errors import NeuroDNAError

RCSB_DOWNLOAD_URL = "https://files.rcsb.org/download/{entry_id}.{ext}"
RCSB_ENTRY_PAGE = "https://www.rcsb.org/structure/{entry_id}"
FORMATS = {"pdb": "pdb", "cif": "cif"}
_ENTRY_ID = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

# Indirection so tests can substitute a fake opener (no network in the test suite).
_urlopen: Callable[..., Any] = urllib.request.urlopen


class DownloadError(NeuroDNAError):
    """The download failed or returned unexpected content."""


class ChecksumError(NeuroDNAError):
    """A file does not match its expected or recorded SHA-256 checksum."""


@dataclass(frozen=True)
class FetchedFile:
    """A cached PDB file and its retrieval metadata."""

    path: Path
    metadata: dict[str, Any]
    from_cache: bool

    @property
    def sha256(self) -> str:
        return str(self.metadata["sha256"])


def default_cache_dir() -> Path:
    """``$NEURODNA_CACHE_DIR``, else ``$XDG_CACHE_HOME/neurodna``, else ``~/.cache/neurodna``."""
    if os.environ.get("NEURODNA_CACHE_DIR"):
        return Path(os.environ["NEURODNA_CACHE_DIR"]).expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return Path(base).expanduser() / "neurodna"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_pdb(
    entry_id: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    file_format: str = "pdb",
    expected_sha256: str | None = None,
    refresh: bool = False,
    timeout: float = 60.0,
) -> FetchedFile:
    """Return a local copy of an RCSB PDB entry, downloading it if needed.

    Args:
        entry_id: Four-character PDB ID, e.g. ``"3C2I"`` (case-insensitive).
        cache_dir: Cache root (default :func:`default_cache_dir`).
        file_format: ``"pdb"`` (legacy PDB format, read by neurodna) or ``"cif"``.
        expected_sha256: Pin the file content. Checked for both cached and new files.
        refresh: Download again even if a cached copy exists.
        timeout: Network timeout in seconds.

    Raises:
        ValueError: malformed entry ID or format.
        DownloadError: network failure or content that is not the requested entry.
        ChecksumError: the file does not match ``expected_sha256``, or a cached
            file no longer matches its recorded checksum.
    """
    entry = entry_id.strip().upper()
    if not _ENTRY_ID.match(entry):
        raise ValueError(f"not a PDB entry ID: {entry_id!r}")
    if file_format not in FORMATS:
        raise ValueError(f"file_format must be one of {sorted(FORMATS)}, got {file_format!r}")
    expected = expected_sha256.lower() if expected_sha256 else None
    ext = FORMATS[file_format]
    directory = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    directory = directory / "pdb"
    path = directory / f"{entry}.{ext}"
    sidecar = directory / f"{entry}.{ext}.json"

    if path.exists() and sidecar.exists() and not refresh:
        metadata = json.loads(sidecar.read_text())
        actual = sha256_file(path)
        if actual != metadata.get("sha256"):
            raise ChecksumError(
                f"cached {path} no longer matches the checksum recorded at download "
                f"({metadata.get('sha256')}); it was modified. Re-download with refresh=True."
            )
        _check_expected(actual, expected, str(path))
        return FetchedFile(path, metadata, from_cache=True)

    url = RCSB_DOWNLOAD_URL.format(entry_id=entry, ext=ext)
    request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    try:
        with _urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", None)
            headers = response.headers
            body = response.read()
    except OSError as exc:  # URLError, HTTPError and timeouts are OSErrors
        raise DownloadError(f"could not download {url}: {exc}") from exc
    if status is not None and status != 200:
        raise DownloadError(f"{url} returned HTTP {status}")
    _check_content(body, entry, file_format, url)

    digest = hashlib.sha256(body).hexdigest()
    _check_expected(digest, expected, url)
    metadata = {
        "entry_id": entry,
        "format": file_format,
        "source_url": url,
        "entry_page": RCSB_ENTRY_PAGE.format(entry_id=entry),
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sha256": digest,
        "size_bytes": len(body),
        "http_status": status,
        "http_last_modified": headers.get("Last-Modified"),
        "http_etag": headers.get("ETag"),
        "content_type": headers.get("Content-Type"),
        "retrieved_by": _user_agent(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, body)
    _atomic_write(sidecar, (json.dumps(metadata, indent=2) + "\n").encode())
    return FetchedFile(path, metadata, from_cache=False)


def _check_expected(actual: str, expected: str | None, where: str) -> None:
    if expected is not None and actual != expected:
        raise ChecksumError(
            f"SHA-256 of {where} is {actual}, expected {expected}. The entry may have been "
            "remediated by RCSB (check its revision history); verify the new file before "
            "updating the pinned checksum."
        )


def _check_content(body: bytes, entry: str, file_format: str, url: str) -> None:
    head = body[:4096].decode("ascii", errors="replace")
    if file_format == "pdb":
        first = head.splitlines()[0] if head else ""
        if not first.startswith("HEADER") or first[62:66].upper() != entry:
            raise DownloadError(f"{url} did not return a PDB-format file for {entry}")
    elif not head.startswith(f"data_{entry}"):
        raise DownloadError(f"{url} did not return an mmCIF file for {entry}")


def _atomic_write(path: Path, data: bytes) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _user_agent() -> str:
    from neurodna import __version__

    return f"neurodna/{__version__}"


def main(argv: Sequence[str] | None = None) -> int:
    """Command line: ``neurodna-fetch 3C2I [--cache-dir DIR] [--sha256 HEX] [--refresh]``."""
    parser = argparse.ArgumentParser(
        prog="neurodna-fetch", description="Download a PDB entry from RCSB into the local cache."
    )
    parser.add_argument("entry_id", help="PDB ID, e.g. 3C2I")
    parser.add_argument("--cache-dir", default=None, help="cache root (default: %(default)s -> "
                        f"{default_cache_dir()})")
    parser.add_argument("--format", dest="file_format", choices=sorted(FORMATS), default="pdb")
    parser.add_argument("--sha256", dest="expected_sha256", default=None,
                        help="expected SHA-256 of the file")
    parser.add_argument("--refresh", action="store_true", help="download even if cached")
    args = parser.parse_args(argv)
    try:
        fetched = fetch_pdb(args.entry_id, cache_dir=args.cache_dir, file_format=args.file_format,
                            expected_sha256=args.expected_sha256, refresh=args.refresh)
    except (NeuroDNAError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"path": str(fetched.path), "from_cache": fetched.from_cache,
                      **fetched.metadata}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
