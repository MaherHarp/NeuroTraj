"""Packaging invariants: the installed distribution matches the importable package."""

from __future__ import annotations

from importlib import metadata, resources

import neurodna
import neurodna.md


def test_version_matches_installed_metadata() -> None:
    assert metadata.version("neurodna") == neurodna.__version__


def test_public_api_is_importable() -> None:
    for module in (neurodna, neurodna.md):
        missing = [name for name in module.__all__ if not hasattr(module, name)]
        assert not missing, f"{module.__name__}.__all__ lists undefined names: {missing}"


def test_ships_type_marker() -> None:
    assert resources.files("neurodna").joinpath("py.typed").is_file()


def test_console_scripts_resolve() -> None:
    scripts = {ep.name: ep for ep in metadata.entry_points(group="console_scripts") if ep.dist
               and ep.dist.name == "neurodna"}
    assert set(scripts) == {"neurodna-fetch", "neurodna-md"}
    for ep in scripts.values():
        assert callable(ep.load())
