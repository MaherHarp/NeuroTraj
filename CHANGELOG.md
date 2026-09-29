# Changelog

All notable changes to neurodna are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/); while the version is below 1.0,
minor releases may change the public API.

## [Unreleased]

### Added
- README animation (`docs/assets/neurotraj.gif`) rendered from real frames of the
  3C2I micro-pilot by `docs/make_readme_gif.py`; rewritten README.
- Project URLs for https://github.com/MaherHarp/NeuroTraj.
- MIT `LICENSE`, author, keyword and classifier metadata in `pyproject.toml`.
- `MANIFEST.in`, so the source distribution contains everything the test suite
  needs (`tests/conftest.py`, `tests/data/3C2I.pdb`, `docs/`, example scripts and
  configs) and excludes cached downloads and MD outputs.
- Continuous integration (`.github/workflows/ci.yml`): tests on Python
  3.10–3.13, a job at the minimum supported dependency versions, strict mypy,
  ruff, and an sdist/wheel build that runs the tests from the unpacked sdist.
- Ruff lint configuration; `ruff`, `build` and `twine` in the `dev` extra.
- Packaging tests (`tests/test_package.py`): version metadata, `__all__`,
  `py.typed`, console scripts.

### Changed
- The package version is defined once, in `neurodna.__version__`, and read by
  setuptools; it is no longer duplicated in `pyproject.toml`.
- Pytest runs with `--strict-markers`.

### Fixed
- `neurodna.md.run`: the equilibration log is closed if a stage raises.
- `neurodna.md.run`: automatic platform selection (`platform=None`) always fell
  back to CPU, because its probe used an empty System, which OpenMM rejects on
  every platform. It now picks the fastest usable platform (for example a GPU).
- CI: every job has a time limit, and the OpenMM job pins `OPENMM_CPU_THREADS`
  to the runner's cores.

### Known issues
- On GitHub's x86 runners the OpenMM smoke run stalls inside OpenMM's
  `minimizeEnergy()`, so the CI OpenMM job is non-blocking. The same tests pass
  on macOS arm64, Linux arm64 and emulated x86 Linux; run them locally with
  `pytest -m openmm`.

## [0.3.0]

State of the project when release tracking began:

- `Complex` for static structures, ensembles and trajectories: residue identity
  (`ResidueKey`), explicit and validated selections, modified-residue
  registries (5mC via `5CM`), CpG/mCpG steps, contacts, minimum distances,
  contact occupancy, contact episodes, backbone RMSD and RMSF with
  whole-molecule checks.
- `neurodna.fetch` / `neurodna-fetch`: checksummed, cached PDB downloads.
- `neurodna.pdbheader`: PDB header records needed to verify an entry.
- `neurodna.plotting` (optional matplotlib).
- `neurodna.md` / `neurodna-md` (optional OpenMM): assess, prepare, run, analyse.
