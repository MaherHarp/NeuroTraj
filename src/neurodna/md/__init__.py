"""Optional OpenMM simulation workflow (``pip install 'neurodna[md]'``).

Importing this package does not import OpenMM; each step imports it when
called. The workflow is:

1. :func:`neurodna.md.assess.assess` -- can the structure be prepared with
   documented parameters? Lists blockers, required decisions and caveats.
2. :func:`neurodna.md.prepare.prepare` -- explicit conversions, hydrogens,
   solvent and ions, CHARMM36 (July 2024) system, provenance report.
3. :func:`neurodna.md.run.run` -- minimisation, restrained equilibration and
   production, with seeds, checkpoints, an unwrapped XTC trajectory and metadata.
4. :func:`neurodna.md.analyze.analyze_run` -- neurodna contact and RMSD/RMSF
   analysis of the production trajectory.

Externally prepared OpenMM systems enter at step 3 via
:func:`neurodna.md.cli.import_external`. Trajectories from other engines go
straight to :meth:`neurodna.Complex.load`.
"""

from neurodna.md.config import PRESETS, PreparationConfig, Protocol, Stage

__all__ = ["PRESETS", "PreparationConfig", "Protocol", "Stage"]
