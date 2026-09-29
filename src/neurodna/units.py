"""Units used throughout neurodna.

neurodna uses the native units of MDAnalysis and never converts:

* length / distance: angstrom (Å); 1 Å = 0.1 nm
* time: picosecond (ps); 1 ns = 1000 ps

GROMACS files store nanometres, but MDAnalysis converts them to Å when reading,
so a 0.4 nm cutoff is ``cutoff=4.0`` here. Output columns carry their unit as a
suffix: ``_A`` for ångström (``min_distance_A``, ``rmsd_A``) and ``_ps`` for
picoseconds (``time_ps``, ``observed_duration_ps``).
"""

from typing import Final

LENGTH_UNIT: Final = "angstrom"
TIME_UNIT: Final = "ps"
