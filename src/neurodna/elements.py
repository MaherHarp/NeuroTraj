"""Heavy-atom identification.

A heavy atom is any atom that is not a hydrogen isotope (H, D, T). The element is
taken from the topology when it provides one (PDB element column, TPR atomic
numbers, ...). The atom name is used as a cross-check, and as the only source when
the topology has no element for that atom.

Name rule (standard PDB/AMBER/CHARMM protein and nucleic-acid names): strip
leading digits, then a first letter ``H`` means hydrogen and ``C``, ``N``, ``O``,
``S`` or ``P`` means heavy. Anything else cannot be classified from the name.

An assignment is rejected as ambiguous when:

* the element and the name disagree on hydrogen vs heavy (e.g. a cysteine
  ``HG`` atom whose element column says ``Hg``, mercury);
* the element is not a chemical element symbol;
* there is no element and the name cannot be classified (e.g. virtual sites).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

ELEMENT_SYMBOLS = frozenset(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
    Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce
    Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn
    Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl
    Mc Lv Ts Og D T""".split()
)
_HYDROGEN_ISOTOPES = frozenset({"H", "D", "T"})
_HEAVY_NAME_INITIALS = frozenset("CNOSP")

NameClass = Literal["hydrogen", "heavy"]


def classify_by_name(atom_name: str) -> NameClass | None:
    """Hydrogen/heavy classification from a standard atom name, or ``None`` if unclear."""
    stripped = atom_name.strip().lstrip("0123456789")
    if not stripped:
        return None
    first = stripped[0].upper()
    if first == "H":
        return "hydrogen"
    if first in _HEAVY_NAME_INITIALS:
        return "heavy"
    return None


def _normalize_element(element: str) -> str:
    e = element.strip()
    return e[:1].upper() + e[1:].lower()


def classify_heavy_atoms(
    names: Sequence[str], elements: Sequence[str] | None
) -> tuple[npt.NDArray[np.bool_], list[tuple[int, str]]]:
    """Classify atoms as heavy (True) or hydrogen (False).

    Returns:
        ``(is_heavy, problems)``: ``problems`` lists ``(position, reason)`` for
        every atom whose classification is ambiguous. ``is_heavy`` is only
        meaningful when ``problems`` is empty.
    """
    is_heavy = np.zeros(len(names), dtype=bool)
    problems: list[tuple[int, str]] = []
    for i, name in enumerate(names):
        from_name = classify_by_name(str(name))
        element = _normalize_element(str(elements[i])) if elements is not None else ""
        if element:
            if element not in ELEMENT_SYMBOLS:
                problems.append((i, f"element {element!r} is not a chemical element"))
                continue
            hydrogen = element in _HYDROGEN_ISOTOPES
            if from_name is not None and (from_name == "hydrogen") != hydrogen:
                problems.append(
                    (i, f"element {element!r} contradicts the atom name, which denotes a "
                        f"{from_name} atom")
                )
                continue
            is_heavy[i] = not hydrogen
        elif from_name is None:
            problems.append((i, "no element in the topology and the atom name is not a standard "
                                "H/C/N/O/S/P name"))
        else:
            is_heavy[i] = from_name == "heavy"
    return is_heavy, problems
