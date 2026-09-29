"""The 3C2I example, run offline on the bundled copy of the real structure."""

from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "tests" / "data" / "3C2I.pdb"


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    return load_script("run_example", ROOT / "examples" / "mecp2_3c2i" / "run_example.py")


@pytest.fixture(scope="module")
def outputs(example, tmp_path_factory):
    out = tmp_path_factory.mktemp("mecp2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # MDAnalysis notes about PDB header records
        return example.analyze(REAL, out, {"source": "test copy"})


def test_bundled_copy_matches_pinned_checksum(example):
    from neurodna.fetch import sha256_file

    assert sha256_file(REAL) == example.EXPECTED_SHA256


def test_all_outputs_written(outputs):
    for path in outputs.values():
        assert path.exists() and path.stat().st_size > 0
    assert outputs["contact_map"].read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_methylated_residues_preserved_and_contacted(outputs):
    dna = pd.read_csv(outputs["dna_residues"])
    assert set(dna.loc[dna.modification == "5mC", "label"]) == {"B:5CM8", "C:5CM33"}
    methyl = pd.read_csv(outputs["methyl_contacts"])
    assert set(zip(methyl.protein_label, methyl.dna_label)) >= {
        ("A:ARG111", "B:5CM8"), ("A:ARG133", "C:5CM33")}
    assert (methyl.dna_atom == "C5A").all() and (methyl.distance_A <= 4.5).all()


def test_contact_table_identities_and_uniprot_numbering(outputs):
    df = pd.read_csv(outputs["residue_contacts"])
    assert (df.protein_chain == "A").all() and set(df.dna_chain) == {"B", "C"}
    assert (df.protein_uniprot_resnum == df.protein_resnum).all()  # DBREF maps 77-167 1:1
    assert (df.min_distance_A <= 4.5).all()


def test_parameters_record_provenance_and_caveats(outputs, example):
    p = json.loads(outputs["parameters"].read_text())
    assert p["input"]["sha256"] == example.EXPECTED_SHA256
    assert p["selections"] == {"protein": "chainID A and not resname HOH",
                               "dna": "chainID B C and not resname HOH"}
    assert p["cutoff_A"] == 4.5 and p["frame_kind"] == "static"
    assert p["minimum_image_used"] is False and "C 1 2 1" in p["periodic_policy"]
    assert "A:MSE140: ENGINEERED MUTATION (UNP P51608 ALA140)" in p["entry"]["protein"]["sequence_differences"]
    assert p["entry"]["assembly"]["identity_operator_only"] is True
    assert p["excluded"]["crystallographic_waters"] == 47
    assert [s["annotated_sequence_5to3"] for s in p["dna_strands"]] == [
        "TCTGGAA[5mC]GGAATTCTTCTA", "ATAGAAGAATTC[5mC]GTTCCAG"]


def test_verification_rejects_other_entries(example, tmp_path):
    from neurodna.pdbheader import read_pdb_header

    text = REAL.read_text().replace("3C2I", "9XYZ", 1)
    other = tmp_path / "other.pdb"
    other.write_text(text)
    with pytest.raises(RuntimeError, match="not 3C2I"):
        example.verify_entry(read_pdb_header(other))


def test_synthetic_temporal_demo_runs(capsys):
    demo = load_script("synthetic_temporal_demo", ROOT / "examples" / "synthetic_temporal_demo.py")
    demo.main()
    out = capsys.readouterr().out
    assert out.count("SYNTHETIC") >= 3 and "residence" in out
