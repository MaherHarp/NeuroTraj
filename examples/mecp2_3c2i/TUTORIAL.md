# Tutorial: static MeCP2–methylated DNA contacts in PDB 3C2I

This example computes a table of geometric protein–DNA contacts, and a contact
map, from one experimental structure: RCSB PDB entry
[3C2I](https://www.rcsb.org/structure/3C2I) (Ho et al., 2008).

It is a **static** analysis of a **single crystallographic model of a protein
domain**. It computes no dynamics, energies, hydrogen bonds or affinities, and
its output is a description of this one model. It is not a new biological
finding.

## Why MeCP2 and methylated DNA

- **Discovery.** MeCP2 was identified as a chromosomal protein that binds
  methylated DNA (Lewis et al., 1992).
- **The domain.** Binding is carried by a short methyl-CpG-binding domain (MBD)
  (Nan et al., 1993). Its solution structure was solved by NMR (Wakefield et
  al., 1999), and the related MBD1 domain was solved bound to methylated DNA
  (Ohki et al., 2001).
- **Rett syndrome.** Mutations in the X-linked *MECP2* gene cause Rett
  syndrome, a neurodevelopmental disorder (Amir et al., 1999).
- **BDNF.** MeCP2 and DNA methylation have been linked to activity-dependent
  regulation of the gene for brain-derived neurotrophic factor (BDNF), which
  is important for neuronal plasticity (Chen et al., 2003; Martinowich et
  al., 2003).
- **The structure.** 3C2I is the crystal structure of the MeCP2 MBD bound to a
  methylated DNA sequence from BDNF (Ho et al., 2008).

The authors report two main conclusions:

1. The cytosine methyl groups contact a predominantly hydrophilic surface that
   includes tightly bound water molecules. This suggests that MeCP2 recognises
   the hydration of methyl-CpG rather than the methylation itself.
2. T158, the residue most commonly mutated in Rett syndrome, has a structural
   role.

Looking at which residues sit near which nucleotides in such a structure is a
common first step before simulations or experiments. This example shows how to
do that step reproducibly.

## The entry, as verified

These facts were checked on 2026-09-25 against the RCSB data API, UniProt and
the file's own header records. `run_example.py` re-checks them from the header
every time it runs.

| | |
|---|---|
| Method | X-ray diffraction, 2.5 Å. Deposited 2008-01-25; latest revision 1.4 (2024-11-20). |
| Chain A | Human MeCP2, UniProt P51608 residues 77–167, plus a C-terminal His₆ tag (168–173). **This is the MBD only: full-length MeCP2 has 486 residues.** Only residues 91–162 are modelled (77–90 and 163–173 are missing). |
| Sequence differences | A140 is an **engineered mutation** to Met, modelled as selenomethionine (MSE). Met94 is also modelled as selenomethionine. Residue 140 is therefore not native. |
| Chains B, C | Two 20-nt DNA strands, `TCTGGAA[5mC]GGAATTCTTCTA` (B1–B20) and `ATAGAAGAATTC[5mC]GTTCCAG` (C21–C40), from BDNF. Each strand has one 5-methylcytosine (`5CM` B8, C33), forming a symmetrically methylated CpG. |
| Missing atoms | None are annotated (no REMARK 470), and RCSB reports no unobserved atoms. |
| Assembly | One biological assembly (author- and PISA-defined): chains A, B, C with the identity operator. The deposited coordinates therefore already are the complex. |
| Unit cell | Space group C 1 2 1. This is a crystal lattice, not an MD box, so neurodna does not apply periodic images. |
| Waters | 47 crystallographic waters, excluded from the protein–DNA contact tables. |

## Running it

```bash
pip install -e ".[plot]"
neurodna-fetch 3C2I                       # explicit download: URL, UTC time, SHA-256 → cache
python examples/mecp2_3c2i/run_example.py # verifies the pinned checksum, analyses, exports
```

- **Downloading.** Nothing is downloaded on import or by the test suite. The
  example downloads once into the cache (`--cache-dir` sets its location) and
  reuses it afterwards, after re-checking the checksum.
- **Checksum.** The pinned SHA-256 is that of the file retrieved on 2026-09-25.
  If RCSB remediates the entry, the example stops with a checksum error rather
  than silently analysing a different file.
- **Offline.** `--pdb-file PATH` analyses an existing local copy and records
  whether it matches the pinned checksum.

Outputs are written to `examples/mecp2_3c2i/output/`:

| File | Content |
|---|---|
| `residue_contacts.csv` | One row per residue–nucleotide pair with minimum heavy-atom distance ≤ 4.5 Å: chain, number, name, UniProt number, closest atoms, number of atom pairs, caveats. |
| `atom_contacts.csv` | Every heavy-atom pair ≤ 4.5 Å, with phosphate / sugar / base and backbone / side-chain labels. |
| `methyl_contacts.csv` | Protein heavy atoms within 4.5 Å of the 5mC methyl carbons (atom `C5A`). |
| `min_distances.csv` | All residue–nucleotide minimum distances ≤ 8 Å (the data behind the map). |
| `dna_residues.csv` | Nucleotide identities, with 5CM kept as `DC` + `5mC`. |
| `contact_map.png` | Protein residues × nucleotides (each strand 5′→3′), coloured by minimum distance; contacts outlined. |
| `parameters.json` | Input checksum and retrieval metadata, entry verification, selections, cutoff, units, periodic policy, exclusions, software versions. |

## What the measurements mean

- **Minimum heavy-atom distance.** For a protein residue and a nucleotide, the
  smallest distance between any of their non-hydrogen atoms, in Å. 3C2I
  contains no hydrogens.
- **Geometric contact.** A pair is in contact when that distance is ≤ 4.5 Å. The
  cutoff is a convention, not a physical boundary. A pair at 4.4 Å and one at
  4.6 Å are structurally similar, and differences of a few tenths of an ångström
  are within the coordinate uncertainty of a 2.5 Å structure.
- **Closest pair versus a specific group.** The residue table reports the
  closest atom pair, whatever it is. For example, Arg111's closest approach to
  5mC8 is its backbone CA to a phosphate oxygen (3.38 Å). Questions about the
  methyl groups themselves are answered by `methyl_contacts.csv`. There, Arg111
  (NH2, 3.57 Å), Asp121 (OD2, 3.48 Å) and Tyr123 (OH, 4.09 Å) are within 4.5 Å
  of the 5mC8 methyl, and Arg133 (CB, 3.43 Å) of the 5mC33 methyl.
- **What the table shows for 3C2I.**
  - Arg111 and Arg133 each approach the O6 atom of the guanine in an mCpG step
    on a different strand: DG9 at 2.59 Å and DG34 at 2.76 Å.
  - Thr158 is within 3.4 Å of the DT31 phosphate backbone.
  - These are descriptions of this one model, measured here.

## What this cannot establish

- **Dynamics.** One model says nothing about how long contacts last or how
  often they form. Repeating the analysis over crystallographic copies,
  alternative models or perturbed coordinates would not make it a trajectory.
  The time-resolved APIs are therefore demonstrated only on a synthetic toy
  system (`examples/synthetic_temporal_demo.py`).
- **Binding strength or specificity.** Distances are not energies. Nothing
  here predicts affinity, the effect of methylation on binding, or the effect
  of mutations.
- **Water-mediated recognition.** The primary paper's central observation
  concerns bound water at methyl-CpG, and those waters are excluded here.
  Hydrogen bonds are not assessed either, since contacts are purely
  distance-based.
- **Full-length MeCP2, chromatin or cells.** This is an isolated domain
  construct: it includes an engineered A140M substitution and selenomethionine,
  lacks residues 77–90 and 163–167, and is bound to one 20-bp DNA in a crystal
  lattice.
- **Disease.** No Rett syndrome variants are modelled, and the output must not
  be read as predicting pathogenicity.

## References

1. RCSB PDB entry 3C2I: Ho KL, McNae IW, Schmiedeberg L, Klose RJ, Bird AP,
   Walkinshaw MD. *The Crystal Structure of Methyl-CpG Binding Domain of Human
   MeCP2 in Complex with a Methylated DNA Sequence from BDNF.*
   https://www.rcsb.org/structure/3C2I (doi:10.2210/pdb3c2i/pdb).
2. Ho KL, et al. (2008) MeCP2 binding to DNA depends upon hydration at
   methyl-CpG. *Mol Cell* 29(4):525–531. doi:10.1016/j.molcel.2007.12.028.
   PMID 18313390.
3. Lewis JD, et al. (1992) Purification, sequence, and cellular localization
   of a novel chromosomal protein that binds to methylated DNA. *Cell*
   69(6):905–914. doi:10.1016/0092-8674(92)90610-o. PMID 1606614.
4. Nan X, Meehan RR, Bird A (1993) Dissection of the methyl-CpG binding domain
   from the chromosomal protein MeCP2. *Nucleic Acids Res* 21(21):4886–4892.
   doi:10.1093/nar/21.21.4886. PMID 8177735.
5. Wakefield RI, et al. (1999) The solution structure of the domain from
   MeCP2 that binds to methylated DNA. *J Mol Biol* 291(5):1055–1065.
   doi:10.1006/jmbi.1999.3023. PMID 10518942.
6. Ohki I, et al. (2001) Solution structure of the methyl-CpG binding domain
   of human MBD1 in complex with methylated DNA. *Cell* 105(4):487–497.
   doi:10.1016/s0092-8674(01)00324-5. PMID 11371345.
7. Amir RE, et al. (1999) Rett syndrome is caused by mutations in X-linked
   MECP2, encoding methyl-CpG-binding protein 2. *Nat Genet* 23(2):185–188.
   doi:10.1038/13810. PMID 10508514.
8. Chen WG, et al. (2003) Derepression of BDNF transcription involves
   calcium-dependent phosphorylation of MeCP2. *Science* 302(5646):885–889.
   doi:10.1126/science.1086446. PMID 14593183.
9. Martinowich K, et al. (2003) DNA methylation-related chromatin remodeling
   in activity-dependent BDNF gene regulation. *Science* 302(5646):890–893.
   doi:10.1126/science.1090842. PMID 14593184.
