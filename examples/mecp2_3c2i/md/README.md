# Optional OpenMM workflow for MeCP2 MBD–mCpG DNA (PDB 3C2I)

`neurodna.md` prepares, simulates and analyses 3C2I with OpenMM. It is optional:
install it with `pip install -e ".[md]"`. Importing neurodna never imports OpenMM.

This directory holds the preparation config (`preparation.json`) and the
assessment report it produces (`assessment_3C2I.json`). Generated systems,
trajectories and checkpoints go to `output/`, which is git-ignored.

**Runs and analysis so far**

- `micro_pilot_report.md`: the 25 ps laptop micro-pilot (Met140 system).
- `pilot_analysis_report.md`: its full analysis. It covers stability,
  structure, independent validation of neurodna, the readiness checklist and
  the recommended next tiers. Figures and `diagnostics.json` are in
  `pilot_analysis/`.
- `diagnose_run.py`: technical diagnostics plus a brute-force check of
  neurodna's distances, occupancy and episodes, for any run.
- `next_stage/`: validated configs and commands for Tier A (laptop validation)
  and Tier B (GPU pilot), and `tiers.json`. They are written by
  `next_stage/make_configs.py` and **have not been run**.

## 1. Can 3C2I be prepared with documented parameters?

**Yes, with OpenMM's CHARMM36 (July 2024) port and the explicit choices below.**

This was checked on 2026-09-25 with OpenMM 8.6.1.

**Force-field coverage.**

| Component | Parameters | Evidence |
|---|---|---|
| Protein (MeCP2 91–162) | CHARMM36m | `charmm36_2024.xml`; every residue matches a template |
| DNA (2 × 20 nt) | CHARMM36 nucleic acids: base template + `DEOX`, 5′-OH end `DEO5TER`, 3′ end `DEOX` + `3TER` | matched explicitly, residue by residue |
| **5-methylcytosine** (`5CM` B8, C33) | CYT + `DEOX` + **`5MC2`** | See below |
| Water / ions | CHARMM TIP3P, SOD/CLA (`charmm36_2024/water.xml`) | |

Evidence for the 5-methylcytosine parameters:

- The patch is from CHARMM's `stream/na/toppar_all36_na_modifications.str`, where
  it is described as `PRES 5MC2 ! Patch to convert cytosine in DNA to 5-methylcytosine`.
- Its C5 atom type, CN3D, has bonds, angles, dihedrals and Lennard-Jones terms
  annotated `5mc, adm jr. 9/9/93`.
- I verified the source files against the MD5 hashes that OpenMM's XML records.
  The stream file matched 23 independent public copies (the MacKerell lab site
  had an expired TLS certificate, so it was not used).
- A runtime audit checks every bond, angle and proper torsion around the methyl
  group. Seven torsions have no OpenMM term; each is defined with a zero force
  constant in the CHARMM source files, and those files are listed in
  `neurodna/md/forcefield.py`.

Other force fields were considered and not used:

- **Amber** (ff14SB/ff19SB with OL15/OL21/bsc1) as shipped with OpenMM has **no
  5-methyl-deoxycytidine**. Using it would mean either replacing 5mC with C or
  bringing in outside parameters. Neither is done.
- **CHARMM's `5MC` residue** is RNA 5-methylcytidine: it has a 2′-OH. It is not
  used.

**Template matching.** OpenMM's automatic patch matching is ambiguous here.
Thymine matches both `THY+DEOX` and `URA+5MC2+DEOX`, which have different
parameters. So templates are assigned explicitly. OpenMM refuses to guess, and
so does this workflow.

**Structure inspection.** From the header, RCSB and the coordinates:

| Item | Finding | Handling |
|---|---|---|
| Missing residues | 77–90 and 163–173 (N/C-terminal, including the His tag); no internal gaps | not modelled (unobserved, disordered) |
| Missing atoms | none (no REMARK 470; RCSB reports none) | only a C-terminal OXT is added |
| Selenomethionine | MSE94 (native Met), MSE140 (engineered A140M) | **explicit** Se→S conversion to Met (no Se parameters) |
| Engineered residue | A140M | **explicit**: kept as Met, as crystallised (`revert_to_alanine: []`); `["A:140"]` restores native Ala |
| Termini | protein truncated at R91/R162, whose terminal N/C atoms are 7.0/6.4 Å from DNA | **explicit** `charged`; see the caveat below |
| DNA ends | 5′-OH (no phosphate) on B1 and C21; 3′-OH on B20 and C40 | CHARMM 5TER/3TER patches |
| Protonation | no His or Cys; 7 Asp, 3 Glu, 7 Lys, 6 Arg, 4 Tyr | standard states at pH 7 (no pKa calculation) |
| Waters | 47 crystallographic waters | **explicit** `keep` (hydration at mCpG is central to Ho et al. 2008) |
| Ions / ligands | none in the entry | neutralised + 0.15 M NaCl added |
| Alternate conformations | none | |
| Assembly | biological assembly 1 = chains A, B, C, identity operator | deposited coordinates used |

**Choices without a neutral default must be written in the config;** loading
fails otherwise. They are: MSE handling, engineered residues, crystal waters,
and termini. Unsupported options (for example neutral caps) are rejected with
an explanation rather than approximated.

## 2. Workflow

```bash
neurodna-fetch 3C2I --cache-dir examples/mecp2_3c2i/cache
P=examples/mecp2_3c2i/cache/pdb/3C2I.pdb; C=examples/mecp2_3c2i/md/preparation.json
O=examples/mecp2_3c2i/md/output

neurodna-md assess  $P --config $C --output $O/assessment.json
neurodna-md prepare $P --config $C --output-dir $O/prepared
neurodna-md run $O/prepared --protocol smoke --output-dir $O/run_smoke        # integration test
neurodna-md run $O/prepared --protocol production --output-dir $O/run_prod    # scientific template
neurodna-md run $O/prepared --protocol production --output-dir $O/run_prod --resume
neurodna-md analyze $O/run_smoke --output-dir $O/run_smoke/analysis
```

**Preparation** writes:

- `prepared.pdb`: topology and coordinates. Solvent sits in chains W, X, … of at
  most 9,999 residues each, with ions in chain I.
- `system.xml`: the serialised System (PME, 1.2 nm cutoff, 1.0 nm switch,
  H-bond constraints).
- `preparation.json`: every conversion and added atom, the template of every
  nucleotide, the 5mC parameter audit, composition and charge, force-field
  source hashes, software versions, and a per-force energy fingerprint.

**Reproducibility.** Python's `random` module, which OpenMM's Modeller uses for
initial hydrogen positions and ion placement, is seeded, and hydrogens are
placed on the Reference platform. Three preparations gave the identical
`prepared.pdb` (SHA-256 `48937243…`).

`system.xml` can differ byte-wise between processes because OpenMM numbers the
CHARMM NBFIX atom-type tables in hash order. The per-force energies of two such
files were identical at the same coordinates, so compare the energy fingerprint
instead. Allow a relative tolerance of about 10⁻⁶, because CPU PME summation is
not bitwise deterministic.

**A run** proceeds in stages:

1. restrained minimisation;
2. restrained NVT and NPT equilibration (the restraint constant is set per stage);
3. unrestrained NPT production (Langevin middle integrator, 2 fs, 300 K,
   1 bar, Monte Carlo barostat every 25 steps).

Seeds for the velocities, integrator and barostat derive from the protocol seed.

It writes:

- a checkpoint (`.chk`) and a portable state (`.xml`) after every stage;
- periodic production checkpoints;
- `production.xtc`, with real times and `enforcePeriodicBox=False`, so molecules
  stay whole;
- energy/temperature/volume logs;
- `simulation.json`: protocol, seeds, platform, software, input checksums, and
  the wall time and ns/day of every stage.

**Resuming.** `--resume` continues from the last checkpoint. It drops any frames
written after that checkpoint, and refuses to continue if any setting other than
the production length changed.

What a resume preserves was tested on a small synthetic water box (OpenMM 8.6.1,
Reference platform):

- **Without a barostat**, a checkpoint-split run reproduces the uninterrupted run
  bit for bit.
- **With the Monte Carlo barostat**, the random stream is restored, since the seed
  has no effect after loading. The barostat's *adaptive trial step size* is not
  restored: it restarts at its initial value in the new Context. The continuation
  is therefore a valid NPT trajectory, but not bitwise identical to an
  uninterrupted one.
- **Practical rule:** run sessions long enough for the step size to re-adapt
  (thousands of barostat attempts, i.e. ≥ 100 ps at one attempt per 25 steps).
  Resume only when a session is interrupted. Checkpoints are only *saved* on
  schedule, and saving does not reset anything.

`--stop-at-production-step` exists for environments with hard time limits. The
2.5 ps chunks of the laptop micro-pilot (50 barostat attempts each) are far too
short for this rule.

**Analysis.** `analyze` checks the trajectory against `simulation.json`:

- the checksum, frame count and frame spacing;
- the checksum of the prepared system's `preparation.json`.

If the relative path recorded in `simulation.json` does not resolve from the
current directory, pass `--prepared-dir`. `--burn-in-ps` excludes early frames.

It then runs neurodna's occupancy,
episodes, backbone RMSD and RMSF, and labels every output with the protocol's
purpose and `convergence_assessed: false`.

**Externally prepared systems** follow the same route:

```bash
neurodna-md import-external --topology sys.pdb --system system.xml \
    --protein "chainID A and not resname HOH" --dna "chainID B C and not resname HOH" \
    --output-dir prepared_external
```

The preparation is recorded as unchecked by neurodna. Trajectories from other
engines (GROMACS, Amber, NAMD) go straight to `neurodna.Complex.load`.

## 3. What was actually run, and what it cost

All figures below were measured in this repository on an Apple M3 Max (14 CPU
threads), with OpenMM 8.6.1 on the CPU platform.

**The prepared system.** 49,302 atoms: protein + DNA + 47 crystal waters, plus
15,534 added waters, 77 Na⁺ and 42 Cl⁻. The rhombic dodecahedron has 1.2 nm
padding and a 500.5 nm³ box.

**Timings.**

| Step | Result |
|---|---|
| Assessment | 26 s |
| Preparation | 60 s |
| **Smoke protocol** (the only simulation run) | 100 minimisation iterations (46.5 s), 0.2 ps restrained NVT, 0.2 ps restrained NPT, **0.4 ps production** (+0.2 ps in a resume test) |
| Throughput | 1.55–1.72 ns/day in those stages; 1.9 ns/day in a separate 1,000-step benchmark of a 43,106-atom (1.0 nm padding) build |
| Total simulated | **0.4 ps equilibration + 0.6 ps production = 1.0 ps** |
| neurodna analysis of the smoke trajectory | 0.7 s |

**Disk.**

| File | Size |
|---|---|
| XTC | 179 KB per frame |
| Checkpoint | 2.4 MB |
| State XML | 8.2 MB per stage |
| `system.xml` | 17 MB |
| `prepared.pdb` | 4.0 MB |

**The smoke run is an integration test and nothing more.** Temperature and
density were still drifting when it ended; for example, the box volume was
still shrinking in production. During the short restrained NPT stage the
barostat accepted no trial moves: positional restraints resist box scaling.
None of its numbers describe MeCP2–DNA behaviour.

**GPU.** An OpenCL device (Apple M3 Max GPU) is detected, but running it was
not permitted in this session's sandbox, so it was **not benchmarked**. To
measure it yourself:

```bash
neurodna-md run $O/prepared --protocol smoke --output-dir $O/run_gpu --platform OpenCL
python -c "import json; print([(s['name'], s.get('ns_per_day')) for s in json.load(open('$O/run_gpu/simulation.json'))['stages']])"
```

Apple's OpenCL offers single precision only.

**The production template has not been run.** Do not launch it as is: the planned experiment
(conditions, replicates, seeds, preregistered analyses) is in
[docs/experiment_design.md](../../../docs/experiment_design.md). The template itself specifies 1.35 ns of staged
equilibration and 100 ns of production, with a frame every 10 ps. At the
measured CPU speed that is roughly 65 days of wall time. This is arithmetic
from the measured 1.55 ns/day, not a benchmark. Its trajectory would be about
1.8 GB (10,000 frames × 179 KB). In practice it needs a CUDA/OpenCL GPU.

**Convergence.** 100 ns of a single trajectory is not evidence of convergence.
Contact occupancies and episodes should be compared across independent
replicates, with different seeds and ideally independent preparations. The
workflow records durations and never labels a run as converged.

## 4. Limitations and blockers

- **Charged termini at truncation points.** NH₃⁺ at R91 and COO⁻ at R162 do not
  exist in full-length MeCP2, and both lie within about 7 Å of the DNA. Neutral
  ACE/NME capping is **not implemented**; prepare a capped system externally
  and use `import-external`.
- **Unobserved residues are not modelled.** Residues 77–90 and 163–173 were
  not seen in the crystal; their effect on DNA contacts is unknown.
- **5mC parameters are a documented CHARMM36 extension from 1993** (CN3D terms),
  distributed in the official modifications stream. Their validation for
  5mC-DNA with CHARMM36 is not assessed here.
- **Non-bonded switching.** OpenMM applies a potential switch (1.0–1.2 nm);
  CHARMM recommends force switching.
- **Protonation.** Standard states at pH 7, with no pKa prediction.
- **Crystal contacts are removed.** Contacts made with neighbouring molecules in
  the crystal lattice are absent in the simulated solution.
- **Other modified nucleotides are blocked.** Any modified nucleotide other than
  5CM (for example 5hmC, 5fC, 5caC, or a residue renamed in a file) stops the
  preparation with an explicit blocker; nothing is mapped to cytosine. Adding
  one requires documented parameters, a new entry in
  `neurodna/md/forcefield.py`, and a passing parameter audit.
- **Reproducibility is platform-dependent.** Seeds make runs reproducible in
  distribution, not bitwise: CPU/GPU platforms and thread counts change
  summation order.
- **What the run does not cover.** Short runs establish nothing about
  dynamics, and contact durations are not binding residence times (see the
  neurodna README).
