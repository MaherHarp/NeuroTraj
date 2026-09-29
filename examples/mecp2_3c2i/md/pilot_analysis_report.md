# Micro-pilot analysis and readiness to scale: 3C2I MeCP2 MBD – methylated DNA

> **Stable, not equilibrated, not converged.** The micro-pilot is 25 ps of
> dynamics. It shows that the workflow runs correctly and integrates stably.
> It cannot establish anything about MeCP2–DNA contacts, their persistence, or
> methylation effects. No duration proposed below guarantees convergence.

Date: 2026-09-26.

**Inputs:**
- `examples/mecp2_3c2i/md/output/micro_pilot/` (git-ignored run outputs);
- `protocols/micro_pilot.json`;
- `micro_pilot_report.md`;
- `../experiment/`.

**Everything here was recomputed** from the raw outputs by
`examples/mecp2_3c2i/md/diagnose_run.py` (figures and `diagnostics.json` in
`pilot_analysis/`).

**Nothing longer than the micro-pilot was simulated.** The only simulations run
for this analysis were:
- the 6 opt-in OpenMM tests (0.24 ps of dynamics per suite run), run twice;
- two checkpoint tests on a 1,335-atom synthetic water box (2,400 and 600 steps in total).

**Labels used throughout:** *software*, *numerical*, *physical* and *scientific*
mark four separate levels of evidence:
- workflow correctness;
- numerical stability;
- physical plausibility;
- scientific convergence.

---

## 1. Protocol evaluated

| Item | Value |
|---|---|
| System | prepared 3C2I (`md/output/prepared`, `prepared.pdb` SHA-256 `48937243…`): MBD residues 91–162 with **Met140 as crystallised** (MSE→MET), 5CM B8/C33 on CHARMM36 `CYT-DEOX_0-5MC2_4`, 47 crystal waters + 15,534 TIP3P, 77 Na⁺ / 42 Cl⁻ (0.15 M), rhombic dodecahedron, 49,302 atoms |
| Force field / nonbonded | CHARMM36 (July 2024) + CHARMM TIP3P; PME, 1.2 nm cutoff, switch from 1.0 nm; H-bond constraints; rigid water |
| Minimisation | ≤ 100 L-BFGS iterations, solute heavy-atom restraints k = 1000 kJ mol⁻¹ nm⁻² |
| Equilibration | NVT 2.5 ps (k = 1000), then NPT 2.5 ps (k = 100) |
| Production | unrestrained NPT, 20 ps; Langevin middle 2 fs, 300 K, 1 ps⁻¹; MC barostat 1 bar every 25 steps |
| Output | XTC every 0.25 ps (80 frames), unwrapped; checkpoint every 2.5 ps |
| Platform | OpenMM 8.6.1 CPU, 6 threads |
| Execution | 11 foreground sessions: minimisation, NVT, NPT, 8 × 2.5 ps production chunks, each resumed from the previous checkpoint |

This is not the experiment design's system, which uses native Ala140 (see §7).

## 2. Runtime and computational performance

| Measure | Value |
|---|---|
| Wall time, 11 sessions | 2,512.7 s (41.9 min); dynamics 2,402.0 s; minimisation 78.5 s (0.785 s/iteration) |
| Throughput, 6 threads | NVT 0.998, NPT 0.921, production chunks 0.866–0.905, **all dynamics 0.899 ns/day** (96 s per simulated ps) |
| Reference, 14 threads | 1.55–1.72 ns/day (smoke run) |
| CPU use | 5.6–5.9 cores per session; resident memory about 0.5 GB (measured for session 2) |
| Storage | XTC 179.9 kB/frame (3.64 B/atom/frame); checkpoint 2.4 MB; state XML 8.2 MB per stage; run directory 61 MB |
| GPU | **not measured**: OpenCL runs were not permitted in the agent sandbox |

## 3. Stability diagnostics

Production values are over the 80 production log rows (20 ps). Block
statistics use 4 blocks of 5 ps: serially correlated, few, and only a rough
guide.

| Quantity | Result | Level | Reading |
|---|---|---|---|
| NaN / exceptions / constraint failures | none in any session; all 100 logged rows finite; OpenMM raised nothing | numerical | **stable** |
| Minimisation | −174,043 → −783,486 kJ/mol | numerical | clash relief, as expected |
| Temperature | 249.8 → 299.9 K over equilibration. Production **300.81 ± 1.44 K** (297.0–304.9) | numerical / physical | stable. The SD matches the canonical expectation for 100,121 DOF, T·√(2/N_dof) = **1.34 K**. The design's flag window is 298.5–301.5 K |
| Kinetic temperature from saved velocities | 292.595 / 299.869 / 302.006 K at the three saved states, identical to the reporter; DOF = 3·49,302 − 47,782 constraints − 3 = 100,121 | software | reporter temperature and DOF accounting verified independently |
| Potential energy | −726,254 ± 1,202 kJ/mol; half means −725,363 / −727,145 (Δ = 3.2 block SE); slope −161 kJ/mol/ps | physical | **still drifting** |
| Box volume | 499.2 nm³ at the first NPT row → 482.4 nm³ at the end (−3.4%). Production 481.6 ± 1.9 nm³; half means 482.9 / 480.4 (Δ = 2.6 SE) | physical | **still decreasing** |
| System density | production 1.0497 ± 0.0042 g/mL; half means 1.0470 / 1.0525 | physical | still rising. The value is explained by composition (next row) |
| Bulk-water density (new) | water > 10 Å from solute: **1.012 ± 0.004 g/mL**, 33.8 molecules/nm³, still rising slightly | physical | about 1.5% above experimental water at 300 K (0.9965 g/mL). That is the scale of deviation typical of TIP3P-family models, which depend on the variant and the LJ cutoff treatment. **Water is not anomalous** |
| Density bookkeeping | total mass 304,465 g/mol ÷ mean volume = 1.0497 g/mL, matching the reporter. At bulk-water density the 15,581 waters fill 460.6 nm³, leaving 21.0 nm³ for protein + DNA + ions | physical | apparent specific volume ≈ 0.62 cm³/g for protein + DNA, the mass-weighted typical value (protein ~0.73, DNA ~0.55). A rough check: hydration water is not at bulk density |
| Pressure | not logged: the MC barostat samples volume at 1 bar, and instantaneous pressure fluctuates by hundreds of bar in this box | – | not meaningful here; volume and density are used instead |
| Barostat | 48% of 0.25 ps log intervals (5 attempts each) show a volume change. Per-interval volume changes: median 0.35 nm³, maximum 2.2 nm³ (σ_V ≈ √(k_B·T·V·κ_T) ≈ 1 nm³ for water compressibility) | numerical | working. It restarts its step-size adaptation at every resume (§7, item 2) |
| Timestamps | 80 frames, 0.25–20.00 ps, strictly increasing, uniform 0.25 ps; neurodna times equal the XTC header times | software | pass |
| Checkpoint / resume | 10 resumes; exactly 80 frames and 80 + 20 log rows, all steps unique and monotonic | software | pass (see the barostat caveat) |

**Summary.** Numerically the run is **stable**. It is **not equilibrated**:
potential energy, volume and density were still moving when it ended, as
expected after 5 ps of equilibration. Temperature stability says nothing about
equilibration of the solute or solvent structure.

## 4. Structural diagnostics

| Quantity | Result | Level | Reading |
|---|---|---|---|
| Whole molecules | protein: neurodna's wholeness check passed in every frame. DNA: 38 O3′–P links, max 1.682 Å. Water O–H max 0.973 Å (rigid 0.9572 Å plus XTC quantisation ≤ 0.017 Å) | software | no molecule split across the box; no wrapping (`enforcePeriodicBox=False`) |
| Box shape | angles constant at 60/60/90° (rhombic dodecahedron); isotropic scaling; edge 88.41 → 88.03 Å | software | pass |
| Complex integrity | protein–DNA centre-of-mass distance 15.45–16.23 Å (first 15.73, last 15.55) | physical | the complex stays together; no drift apart |
| Protein backbone RMSD vs prepared model | 1.04 ± 0.12 Å (0.58–1.28); half means 0.97 → 1.11; slope +0.015 Å/ps | physical | plausible relaxation from the crystal pose; **still rising; no plateau** |
| Protein backbone RMSD vs first production frame | max 1.15 Å, last 1.01 Å | physical | as above |
| DNA core heavy-atom RMSD vs prepared model (B4–B18 · C24–C38) | 0.93 ± 0.16 Å (0.54–1.19); half means 0.79 → 1.07; slope +0.026 Å/ps | physical | plausible; **still rising** |
| Watson–Crick N1–N3 (all 19 pairs) | fraction ≤ 3.2 Å per pair: 0.95–1.00; CpG-step pairs B8·C34 and B9·C33 **1.00**; max 3.38 Å (B13·C29) | physical | duplex intact. The design flags a core pair below 0.80 |
| CA RMSF | median 0.41 Å, max 0.98 Å (Gly129) | – | **not interpretable** over 20 ps of one run |
| 5mC representation | B:5CM8 and C:5CM33 recognised as 5mC by neurodna in every frame. C5–C5A 1.40–1.59 Å; C5A–H 1.10–1.12 Å (3 H each); template `CYT-DEOX_0-5MC2_4` | software / numerical | 5mC preserved through preparation, simulation and analysis |
| Charged termini | R91 N-terminal N to nearest phosphate O: 5.8–8.1 Å. R162 C-terminal O/OXT: 5.4–7.1 Å. Both are nearest to the B:DC15–DT16 / C:DT30 backbone | physical | artificial charges sit within about one Debye length (≈ 7.9 Å at 0.15 M) of the DNA (§7, item 4) |

Flattening RMSD would not show convergence; here the RMSD is not even flat.

## 5. neurodna validation (on its own trajectory)

`diagnose_run.py` recomputes each neurodna result independently: MDAnalysis
`distance_array` per frame over neurodna's heavy-atom groups, contact runs
found by a plain loop, and censoring from first/last-frame membership. It does
this for **all 19 preregistered primary pairs** over all 80 frames.

| Check | Result |
|---|---|
| Reads the topology and trajectory | `load_run` verifies the XTC SHA-256, frame count (80) and spacing (0.25 ps) against `simulation.json`. `topology.pdb` has the same SHA-256 as `prepared.pdb` |
| Residue identities | protein and DNA residue keys unique; labels match the topology (`A:ARG111`, `B:5CM8`, …) |
| 5mC | B:5CM8 and C:5CM33 carry `modification = 5mC`; the marker atom is present in every frame |
| Units | distances in Å: the B8 O3′–B9 P bond reads 1.551 Å. Times in ps |
| Real times | `time_source = file`; neurodna times equal the XTC times |
| PBC | minimum image used (valid box). Identical to direct distances within 5×10⁻⁶ Å, as expected for whole molecules |
| Minimum distances | max \|neurodna − brute force\| = **5.3×10⁻⁶ Å** (float32 XTC vs float64 recomputation) |
| Occupancy at 4.5 Å (inclusive) | max difference **0** |
| Episodes and censoring | boundaries and censoring flags **identical** for all 19 pairs, including with a 5 ps burn-in window (frames 20–79) |
| Provenance | complete after this task's fixes (below) |

**Verdict:** on this trajectory, neurodna reads, labels, times, measures and
censors correctly (software level).

**Discrepancies found and fixed.** Each was fixed in the software and followed
by the smallest validation.

1. **Incomplete analysis provenance.** `analysis_summary.json` lacked input
   checksums, selections, the frame window and software versions.
   - Fix: all are now recorded, plus the exact command with absolute paths.
   - New `--burn-in-ps` option for `neurodna-md analyze`. A burn-in of 0 now
     keeps a t = 0 frame.
2. **Run loading depended on the working directory.** `simulation.json` stored
   the prepared-system path as typed, so analysis failed from other
   directories, and `preparation.json` was never checked against its recorded
   checksum.
   - Fix: new runs also record the absolute path. `load_run`/`analyze` verify
     the checksum and accept `--prepared-dir`.
   - The micro-pilot, run before this fix, needs `--prepared-dir` when analysed
     outside the project root.
3. **Unsafe resume after an unplanned interruption.** After a Ctrl-C mid-chunk,
   resume truncated the XTC but not the energy logs, so rows from discarded
   dynamics would duplicate steps. Checkpoints were also overwritten in place.
   - Fix: resume now drops log rows after the resumed checkpoint (both logs).
     Checkpoints, states and `progress.json` are written atomically.
   - The micro-pilot was unaffected: it only stopped at checkpoints, and its
     logs have unique, monotonic steps.
   - Validation: 173 offline tests (11 new for QC/provenance, 1 for log
     truncation, 3 for the next-stage configs). The 6 opt-in OpenMM tests were
     re-run after the fix, and the resume test now includes an injected stale
     log row. All pass; mypy is clean.

## 6. Plots and interpretation

All figures are in `pilot_analysis/` and are regenerated by `diagnose_run.py`.
They are labelled as diagnostics.

| Figure | Shows | Interpretation |
|---|---|---|
| `fig1_potential_energy.png` | PE across NVT, NPT and production | rises during heating, then drifts down through production: **not equilibrated** |
| `fig2_temperature.png` | reporter temperature | reaches 300 K by about 4 ps and stays within ±5 K: thermostat working |
| `fig3_volume_density.png` | volume and system density | volume flat in NVT, falls 3.4% under NPT, still decreasing; stair-steps from discrete MC moves |
| `fig4_protein_rmsd.png` | backbone RMSD vs prepared model and vs first frame | about 1 Å, rising: relaxation from the crystal pose, no plateau |
| `fig5_dna_structure.png` | DNA core RMSD; all 19 Watson–Crick distances, CpG-step pairs highlighted | duplex intact; core RMSD rising |
| `fig6_primary_pair_distances.png` | minimum distance vs time for **all 19 preregistered primary pairs** (none selected or omitted), with the crystal value and cutoff | most pairs stay near their crystal distance. A few fluctuate across 4.5 Å: R133–B:5CM8, S113–B:DG9, R111–C:5CM33 |
| `fig7_primary_occupancy.png` | occupancy of the same 19 pairs | 14 at 1.0; R111–C:5CM33 0.94, S113–B:DG9 0.93, R133–B:5CM8 0.66, D121–C:5CM33 0.04, Y123–B:5CM8 0.03. **80 correlated frames from the crystal start: not persistence estimates, and not to be compared** |
| `fig8_termini_to_dna.png` | charged termini to nearest phosphate O | 5.4–8.1 Å throughout |

No exploratory pair plots were made.

## 7. Warnings and unresolved problems

1. **Not equilibrated** (expected). Potential energy, volume, density and
   bulk-water density were all still drifting.
2. **Resume is not a bitwise continuation under the barostat.** This was tested
   on a synthetic water box on the Reference platform, OpenMM 8.6.1.
   - **Without a barostat**, a checkpoint-split run is bit-identical to an
     uninterrupted one.
   - **With the MC barostat**, the two diverge at the first step after the
     resume. The barostat's random stream *is* restored: changing its seed
     after loading has no effect. Its **adaptive trial step size is not
     restored**.
   - Resumed NPT runs are therefore valid NPT sampling, but not reproducible
     bit for bit against an uninterrupted run.
   - The micro-pilot's eight 2.5 ps chunks (50 attempts each) restarted this
     adaptation every time. Its log shows no separable within-chunk pattern, so
     the effect on this trace cannot be quantified.
   - Rule, documented in `md/README.md`: resume only after interruptions, never
     in tiny chunks.
3. **The design system has never been simulated.**
   - This run used Met140; the design uses native Ala140, and the derived CpG
     model has not been integrated.
   - The design's 10,000-iteration minimisation and five-stage restraint ladder
     have not been run either.
   - In the earlier smoke run, the barostat accepted no moves during restrained
     NPT.
4. **Charged-termini artifact.**
   - **Origin.** The model starts and ends at R91 and R162 because residues
     77–90 and 163–167 of the crystallised construct are unobserved. The real
     chain continues on both sides. The model instead carries an NH₃⁺ and a COO⁻
     that do not exist in the protein.
   - **Measured.** Both termini stay 5.4–8.1 Å from DNA phosphate oxygens, near
     the B15–B16 / C30 backbone, 6–7 bp from the mCpG. They are not in direct
     contact (> 4.5 Å), but lie within about one Debye length (≈ 7.9 Å).
   - **Proximity to primary pairs.**
     - Thr158 comes within 4.4 Å of the C-terminal carboxyl group. T158–C:DT31 is
       a primary pair, and T158 is the residue highlighted in the literature.
     - Every other primary-pair residue stays ≥ 13.6 Å from both termini. That
       includes the confirmatory pairs: R111 ≥ 16.4 Å and R133 ≥ 14.9 Å.
   - **Assessment.**
     - *Acceptable* for technical pilots (the micro-pilot, Tier A).
     - *Serious* before any biological comparison. It is identical in both
       conditions, so it does not directly confound a mCpG–CpG difference.
     - It can still shift the local electrostatics and the positioning of the
       domain on the backbone near T158–DT31. Its size is unknown.
   - **Supported remedies.**
     - **Neutral ACE/CT3 caps.** These patches exist in the installed
       `charmm36_2024.xml` (`ACE_0–2`, `CT3_0–8`). neurodna cannot yet build
       the cap heavy atoms (CAY, CY, OY; NT, CAT). A builder using CHARMM
       internal coordinates would need implementing and validating: bonded
       audit, minimisation, and a comparison with an independent build.
     - **An external CHARMM-GUI build** with the same patches plus `5MC2`,
       brought in with `neurodna-md import-external`.
     - The `NNEU`/`CNEU` neutral-terminus patches also exist, but they leave an
       artificial polar chain end. Modelling the unobserved flanks would build
       unobserved structure.
   - **Not done.** No approximation, such as hand-zeroing terminal charges, was
     applied. The design (`docs/experiment_design.md`) was **not changed**.
     Adopting caps is a design amendment for the user to make, on modelling
     grounds rather than on simulation outcomes.
5. **GPU throughput unmeasured.** Tier B and Tier C wall times below are
   formulas with illustrative throughputs.
6. **Solvent-only density reference not run.** The bulk-water analysis resolves
   the 1.05 g/mL concern. A water box under identical settings remains the
   definitive comparison, and costs a few laptop minutes.
7. **5MC2 parameters not independently validated.** This is a standing design
   limitation, unchanged.

## 8. Ready-to-scale checklist

| Area | Status | Basis |
|---|---|---|
| Parameterisation | **PASS** | explicit CHARMM36 templates, including 5mC `5MC2` and nucleotide termini; no missing terms; 5mC geometry intact for 20 ps. Caveat: 5MC2 not independently validated (design limitation) |
| Preparation reproducibility | **PASS** | seeded preparation; energy fingerprint; `prepared.pdb`/`system.xml` checksums verified; run topology identical to the prepared PDB. The design system is prepared but not run (Warning 3) |
| Numerical stability | **PASS** | no NaN, exceptions or constraint failures; T 300.8 ± 1.4 K with canonical-size fluctuations; kinetic T verified from velocities |
| Trajectory integrity | **PASS** | 80/80 frames; uniform real times; checksum verified; whole molecules; constant box shape |
| Checkpoint/resume | **WARNING** | frames and logs continuous across 10 resumes. Interrupted-resume log handling was fixed in this task. Barostat step size resets at every resume: valid sampling, not bitwise |
| Thermodynamic diagnostics | **WARNING** | stable but not equilibrated: PE, volume and density drifting (Δ half means 2.6–3.2 block SE). Density explained; water plausible |
| Structural integrity | **PASS** (diagnostic) | complex intact; duplex paired (≥ 0.95 per pair); 5mC intact; RMSD about 1 Å and rising. Charged-termini proximity noted separately (Warning 4) |
| neurodna analysis correctness | **PASS** | independent brute force agrees (5×10⁻⁶ Å; occupancy and episodes exact) for all 19 primary pairs |
| Provenance / reproducibility | **PASS** (after this task's fixes) | input checksums, selections, frame window, software versions, command; prepared-dir checksum verified |
| Compute feasibility | **WARNING** | laptop 0.9 ns/day at 6 threads: fine for picosecond validation; Tier B would take 28 laptop-days. GPU throughput unmeasured |

There is no FAIL in parameterisation, stability or trajectory integrity, so
the design proceeds to tier planning. This table is not a score.

## 9. Recommended next tier

**Tier A (laptop validation) is the smallest next run worth doing.** The
software is validated. What remains untested is the *design's own* system and
protocol:
- Ala140 preparation;
- the five-stage restraint ladder with its restraint releases;
- the barostat under k = 1000 restraints;
- an unrestrained NPT stage;
- the QC code on the design system.

A preparation or protocol error found here costs nothing. Found on a GPU it
costs money, and found after a pilot it costs time. Tier A runs on this laptop
at 6 threads in about 2.3 h. **Per the task, it is configured but not
started.**

**Three tiers**

- **A. Laptop validation (recommended next).**
  - Design mCpG system; one run; 1,000-iteration minimisation.
  - 27.5 ps staged equilibration mirroring the design's five stages, then
    50 ps production.
  - Optional A2: a CpG build check (derive, prepare, minimise only; about
    15 min).
- **B. Modest GPU pilot (after A).**
  - The design's development tier as written (mCpG r1 and CpG r1, same
    protocol and seeds), plus two extra independent mCpG replicates.
  - Each run: ≤ 10,000-iteration minimisation, 1.35 ns staged equilibration,
    5 ns production, 10 ps output.
  - Step 0 is a GPU benchmark.
- **C. Research scale (documentation only).** The design's pilot
  (3 × 100 ns per condition) and stronger (5 × 500 ns) tiers, unchanged.

**Design considerations**

- **Length vs replicates.**
  - For consistency information, B uses three independent short replicates
    rather than one longer run.
  - 5 ns is kept deliberately, not as a default:
    - it equals the preregistered development tier;
    - it is the shortest length that gives the design's thermodynamic flag
      its ≥ 4 blocks of ≥ 1 ns after a 1 ns sensitivity burn-in.
  - At 5 ns, occupancy is informative only for pairs that exchange quickly.
- **Equilibration.** B uses the design's 1.35 ns ladder. Whether the drift has
  stopped is judged by the §8 flags per replicate, not assumed.
- **Termini** (Warning 4).
  - Decide before B, because B's replicate spread is meant to size C for the
    construct that will actually be compared.
  - If caps are adopted, implement and validate them, or build externally,
    then regenerate the B configs.
  - If charged termini are kept, record that as a limitation. Consider a
    capped sensitivity arm at C.
- **Ions.** 0.15 M NaCl as designed. KCl or divalent ions would be a different
  question and a separate arm; they were not added.
- **mCpG vs CpG.** No comparison is made before C. B's single CpG run
  validates the pipeline only (interpretation "none", as preregistered).
- **Alternative preparation.** A CHARMM-GUI build of the same model, imported
  with `import-external`, would cross-check neurodna's preparation (parameter
  and energy agreement on identical coordinates). It is also the fastest route
  to capped termini.

## 10. Exact proposed protocol (Tier A; not executed)

Configs are generated and validated by
`examples/mecp2_3c2i/md/next_stage/make_configs.py`. The Tier A files are in
`next_stage/tier_a_laptop/`:
- `preparation.json`: the design's shared preparation with
  `preparation_seed` 20261001;
- `protocol.json`;
- `commands.sh`.

| Setting | Value |
|---|---|
| System | deposited 3C2I (cached, SHA-256 `f96ee919…`), design preparation: Ala140, crystal waters, charged termini, 0.15 M NaCl, dodecahedron 1.2 nm |
| Minimisation | ≤ 1,000 iterations, tolerance 10 kJ mol⁻¹ nm⁻¹, k = 1000 |
| Equilibration | NVT k1000 2.5 ps; NPT k1000 5 ps; NPT k100 5 ps; NPT k10 5 ps; NPT unrestrained 10 ps (**27.5 ps**) |
| Production | **50 ps** NPT, 2 fs, 300 K, 1 bar, barostat every 25 steps |
| Output | every 0.5 ps (**100 frames**); checkpoint every 5 ps |
| Seeds | run 20261011 (velocities), 20261012 (integrator), 20261013 (barostat); none collides with the design |
| Platform | CPU, `cpu_threads: 6`, foreground, one run at a time |

**Commands** (from the project root; also in `commands.sh`):

```bash
export OPENMM_CPU_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6
O=examples/mecp2_3c2i/md/output/tier_a
A=examples/mecp2_3c2i/md/next_stage/tier_a_laptop
.venv/bin/neurodna-fetch 3C2I --cache-dir examples/mecp2_3c2i/cache --sha256 f96ee9192eabe5eafc50ff0b161c51275ad6ad8f03a2c0d0e447f18a820ed7bf
.venv/bin/neurodna-md prepare examples/mecp2_3c2i/cache/pdb/3C2I.pdb --config $A/preparation.json --output-dir $O/prepared
.venv/bin/neurodna-md run $O/prepared --protocol $A/protocol.json --output-dir $O/run
.venv/bin/python examples/mecp2_3c2i/md/diagnose_run.py --run-dir $O/run --out $O/diagnostics
```

**Stop and resume.**
- **To stop:** press Ctrl-C at any time. Work since the last checkpoint is
  lost:
  - minimisation: ≤ about 13 min;
  - an equilibration stage restarts from the previous stage's end: ≤ about
    16 min;
  - production: ≤ one 5 ps checkpoint interval, about 8 min.
- **To resume:** re-run the `run` command with `--resume`. Frames and log rows
  after the checkpoint are discarded and regenerated.
- **Planned stops:** use `--stop-after-stage <stage>` or
  `--stop-at-production-step <multiple of 2500>`.
- Resume sparingly (barostat, Warning 2).

**Tier A passes if:**
- the run completes without NaN or exceptions;
- after the first NVT stage, temperature holds at 300 ± 5 K through every restraint change;
- volume responds in the NPT stages;
- the complex stays together and the core base pairs stay intact (≥ 0.8);
- 5mC is preserved;
- `diagnose_run.py` validation shows zero mismatches.

It does not need to be equilibrated.

**Tier B, for later** (GPU; prerequisites in §9). Command:

```bash
PLATFORM=CUDA bash examples/mecp2_3c2i/md/next_stage/tier_b_gpu/commands.sh
```

Use `PLATFORM=OpenCL` for the M3 Max GPU. The script runs the step-0
benchmark first; review `ns_per_day` before continuing.

## 11. Expected compute and storage

Computed by `make_configs.py` and recorded in `next_stage/tiers.json`. It uses
measured constants: 0.899 ns/day and 0.785 s/iteration at 6 threads, and
3.64 B/atom/frame. The design system has 46,348 atoms.

| Tier | Simulated | Frames | Storage | Wall clock |
|---|---|---|---|---|
| **A** laptop, 6 threads | 1,000 min. iterations + 27.5 ps + 50 ps = 77.5 ps | 100 | 17 MB trajectory + 98 MB other = **0.12 GB** | preparation ~1–2 min; minimisation ≤ 13 min; dynamics **2.07 h**; diagnostics ~3 min: **≈ 2.3 h** (the 46k-atom system should be ~5% faster than the 49k measurement) |
| A2, optional CpG build check | minimisation only | 0 | ~40 MB | ≈ 15 min |
| **B** GPU | 4 runs × (1.35 + 5) ns = **25.4 ns** | 500 per run | 84 MB trajectory + 98 MB other per run = **0.73 GB** | GPU-hours = 25.4 ns ÷ (ns/day) × 24: **12.2 / 6.1 / 3.0 / 1.5 h** at an *illustrative* 50 / 100 / 200 / 400 ns/day. Runs are independent. On the laptop CPU it would take 28 days (not recommended) |
| **C** pilot, not run | 6 runs × 101.35 ns = 608 ns | 10,000 per run | 10.1 GB trajectory + 0.6 GB | 6.1 GPU-days at 100 ns/day |
| **C** stronger, not run | 4,405 ns new | 50,000 per run | 84 GB total | 44 GPU-days new at 100 ns/day |

## 12. What the next run could establish

**Tier A**
- The design system prepares, minimises and integrates stably through every
  restraint release.
- How the barostat behaves under strong restraints and after their release.
- Resume and QC code work on the design system.
- Refined per-stage laptop timings and file sizes.

**Tier B** (later)
- Measured GPU throughput and file sizes.
- The full 10,000-iteration minimisation and 1.35 ns ladder run on the GPU.
- Whether the design's thermodynamic flags clear in three independent
  replicates.
- Whether the duplex stays paired over 5 ns.
- A crude between-replicate spread of QC quantities and primary-pair
  occupancy, to judge whether 3 × 100 ns is a sensible pilot.
- The CpG pipeline works end to end.
- The inputs for the design's development → pilot decision rule.

## 13. What it could not establish

**Neither tier can establish:**
- equilibration or convergence of any quantity;
- contact persistence, residence times or binding kinetics;
- mCpG–CpG differences (Tier A has no CpG dynamics; Tier B has one CpG run,
  preregistered as uninterpretable);
- binding affinity or specificity;
- the size of the charged-termini effect;
- anything about full-length MeCP2, chromatin, or disease.

**Tier A cannot establish:** GPU throughput, or replicate variability.

**Tier B's 5 ns cannot establish** anything about slower processes such as
sliding, register shifts or dissociation.
