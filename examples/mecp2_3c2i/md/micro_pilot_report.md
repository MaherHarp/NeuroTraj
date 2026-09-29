# Micro-pilot report: 3C2I MeCP2 MBD – methylated DNA (laptop, CPU)

> **THIS RUN IS NOT EQUILIBRATED OR CONVERGED AND CANNOT ESTABLISH MeCP2–DNA BEHAVIOUR.**
>
> It is 25 ps of dynamics: 5 ps of restrained equilibration and 20 ps of
> production. It was run only to check numerical stability, trajectory
> writing, checkpoint/resume, and neurodna analysis on a trajectory slightly
> longer than the smoke test.
>
> Box volume, density and potential energy were still drifting when it ended.
> Every time-dependent quantity below is a **diagnostic of the software and the
> integration only**: RMSD, RMSF, occupancy and episodes. None is a biological
> result. Differences between individual contacts must not be interpreted.

Date: 2026-09-25 to 2026-09-26. **Follow-up analysis:** see `pilot_analysis_report.md`. It covers:

- bulk-water density;
- independent validation of neurodna;
- barostat behaviour across resumes;
- the readiness checklist and next tiers.

It also resolves warnings 3 and 6 below. Outputs are in `examples/mecp2_3c2i/md/output/micro_pilot/`
and `…/micro_pilot_logs/` (git-ignored).

## 1. Protocol

`examples/mecp2_3c2i/md/protocols/micro_pilot.json`:

| Setting | Value |
|---|---|
| System | the existing prepared, smoke-tested 3C2I system (`md/output/prepared`, `prepared.pdb` SHA-256 `48937243…`). Methylated (5CM B8, C33 on CHARMM36 `5MC2`); residue 140 kept as the crystallised Met; 49,302 atoms; CHARMM36 (July 2024), CHARMM TIP3P, 0.15 M NaCl; rhombic dodecahedron, 500.5 nm³ initially |
| Minimisation | existing smoke procedure: ≤ 100 L-BFGS iterations, restraints k = 1000 kJ mol⁻¹ nm⁻² on solute heavy atoms, tolerance 10 kJ mol⁻¹ nm⁻¹ |
| Equilibration | NVT 1,250 steps (2.5 ps), k = 1000; then NPT 1,250 steps (2.5 ps), k = 100 |
| Production | unrestrained NPT, 10,000 steps (20 ps) |
| Integrator | Langevin middle, 2 fs, 300 K, friction 1 ps⁻¹; Monte Carlo barostat, 1 bar, every 25 steps (NPT stages only) |
| Seeds | protocol seed 20260926: velocities 20260926, integrator 20260927, barostat 20260928 |
| Output | XTC every 125 steps (0.25 ps), unwrapped (`enforcePeriodicBox=False`); checkpoint every 1,250 steps (2.5 ps) |
| Platform | OpenMM 8.6.1 CPU platform, **6 threads** (`cpu_threads: 6`; `OPENMM_CPU_THREADS=6`, `OMP_NUM_THREADS=6` and `VECLIB_MAXIMUM_THREADS=6` also set). The machine has 14 cores (10 performance + 4 efficiency). OpenCL is available only outside this session's sandbox, so it was not used. |

**Workflow additions made for this run.** Both are backward-compatible and
covered by offline tests.

- An optional protocol field `cpu_threads`, passed to OpenMM's CPU platform. Resume ignores it, because
  it changes speed, not physics.
- The flags `--stop-after-stage` and `--stop-at-production-step`, which end a session cleanly at a
  checkpoint.

They were needed because commands in this session are cut off after 10 minutes
and background simulations were not allowed. The run was therefore executed as
**11 sequential foreground sessions**, each resuming from the previous
checkpoint:

1. minimisation;
2. the NVT stage;
3. the NPT stage;
4. production in eight 2.5 ps chunks.

Exact commands, each printed before it ran and stored in
`micro_pilot_logs/session*.cmd`:

```bash
E="OPENMM_CPU_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6"
R="examples/mecp2_3c2i/md/output/prepared --protocol examples/mecp2_3c2i/md/protocols/micro_pilot.json --output-dir examples/mecp2_3c2i/md/output/micro_pilot"
$E .venv/bin/neurodna-md run $R --stop-after-stage minimization            # session 1
$E .venv/bin/neurodna-md run $R --resume --stop-after-stage nvt_restrained # session 2
$E .venv/bin/neurodna-md run $R --resume --stop-after-stage npt_restrained # session 3
$E .venv/bin/neurodna-md run $R --resume --stop-at-production-step 1250    # session 4
# ... 2500, 3750, 5000, 6250, 7500, 8750                                   # sessions 5-10
$E .venv/bin/neurodna-md run $R --resume                                   # session 11 (to completion)
```

(They were typed out literally in zsh; this block uses shell variables only for readability.)

## 2. Simulated duration

| Stage | Simulated time |
|---|---|
| Minimisation | 100 iterations |
| Restrained equilibration | **5.0 ps** |
| Production | **20.0 ps** (80 frames, t = 0.25 … 20.00 ps) |
| **Total dynamics** | **25.0 ps** |

## 3. CPU threads

6 OpenMM threads, confirmed by the platform property `Threads = 6` in
`simulation.json`. Measured average CPU use per session (CPU time ÷ wall time)
was 5.6–5.9 cores. Simulations ran one at a time, in the foreground only.

## 4. Wall-clock runtime

| Item | Wall time |
|---|---|
| 11 simulation sessions (incl. setup) | **2,512.7 s = 41.9 min** |
| Minimisation | 78.5 s |
| Dynamics | 2,402.0 s |
| Setup per session | about 3 s |

- **Pause:** there was a user-initiated pause of about 24 h between sessions 6 and 7. It is not
  included above.
- **Failed invocation:** one additional invocation failed in 0.6 s because of a shell-quoting mistake.
  No simulation started (§10).

Pre-run estimate for comparison: 21–23 min of dynamics at the previously measured
14-thread speed. The pessimistic 6-thread bound (speed scaling linearly with
thread count) was 49–54 min. The actual dynamics took 40.0 min.

## 5. Measured throughput (6 threads)

| Stage | ns/day |
|---|---|
| NVT restrained | 0.998 |
| NPT restrained | 0.921 |
| Production chunks | 0.866–0.905 |
| **All 25 ps of dynamics** | **0.899** |

For comparison, the earlier measurement with 14 threads was 1.55–1.72 ns/day.
On this machine, the 6-thread run achieved about 52–58% of that speed.
Minimisation took 78.5 s for 100 iterations, against 46.5 s with 14 threads.

## 6. Trajectory size

| File | Size |
|---|---|
| `production.xtc` (80 frames, all 49,302 atoms) | **14,393,320 bytes** (179.9 KB per frame) |
| checkpoints (`.chk`), each | 2.38 MB |
| stage states (`.xml`), each | 8.19 MB |
| `topology.pdb` | 4.0 MB |
| Whole run directory | 61 MB |

## 7. Checkpoint/resume

**Tested and working.** There were ten resumes:

- two resumes into equilibration stages, from stage checkpoints;
- seven resumes into production from `production.chk`;
- one resume to completion.

One resume came after a pause of about 24 h. The resulting trajectory has
exactly 80 frames. The last frame of each chunk falls at t = 2.5, 5.0, …,
20.0 ps, with uniform 0.25 ps spacing, so no frames were duplicated or lost.
`progress.json` ends with every stage completed and 10,000 production steps.
`simulation.json` records each production chunk with its `resumed_from_step`.

## 8. Numerical stability

| Check | Result |
|---|---|
| NaNs / integration failures | none: no OpenMM exceptions, no NaN or error lines in any session's output, all logged energies finite |
| Minimisation | potential energy −174,043 → −783,486 kJ/mol |
| Temperature | Rose from 249.8 K to 299.9 K during equilibration (velocities assigned at 300 K after minimisation). Production: 300.8 ± 1.4 K (range 297.0–304.9 K; half means 301.4 / 300.3 K) |
| Potential energy (production) | −726,254 ± 1,202 kJ/mol; half means −725,363 / −727,145 kJ/mol (**still drifting**) |
| Box volume | 500.5 nm³ at the start of NPT → 489.7 nm³ at the end of restrained NPT. Production: 481.6 ± 2.0 nm³; half means 482.9 / 480.4 nm³ (**still decreasing**) |
| Density | 1.010 → 1.033 g/mL during restrained NPT. Production: 1.050 ± 0.004 g/mL; half means 1.047 / 1.052 (**still drifting**) |
| Frame times | 80 frames, strictly increasing, uniform 0.25 ps spacing, real times from the XTC file |
| Molecules whole | Protein: neurodna's whole-molecule check (bonds and residue extent) passed in every frame during RMSD/RMSF. DNA: the maximum O3′–P link distance over all 38 links and 80 frames is 1.682 Å. Water: the maximum O–H1 distance is 0.973 Å. No splitting across periodic boundaries. |

## 9. neurodna analysis (diagnostic only)

`neurodna.md.analyze.load_run` accepted the run. It checked the trajectory
checksum, frame count and frame spacing, then built a trajectory-kind complex
with file times and uniform sampling. Minimum-image distances were used, since
the XTC box is a valid P1 simulation box.

| Quantity | Result (**diagnostic; not interpretable at 20 ps**) |
|---|---|
| Contacts (≤ 4.5 Å, heavy atoms) | computed: 46 residue–nucleotide pairs in contact in at least one frame |
| Contact episodes | computed: 198 episodes |
| Minimum distances (≤ 12 Å) | computed: 29,523 rows |
| Contacts to the 5mC methyl carbons | computed (Arg111, Asp121 and Tyr123 with B:5CM8; Arg133 and Glu137 with C:5CM33) |
| Protein backbone RMSD (to the first production frame) | mean 0.88 Å, maximum 1.15 Å, last frame 1.00 Å |
| CA RMSF | median 0.41 Å, maximum 0.98 Å (A:GLY129). Of limited meaning over 20 ps. |
| Preregistered primary pairs (`experiment/analysis_plan.json`) | 14 of 19 have occupancy 1.0 over the 80 frames; five do not (Arg111–C:5CM33 0.94, Ser113–B:DG9 0.93, Arg133–B:5CM8 0.66, Asp121–C:5CM33 0.04, Tyr123–B:5CM8 0.03) |

The occupancy values above come from 80 highly correlated frames covering 20
ps of an unequilibrated system, starting from the crystal pose. **They show
only that the analysis runs. They do not measure contact persistence and must
not be compared or interpreted.**

## 10. Warnings and problems

1. **A shell-quoting mistake, not a simulation failure.** The first attempt at session 2 passed its
   arguments through an unquoted zsh variable, which zsh did not split. `neurodna-md` exited with a
   usage error after 0.6 s, no simulation started, and the run state was unchanged. It was re-issued
   with literal arguments. Its log was overwritten by the successful session 2 log. No simulation was
   repeated.
2. **Not equilibrated.** Volume, density and potential energy were still drifting at the end of
   production. That is expected after 5 ps of equilibration.
3. **Density check needed.** Production density reached about 1.05 g/mL and was still rising. Whether
   this value is appropriate for this solution with CHARMM TIP3P at 300 K was not assessed. The planned
   development run should report the density after a longer equilibration and compare it with a
   solvent-only box under the same settings.
4. **Truncated minimisation.** Only 100 iterations were run (the smoke procedure), not the 10,000 in
   the experiment design's protocols. That full procedure is still untested.
5. **Not the planned experimental system.** This is the as-crystallised Met140 system, not the
   experiment design's mCpG system (native Ala140). The latter has so far only been prepared, not
   simulated.
6. **OpenMM integration tests not re-run.** They were not re-run after adding `cpu_threads` and the
   stop options, to avoid extra simulations on the laptop. The pilot exercised the new paths
   (stage stops, production stops, resumes and a run to completion). The 158 offline tests pass.
7. **MDAnalysis offset files.** MDAnalysis wrote XTC offset cache files
   (`.production.xtc_offsets.*`) next to the trajectory. They are harmless.

## 11. Statement

**This 25 ps run is neither equilibrated nor converged. It cannot establish anything about MeCP2–DNA
behaviour, contact persistence, or methylation effects.** It shows only that the prepared system
integrates stably on the CPU for 25 ps, and that the software pipeline works:

- trajectory writing with real times;
- whole molecules in the output;
- repeated checkpoint/resume;
- neurodna analysis.

The experiment design (`docs/experiment_design.md`) was not changed in response to this run.
