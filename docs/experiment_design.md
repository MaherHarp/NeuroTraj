# MeCP2 MBD – DNA molecular-dynamics experiment design (PDB 3C2I)

**Status: design only.** No production or pilot simulation has been run. The
only computations performed for this design were cheap validation:

- config checks;
- construction of the comparison model;
- preparation of one system per condition;
- time-zero distance tables.

They are listed in [Appendix A](#appendix-a-validation-performed-for-this-design).
The machine-readable design is in
[`examples/mecp2_3c2i/experiment/`](../examples/mecp2_3c2i/experiment/):

| File | Contents |
|---|---|
| `experiment.json` | conditions, tiers, seeds, measured constants |
| `preparation.json` | the preparation config shared by both conditions |
| `protocols/{development,pilot,stronger}.json` | the three protocol tiers |
| `analysis_plan.json` | the preregistered measurements, pairs, cutoffs, burn-in, tests and QC flags |
| `plan.py` | validates all of the above; writes per-run configs and launch scripts; never simulates |

Everything reuses the existing workflow:

- `neurodna-fetch`;
- `neurodna-md assess | prepare | run --resume | analyze`;
- the `neurodna` analysis API.

The one addition is `neurodna-md derive-unmethylated`
(`neurodna/md/variants.py`), which builds the unmethylated comparison model by
deleting atoms only.

---

## 1. Scientific question

> Which MeCP2–DNA contacts persist, and how do those patterns differ across
> simulations?

Operationally, for the MeCP2 methyl-CpG-binding domain (MBD, residues 91–162
modelled) bound to the 20-nt BDNF duplex of 3C2I:

1. **Persistence:** for each protein-residue–nucleotide pair, the fraction of
   post-burn-in time in geometric contact, and the lengths of contiguous contact
   episodes. A contact means minimum heavy-atom distance ≤ 4.5 Å.
2. **Variation across simulations:** how these quantities vary between
   independent replicates of the same system.
3. **Variation across conditions:** how they differ between the deposited
   methylated DNA (mCpG) and the same complex with the two 5-methyl groups
   removed (CpG).

"Contact" is a geometric criterion, not a hydrogen bond, energy or affinity
(see §9).

## 2. Hypotheses and descriptive comparisons

MeCP2 was identified as a protein that binds methylated DNA (Lewis et al.,
1992), through its MBD (Nan et al., 1993). Ho et al. (2008) report that in 3C2I
the 5-methyl groups contact a largely hydrophilic surface including tightly
bound water. These papers motivate methylation as the variable to perturb.
They do not predict how residue–nucleotide contact occupancies in a simulation
of this construct should change, and neither does the crystal structure.

**No directional biological hypothesis is preregistered.** The experiment makes
the following comparisons:

- **D1 (persistence, mCpG).** For each preregistered primary pair (§7.1),
  post-burn-in occupancy and episode statistics in each replicate.
- **D2 (reproducibility).** Between-replicate spread of those quantities within
  each condition.
- **D3 (condition difference).** Per primary pair: mean occupancy in mCpG minus
  mean occupancy in CpG, with every replicate value shown.
- **C1 (confirmatory, stronger tier only).** Two-sided test of D3 for a
  two-pair family: Arg111–DG9 and Arg133–DG34 (§7.3).

  - *Null hypothesis:* the replicate-mean occupancy is the same in both
    conditions.
  - *Why these pairs:* they are the arginine–guanine contacts of the symmetric
    mCpG step, and neither is mediated by the methyl carbon. So a difference
    cannot be created by the construction alone (§3.3).

The pilot is descriptive: it estimates effect sizes and variability, and runs no
tests.

## 3. Conditions

### 3.1 Choice of comparison

| Option | Assessment |
|---|---|
| **A. 3C2I mCpG vs derived CpG (chosen)** | Isolates the methylation state. Same sequence, protein, solvent model and starting coordinates. The CpG model is built by **deleting atoms only** (2 methyl carbons), and both nucleotides have documented CHARMM36 parameters. Caveats in §3.3. |
| B. Experimental mCA vs unmethylated CA complexes (6OGK / 6OGJ, Lei et al., 2019) | Both starting structures are experimental, which avoids a hypothetical complex. But this is a different recognition context (mCA, not mCpG), a different 12-bp DNA, different crystal forms and a different chain composition (6OGJ has two protein chains). It answers a different question. **Recommended later** as an orthogonal check on whether conclusions depend on starting from a methylated-bound pose. It has not been assessed for preparation here. |
| C. Hemi-methylated mCpG (one 5-methyl removed) | Chemically meaningful, but adds conditions. `derive-unmethylated` currently requires every 5CM to be listed, to keep header records unambiguous, so it would need extending. Future. |
| D. Protein variants (e.g. A140V, experimental in 5BT2 with the same BDNF DNA, Chia et al., 2016; Rett-associated substitutions) | A different question (protein variants, not methylation). Most substitutions need side-chain modelling beyond atom deletion; A140V has an experimental structure (5BT2). Out of scope. |
| E. Free DNA, mCpG vs CpG | Would separate DNA-intrinsic methylation effects. Not about MeCP2 contacts; optional aid to interpretation. |
| F. 5-hydroxymethylcytosine complexes (6YWW, Ibrahim et al., 2021; 9RPP/9RPQ) | **Blocked:** the supported force field has no validated 5hmC parameters. `neurodna-md assess` reports a blocker for 5HC. |

The RCSB search for UniProt P51608 (2026-09-25) found these entries:

- X-ray complexes with DNA: 3C2I, 5BT2, 6C1Y, 6OGJ, 6OGK, 6YWW, 9RPP, 9RPQ;
- NMR structures: 1QK9, 8AJR, 8ALQ.

Among these, no experimental structure of this MBD construct bound to
*unmethylated CpG* in the 3C2I register was identified. **The CpG complex is
therefore a computational construct.**

### 3.2 Per-condition specification

Everything not listed as differing is **identical** in the two conditions:
`preparation.json` is shared and `plan.py` checks that.

| | **mCpG** | **CpG** |
|---|---|---|
| Molecular system | 3C2I rev. 1.4 as deposited (SHA-256 `f96ee919…d7bf`): chains A (protein), B and C (DNA), plus 47 crystal waters. Assembly 1 is the identity operator. | Derived model `3C2I_CpG_derived.pdb`: same file, 5-methyl groups deleted. |
| DNA | B: 5′-TCTGGAA**[5mC]**GGAATTCTTCTA-3′; C: 5′-ATAGAAGAATTC**[5mC]**GTTCCAG-3′. A 19-bp duplex (B2–B20 · C40–C22) with single-nucleotide 5′ overhangs (B1 T, C21 A). Symmetric mCpG pairs: B8·C34 and B9·C33. | Identical sequence, with **DC** at B8 and C33 (C5A deleted; H5 added during preparation). |
| Nucleotide modifications | 5-methyl-2′-deoxycytidine at B8 and C33 | none |
| DNA force field | CHARMM36 NA; 5mC = CYT + DEOX + **5MC2** (documented patch; bonded-term audit complete) | CHARMM36 NA; DC = CYT + DEOX |
| Protein | MeCP2 residues 91–162 (UniProt P51608 numbering). Unobserved 77–90 and 163–173 are not modelled. MSE94 → Met (Se→S; no Se parameters). **Residue 140 → native Ala** (the deposited MSE140 is an engineered A140M substitution; CG, SD and CE deleted). No His or Cys. Standard pH 7 protonation, no pKa calculation. Charged termini (NH₃⁺ R91, COO⁻ R162; OXT added). | identical |
| Protein force field | CHARMM36m (OpenMM `charmm36_2024.xml`) | identical |
| Solvent | CHARMM TIP3P; rhombic dodecahedron, 1.2 nm padding; the 47 crystal waters kept | identical |
| Ions | Neutralising plus 0.15 M NaCl, with CHARMM ion parameters and the pair-specific corrections included in the OpenMM port | identical |
| Nonbonded | PME, 1.2 nm cutoff, 1.0 nm potential switch (an approximation to CHARMM's force switch), H-bond constraints | identical |
| Temperature / pressure | 300 K (Langevin middle integrator, 1 ps⁻¹); 1 bar (Monte Carlo barostat every 25 steps); 2 fs time step | identical |
| Equilibration | Restrained minimisation (≤ 10,000 iterations, k = 1000 kJ mol⁻¹ nm⁻² on solute heavy atoms). Seeded 300 K velocities. NVT 100 ps (k = 1000), then NPT 250 ps each at k = 1000, 100 and 10, then NPT 500 ps unrestrained: 1.35 ns in total | identical |
| Production | Unrestrained NPT. Development 5 ns; pilot 100 ns; stronger 500 ns | identical |
| Saving | All atoms, XTC, every 10 ps (5,000 steps), unwrapped (molecules whole); checkpoint every 1 ns | identical |
| Replicates | development 1, pilot 3, stronger 5 | identical |
| Seeds | From `experiment.json` (§5) | distinct from mCpG |

Measured prepared systems (pilot replicate 1, Appendix A):

| | mCpG | CpG |
|---|---|---|
| Atoms | 46,348 | 46,339 |
| Box volume | 472.95 nm³ | 473.01 nm³ |
| Ions | 75 Na⁺, 40 Cl⁻ | 75 Na⁺, 40 Cl⁻ |

The solute charge is −35 before ions and exactly 0 after, in both systems.

### 3.3 Problems with the CpG construct (documented, not approximated)

1. **It is a hypothetical complex.** It starts from the methyl-specific bound
   pose. Simulations lasting nanoseconds to microseconds measure how contacts
   in *this pose* respond to losing the methyl groups. They do not sample
   alternative binding registers, sliding or dissociation. They also do not
   tell us what an MBD–unmethylated-DNA complex looks like.
2. **Common-start bias.** Both conditions, and all replicates, begin from the
   same crystal coordinates. Agreement between replicates is therefore
   agreement *from this start*, not independence from it.
3. **Crystal waters were refined against the methylated complex.** Six crystal
   waters lie within 4.5 Å of the two methyl carbons: A:HOH180, A:HOH182,
   A:HOH183, A:HOH187, B:HOH32 and C:HOH43. B:HOH32 is near both, and the
   nearest is 3.37 Å from C5A of 5CM33.

   They are kept in both conditions so that the starting hydration is
   identical. In CpG they initially sit where they were modelled around a
   methyl group. Removing crystal waters instead would put an
   algorithm-dependent water arrangement exactly at the site of interest.
   Removing them in both conditions is a sensitivity analysis (exploratory).
4. **Some differences are created by the construction.** At time zero, 17 of
   the 19 primary pairs have identical minimum distances in the two
   conditions. The two **methyl-mediated** pairs, whose closest crystal atom
   is C5A, change by construction:

   | Pair | mCpG | CpG |
   |---|---|---|
   | Asp121–position 8 | 3.48 Å (to C5A) | 4.60 Å (to C5) |
   | Tyr123–position 8 | 4.09 Å (to C5A) | 5.03 Å (to N4) |

   Both are flagged in `analysis_plan.json` and excluded from the confirmatory
   family. Their differences are reported, never interpreted as a dynamic
   consequence of demethylation alone.
5. **Residue 140 is a deliberate choice.** The deposited engineered Met140 side
   chain reaches 4.93 Å from a DNA phosphate (CE–OP2). The native Ala CB is
   8.64 Å away. Reverting to Ala uses the native sequence and deletes atoms
   only, so it is applied to both conditions. An as-crystallised Met140 run
   would be an exploratory sensitivity analysis.

## 4. Controls

| Control | Purpose |
|---|---|
| Independent replicates (§5) | Reproducibility; they set the scale of chance differences |
| CpG derived model | Methylation perturbation, with the caveats in §3.3 |
| Time-zero table (Appendix A) | Baseline; separates construction effects from dynamic ones |
| Methyl-mediated pairs (Asp121/Tyr123–position 8) | Check that the residue correspondence (5CM8 ↔ DC8) is applied correctly; their time-zero difference is known |
| Within-condition split (exploratory, stronger tier) | e.g. replicates 1–2 vs 3–5 of the same condition. Shows the size of differences that arise with no condition difference |
| Parameter audit | Already done: every bonded term around the 5mC methyl is present; term-less torsions are zero-force-constant in CHARMM36 |
| Pipeline controls | Synthetic-fixture tests of occupancy and episode arithmetic; the 3C2I smoke run for end-to-end mechanics |

## 5. Replicate strategy

- **The replicate is the statistical unit.** Frames are serially correlated and
  are never used as independent observations. All uncertainty is computed over
  replicates.
- **Independence.** Each replicate gets its own preparation, which re-seeds
  hydrogen placement and the choice of waters replaced by ions, and its own
  velocity, integrator and barostat seeds. Solute starting coordinates are
  shared by design (§3.3, point 2).
- **Seeds.** Every run is listed explicitly in `experiment.json`, using this
  scheme:
  - `preparation_seed` = 1,000,000 + condition (100,000 mCpG / 200,000 CpG)
    + tier (1,000 development / 2,000 pilot and stronger) + 10 × replicate;
  - `run_seed` follows the same scheme with 2,000,000;
  - a run uses `run_seed`, +1 and +2, and `plan.py` verifies that every derived
    seed is unique.
- **Tiers share runs.** Stronger replicates 1–3 are the pilot runs extended
  from their checkpoints with `--resume`. `plan.py` verifies that only
  `production_steps` differs, and neurodna refuses to resume otherwise.
  Replicates 4–5 are new. So the pilot is not wasted, and the stronger
  experiment's first 100 ns are the pilot.
- **Analysis is unpaired.** Replicate *i* of mCpG and replicate *i* of CpG are
  not matched.
- **Why five replicates.** With 5 vs 5, an exact permutation test over
  replicate labels has 252 splits, so the smallest attainable two-sided p-value
  is 2/252 ≈ 0.008. That falls below the Holm threshold for a two-test family
  (0.025). With 3 vs 3, no test is meaningful, which is why the pilot is
  descriptive. This is also why the confirmatory family is limited to two
  pairs: with 19 pairs, 5 vs 5 could never pass a multiplicity correction.

## 6. Simulation protocol

| Tier | Replicates per condition | Production per replicate | Frames per replicate | Purpose | Interpretation |
|---|---|---|---|---|---|
| **Development** | 1 | 5 ns | 500 | Measure GPU throughput and file sizes; validate QC and analysis code on real output | None |
| **Pilot** | 3 | 100 ns | 10,000 | Replicate variability, effect sizes, QC behaviour | Descriptive |
| **Stronger** | 5 | 500 ns (pilot replicates extended to 500 ns, plus 2 new) | 50,000 | Stability of estimates over a 5× longer window; confirmatory family C1 | Descriptive + C1 |

Every tier uses the 1.35 ns staged equilibration of §3.2. The first 10 ns of
production are additionally excluded from analysis as burn-in (sensitivity: 0
and 20 ns).

**What these lengths can establish.** At 100 ns, contacts that break and
re-form on nanosecond to tens-of-nanoseconds timescales can be observed
repeatedly within a replicate. Slower processes will be absent or seen as
isolated events: dissociation, sliding, register changes and large
rearrangements of the DNA or protein. 500 ns widens that window. **No duration
guarantees convergence.** Convergence is judged per quantity with the §8
diagnostics, and a contact present for a whole trajectory gives only a lower
bound on its persistence.

**Decision rules between tiers** (preregistered):

- **Development → pilot**, if all of the following hold:
  - both development runs finish without technical failure (crash, NaN,
    thermostat or barostat failure);
  - the §8 thermodynamic flags are clear;
  - DNA core base pairs are intact (Watson–Crick fraction ≥ 0.8);
  - the QC and aggregation code (Appendix B) has been validated on these
    outputs;
  - measured GPU throughput is recorded.
- **Pilot → stronger** is decided **only on QC and stability diagnostics, never
  on the size or direction of mCpG–CpG differences**, to avoid outcome-dependent
  stopping. The stronger tier runs if the pilot has no unresolved technical or
  model problem, such as DNA unpairing at the mCpG step or loss of all primary
  contacts for a technical reason.
- **Failed replicates.** A replicate is excluded only for technical failure,
  and is then replaced with the next seed and reported. Biological events such
  as dissociation are results and are never grounds for exclusion.

## 7. Planned neurodna analyses

Definitions, cutoffs and pairs are fixed in `analysis_plan.json` before any
production data exist. All distances are heavy-atom only, in Å; times are in ps
or ns.

### 7.1 Preregistered primary pairs

**Rule:** every residue–nucleotide pair with minimum heavy-atom distance ≤ 4.5 Å
in the 3C2I crystal model that involves an mCpG nucleotide (B8, B9, C33, C34),
plus Thr158–DT31. Thr158 is the residue that Ho et al. (2008) single out: the
residue most commonly mutated in Rett syndrome, with a structural role in the
complex. The rule gives **19 pairs**:

- **Position B8** (5mC/C): Arg111, Lys112, Ser113, Ser116, Asp121\*, Tyr123\*,
  Arg133.
- **B9** (G): Arg111†, Ser113, Gly114, Arg115, Ser116.
- **C33** (5mC/C): Arg111, Asp121, Arg133, Ser134, Val136.
- **C34** (G): Arg133†.
- **C31**: Thr158.

\* methyl-mediated in the crystal (§3.3, point 4). † confirmatory family C1.

These are chosen from crystal geometry; they are measurements to report, not
hypotheses about function. **Everything else is exploratory** and is labelled
as such.

**Residue correspondence between conditions:** match by chain and number
(B:5CM8 ↔ B:DC8, C:5CM33 ↔ C:DC33). The protein is identical in both.

### 7.2 Per-replicate measurements

Run on the production trajectory after the burn-in. Frames are 10 ps apart and
the first frame is at t = 10 ps, so `start=1000` begins at t = 10.01 ns.

```python
from neurodna.md.analyze import load_run
cx = load_run(run_dir)                         # checks checksum, frame count, spacing
occ  = cx.contact_occupancy(4.5, start=1000)   # all pairs; primary pairs reported explicitly
eps  = cx.contact_episodes(4.5, start=1000)    # observed durations (lower bounds), censoring
dist = cx.min_distances(12.0, start=1000)      # primary pairs: median, 10th/90th percentile (absent = > 12 Å)
rmsd = cx.backbone_rmsd(reference_frame=1000, start=1000)
rmsf = cx.rmsf("name CA", reference_frame=1000, start=1000)
# sensitivity: cutoff 4.0 and 5.0 Å; burn-in 0 and 20 ns (start=0, start=2000)
```

For each primary pair and replicate this gives:

- occupancy;
- closest distance;
- number of episodes;
- median observed episode duration, reported as a lower bound;
- fraction of contact time in censored episodes;
- median, 10th and 90th percentile of the minimum distance.

`neurodna-md analyze` produces the same tables over the whole production
(no burn-in) for quick inspection.

### 7.3 Between-condition comparison

- **Per primary pair:** the replicate values of each condition, the mean per
  condition, and the difference Δ = mean(mCpG) − mean(CpG).
- **Pilot (3 vs 3):** Δ and replicate ranges only. No p-values; no bootstrap
  intervals, which are unreliable with 3 replicates.
- **Stronger (5 vs 5):**
  - Δ with a bootstrap interval over replicates (10,000 resamples);
  - for **C1 only**, the exact two-sided permutation test over replicate labels
    (252 splits), with Holm correction over the 2 pairs;
  - all other pairs remain descriptive.
- **Episodes** are compared through per-replicate summaries, never by pooling
  episodes across replicates as if independent.

### 7.4 Exploratory analyses (reported as exploratory)

- All-pair occupancy maps and their differences.
- Contacts to the 5-methyl marker atom (mCpG only).
- Distances from Arg111, Arg133, Asp121 and Tyr123 to C5 of positions 8 and 33.
  The C5 atom exists in both conditions.
- Water-mediated contacts at the mCpG step.
- Per-residue RMSF differences.
- The within-condition split control.
- Crystal-water-removed and Met140 sensitivity runs, if performed.

## 8. Quality-control and convergence checks

Each check is computed per replicate. Flags trigger inspection and are reported
alongside results; the thresholds are pragmatic, preregistered choices, **not
proof of convergence**. RMSD flattening alone is never taken as convergence.

| Check | Data | Flag |
|---|---|---|
| Temperature | `production_log.csv` | mean outside 298.5–301.5 K |
| Potential energy, box volume, density | `production_log.csv` | first- and second-half means differ by more than 3 block standard errors (blocks ≥ 1 ns) |
| Protein backbone RMSD | `backbone_rmsd` (all backbone N CA C O, including the flexible termini) | reported as a distribution and time series; a trend over the last half is noted, but is not a pass/fail criterion |
| Protein RMSF | `rmsf("name CA")` | reported; replicate agreement per residue |
| DNA structural stability | core duplex heavy-atom RMSD (B4–B18 · C24–C38; excludes the overhangs and the 2 terminal base pairs at each end); Watson–Crick N1–N3 ≤ 3.2 Å per base pair, including the mCpG pairs B8·C34 and B9·C33 | any core base pair intact in < 80% of frames |
| Protein–DNA association | residue pairs in contact per frame (`min_distances`) | no primary pair in contact for > 10 ns |
| Occupancy stability within a replicate | `contact_occupancy` on the first 25, 50, 75 and 100% of post-burn-in frames (`stop=`), and on 5 equal blocks | \|occ(50%) − occ(100%)\| > 0.10 for a primary pair |
| Replicate agreement | primary-pair occupancy range and SD across replicates; whole-map mean absolute difference and Spearman correlation between replicates | range > 0.30 for a primary pair |
| Episode censoring | `contact_episodes` | a primary pair whose episodes are mostly censored: persistence ≥ trajectory, no duration estimate |

A primary-pair estimate is labelled **"stable within this design"** only if its
occupancy-stability flag is clear in every replicate **and** its replicate range
is ≤ 0.30. Otherwise it is reported as unresolved, with the diagnostics shown.

## 9. Interpretation limits

- **Geometric contacts** are not hydrogen bonds, salt bridges, energies or
  binding affinities. Nothing here estimates ΔG, specificity or affinity
  differences, and an absence of occupancy differences says nothing about
  affinity.
- **Contact episodes** are sampling-limited, per-pair durations. They are
  **not** protein–DNA binding residence times. Transitions are resolved only to
  the 10 ps saving interval.
- **The CpG complex is hypothetical** and shares a start with mCpG (§3.3).
  Results describe responses from the methylated-bound pose over the simulated
  time.
- **This is a domain.** Residues 91–162 of a 486-residue protein were modelled.
  Charged termini sit at artificial truncation points about 7 Å from DNA; one
  20-nt BDNF-derived duplex was used; there is no chromatin, and there are no
  crystal-lattice contacts.
- **Force field.** The 5mC parameters (CHARMM36 `5MC2`, CN3D terms annotated
  1993) are documented and distributed, but not independently validated here,
  and a methylation effect in the simulation is partly a property of these
  parameters. Potential switching approximates CHARMM's force switching.
  Protonation uses standard states at pH 7.
- **Replicates.** Five replicates measure reproducibility of the dynamics from
  one structure, not structural uncertainty in the crystal model. Frames are
  never statistical replicates.
- **No disease or pathogenicity claims.** The mention of Thr158 follows the
  literature; no variant is simulated.

## 10. Estimated compute and storage

**Simulated time** (from the protocols; equilibration counted once per new run):

| Tier | Runs | New simulated time |
|---|---|---|
| Development | 2 | 12.7 ns |
| Pilot | 6 | 608.1 ns |
| Stronger | 10 (6 extended + 4 new) | 4,405.4 ns new (5,013.5 ns in total, including the pilot) |

**Throughput.**

- **Measured CPU speed:** 1.55 ns/day (Apple M3 Max, 14 threads, CPU
  platform, 49,302-atom 3C2I build). At that speed the pilot would take about
  392 machine-days and the stronger tier about 2,842 more. So a GPU is
  required.
- **GPU throughput has not been measured.** The development run measures it:
  `ns_per_day` per stage in `simulation.json`, CUDA and mixed precision.

GPU-days follow as *simulated ns ÷ measured ns/day*. The table below uses
**illustrative throughputs, not measurements**:

| Tier | at 50 ns/day | 100 ns/day | 200 ns/day | 400 ns/day |
|---|---|---|---|---|
| Development | 0.3 | 0.1 | 0.1 | < 0.1 |
| Pilot | 12.2 | 6.1 | 3.0 | 1.5 |
| Stronger (new) | 88.1 | 44.1 | 22.0 | 11.0 |

Runs are independent, so wall time divides by the number of GPUs used in
parallel. Each pilot run takes 101.35 ns ÷ throughput.

**Storage.** Based on the measured 3.64 bytes per atom per XTC frame (from the
3C2I smoke trajectory) and 46,348 atoms:

| Tier | Trajectory per replicate | Trajectory in total | Other files |
|---|---|---|---|
| Development | 0.08 GB | 0.17 GB | 0.2 GB |
| Pilot | 1.69 GB | 10.1 GB | 0.6 GB |
| Stronger | 8.43 GB | 84.3 GB (74.2 GB new) | about 1 GB |

"Other files" covers checkpoints, stage states, `system.xml` and topologies.
Analysis tables are small by comparison: primary-pair distances over 50,000
frames are about tens of MB.

## 11. Commands

Validate the design and generate per-run configs and launch scripts. This runs
nothing expensive:

```bash
python examples/mecp2_3c2i/experiment/plan.py \
    --plan-dir  examples/mecp2_3c2i/experiment/plan \
    --experiment-dir /path/to/storage/mecp2_3c2i_runs \
    --atoms 46348 --platform CUDA
```

It writes `plan/<tier>/<condition>/r<i>/{preparation,protocol}.json` and
`plan/commands_{development,pilot,stronger}.sh`. The pilot script contains, per
run:

```bash
neurodna-fetch 3C2I --cache-dir examples/mecp2_3c2i/cache --sha256 f96ee9192eabe5eafc50ff0b161c51275ad6ad8f03a2c0d0e447f18a820ed7bf
neurodna-md derive-unmethylated examples/mecp2_3c2i/cache/pdb/3C2I.pdb --residues B:8 C:33 \
    --output $RUNS/inputs/3C2I_CpG_derived.pdb
# mCpG replicate 1
neurodna-md prepare examples/mecp2_3c2i/cache/pdb/3C2I.pdb \
    --config $PLAN/pilot/mCpG/r1/preparation.json --output-dir $RUNS/pilot/mCpG/r1/prepared
neurodna-md run $RUNS/pilot/mCpG/r1/prepared --protocol $PLAN/pilot/mCpG/r1/protocol.json \
    --output-dir $RUNS/pilot/mCpG/r1/md --platform CUDA
# ... likewise for mCpG r2, r3 and CpG r1-r3 (CpG uses the derived input)
```

The stronger script extends pilot replicates 1–3 in place and prepares and runs
replicates 4–5:

```bash
neurodna-md run $RUNS/pilot/mCpG/r1/prepared --protocol $PLAN/stronger/mCpG/r1/protocol.json \
    --output-dir $RUNS/pilot/mCpG/r1/md --platform CUDA --resume
```

**Order of work:**

1. development script;
2. implement and validate the Appendix B analysis code on its output;
3. pilot script;
4. pilot QC and descriptive report;
5. stronger script, if the §6 rules allow;
6. confirmatory analysis.

Record the git commit of neurodna and the `plan.py` output with every run
directory.

---

## Appendix A. Validation performed for this design

All of this was done on 2026-09-25, using neurodna in this repository and OpenMM
8.6.1 on the CPU. No MD was integrated.

1. **Config validation.** `plan.py` validated the design, generating and
   reloading 36 per-run config files through `PreparationConfig` and
   `Protocol`. Every seed is unique, and the stronger tier can resume the
   pilot. The same checks run in `tests/test_experiment_design.py`.
2. **Deriving the CpG model.** `neurodna-md derive-unmethylated` removed
   exactly two atoms (C5A of B:5CM8 and C:5CM33) and renamed the residues DC.
   It dropped or updated 58 header records (MODRES, HET, HETNAM, FORMUL, LINK
   and CONECT for 5CM; SEQRES and COMPND text), marked the file as a derived
   model in `REMARK 999`, and wrote a provenance JSON. All other atom records
   are byte-identical to the deposited file (`tests/test_md_offline.py`).
3. **Preparation.** `neurodna-md assess` and `prepare` succeeded for both
   conditions with the pilot replicate-1 configs, about 60 s each, with no
   blockers:
   - mCpG: 5CM8 and 5CM33 matched `CYT-DEOX_0-5MC2_4`, and the bonded-term
     audit is complete.
   - CpG: DC8 and DC33 matched `CYT-DEOX_0`.
   - Residue 140 is Ala in both.
   - Composition, box and ions are as in §3.2.
   - Each preparation also evaluates the per-force energies of the full system
     (the energy fingerprint), confirming both systems can be computed.
4. **Time-zero primary-pair distances** (prepared structures, before any
   dynamics): identical in the two conditions except the two methyl-mediated
   pairs (§3.3, point 4).

## Appendix B. Not yet implemented (needed before the pilot is analysed)

These must be written, unit-tested and run on the development output **before**
the pilot is analysed:

- DNA core RMSD and Watson–Crick pair integrity, using MDAnalysis directly;
  neurodna has no DNA RMSD yet.
- Thermodynamic half and block statistics from `production_log.csv`.
- Cumulative and block occupancy and replicate-agreement summaries. These use
  the existing `contact_occupancy(start=, stop=)`.
- Cross-condition aggregation with the residue correspondence, the bootstrap
  over replicates, and the exact permutation test for C1.

Launching the development run does not depend on them.

## References

- RCSB PDB 3C2I; Ho KL, et al. (2008) MeCP2 binding to DNA depends upon hydration at
  methyl-CpG. *Mol Cell* 29:525–531. doi:10.1016/j.molcel.2007.12.028 (PMID 18313390).
- Lewis JD, et al. (1992) *Cell* 69:905–914. doi:10.1016/0092-8674(92)90610-o (PMID 1606614).
- Nan X, Meehan RR, Bird A (1993) *Nucleic Acids Res* 21:4886–4892. doi:10.1093/nar/21.21.4886 (PMID 8177735).
- Chia JY, et al. (2016) A/T run geometry of B-form DNA is independent of bound methyl-CpG
  binding domain, cytosine methylation and flanking sequence. *Sci Rep* 6:31210.
  doi:10.1038/srep31210 (PMID 27502833). [PDB 5BT2]
- Lei M, et al. (2019) Plasticity at the DNA recognition site of the MeCP2 mCG-binding domain.
  *Biochim Biophys Acta Gene Regul Mech* 1862:194409. doi:10.1016/j.bbagrm.2019.194409
  (PMID 31356990). [PDB 6C1Y, 6OGJ, 6OGK]
- Ibrahim A, et al. (2021) MeCP2 is a microsatellite binding protein that protects CA repeats
  from nucleosome invasion. *Science* 372. doi:10.1126/science.abd5581 (PMID 34324427). [PDB 6YWW]
- CHARMM36 `toppar_c36_jul24`, `stream/na/toppar_all36_na_modifications.str` (`PRES 5MC2`),
  as verified in `examples/mecp2_3c2i/md/README.md`.
