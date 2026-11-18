# Lexicon — Spec

**Status:** frozen predeclaration (L0 v1.0) plus the v1.1 amendments, both
written before any held-out biological result was computed. The canonical L0
run has since been executed; its outcome is reported in `README.md` and
`runs/l0-v1/results.json`, not here.
**Created:** 2026-08-06.
**Authorship:** framed and verified by an AI agent (Claude-based); the L0
protocol itself was drafted and reviewed by a second, independent LLM from a
different model family, acting as an adversarial reviewer. Referred to below
as "the reviewer".

> **Editorial note for this public copy.** The project-context preamble below
> has been shortened and references to private projects and people removed.
> The L0 protocol (§1–§7) and the v1.1 amendments are unchanged in substance;
> only attributions were replaced with "the reviewer".

---

## The thesis

An LLM does not simulate word interactions; it compresses **reusable
transformations of language** into weights. Lexicon asks whether the same
holds for cellular response: can weights encode a compact, reusable
**grammar of non-additive perturbation response** — composition rules, not
biochemical constants?

The binding constraint is NOT compute. It is **the teacher**. In physics a
simulator can generate unlimited ground truth from known governing equations.
Biology is underdetermined: no governing equations, unmeasured coupling
constants, no validated cross-scale bridge. So the design question is what
generates ground truth, and what architecture learns from a sparse teacher.

**The distinction that makes the question testable:** other people's
*conclusions* are meta-analysis (out of scope). Other people's *experiments*
are measurement (allowed). Predicting an unseen perturbation dataset is a
test set, not a literature review.

**Why not reuse a physics-style "exact spine".** A gene regulatory network has
uncertain context-dependent edges, partially observed state, interventions
instead of dense trajectories, no conservation law, and sparse endpoint
labels. Treating a known pathway as an exact spine is actively dangerous —
pathways are declared priors with uncertainty, not laws. What carries over
from physics-style learned simulation is the discipline, not the code:
spine + residual (here the spine is the *measured* additive effect),
architecture before parameter scale, strong baselines, predeclared kill
gates, and independent verification.

## The long-range acceptance channel

A learned grammar only matters for health if, eventually, something outside
this repository can say yes or no. One such external verifier exists and is
open to outside nominations:

- **NIA ITP** (Interventions Testing Program) — accepts proposals from "all
  scientific quarters, including U.S. and foreign universities, commercial
  entities, and even individuals without institutional affiliations."
  Three independent sites, genetically heterogeneous UM-HET3 mice, followed
  to natural death, powered to detect a 10% lifespan increase.
- **NIA CITP** (*Caenorhabditis* Intervention Testing Program) — also accepts
  outside nominations; faster, but **not a validated mouse surrogate**.

## The honest ladder

| Rung | Claim earned | Status at time of writing |
|---|---|---|
| 0 | Perturbation grammar: sparse non-additive rules compress into weights | **This spec** |
| 1 | Drug grounding: structure/target/dose invokes those programs for unseen compounds | Unbuilt, testable |
| 2 | Context coordinate: new cell type/donor adapts from k obs via low-rank correction | Unbuilt, testable |
| 3 | Frozen response codec, frozen without lifespan-label access | Unbuilt, testable |
| 4 | **Lifespan bridge** — frozen features carry lifespan info beyond chemical class/dose/sex | **Pivotal.** Severely data-limited |
| 5 | Cross-species auxiliary teacher improves mouse ranking under frozen transfer rule | Unbuilt, testable |
| 6 | Candidate selection: ranks an untested, obtainable, chow-stable candidate | Unbuilt |
| 7 | ITP nomination submitted | Reachable in principle |
| 8 | ITP acceptance (~6 slots/yr) | Externally contingent |
| 9 | Three-site lifespan verdict | **The honest ceiling.** Years. A null is a legitimate outcome |
| 10 | Human longevity claim | **NOT bridged by this ladder.** Mouse lifespan is not a validated human surrogate |

**Where weights pay rent:** weights are the object under test from the first
artifact, but they only start paying rent when a frozen learned response
grammar beats fixed pathway, PCA, and additive representations on unseen
drugs and contexts. If the learned representation never beats those
controls, the honest verdict is **this is a statistics project and the ML was
ornamental.** That is a permitted outcome.

## Rung 4 is falsification, not fitting

With ~50 ITP compounds under heterogeneous doses/ages/sexes you cannot FIT a
bridge — you can only FALSIFY candidate bridges. Enumerate parameter-poor
mechanistic bridges and test each against the ITP record with
leave-mechanism-out and temporal censoring. n≈50 is fatal for fitting and
adequate for killing. This requires multiple-testing control (online FDR,
e.g. SAFFRON/ADDIS) or it is ill-posed fitting with extra steps.

**Leakage warning (severe):** any retrospective ITP evaluation using
contemporary LLM or literature embeddings risks catastrophic label leakage —
the weights may already encode published ITP results. Admit NO
literature-derived or LLM-derived features. Chemical structure descriptors
are timeless; measured cellular response is timestamped. Residual risk:
compound *inclusion* in atlases like LINCS is itself outcome-correlated
selection.

## Deferred, not dead

- **Deprioritization engine** (NOT a rejection engine). An automatic veto is
  self-confirming: prospectively rejected compounds are never tested, so the
  system manufactures its own labels and an unobservable false-negative cost
  makes it *less* falsifiable. The legitimate form is auditable abstention —
  "no surviving bridge distinguishes this from historical nulls; do not spend
  a nomination until assay X resolves it."
- **Optimal experimental design over candidate bridges.** It presupposes
  learned response programs to define bridges over, so it runs AFTER Rung 0
  passes. If Rung 0 fails it is still worth doing, but becomes a
  mechanistic-statistics project and must NOT be called fulfilment of the
  weights thesis.

---

# FIRST ARTIFACT (the reviewer's locked L0 protocol)

# Lexicon L0 — Syntax Gate v1.0

**Question:** Can learned weights encode a reusable grammar of non-additive cellular responses, beyond adding the experimentally observed effects of each constituent perturbation?

**Scope:** One assay, one cell line, CRISPRa gene pairs. Passing keeps the weights thesis alive; it establishes no drug, lifespan, or cross-context claim.

## 1. Data

**Canonical source:** Norman et al., 2019, *Exploring genetic interaction manifolds constructed from rich single-cell phenotypes*, GEO **GSE133344**, last GEO update 2020-04-23. [GEO record](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) [paper](https://pubmed.ncbi.nlm.nih.gov/31395745/)

Use only these author-provided filtered files:

- `GSE133344_filtered_matrix.mtx.gz`
- `GSE133344_filtered_genes.tsv.gz`
- `GSE133344_filtered_barcodes.tsv.gz`
- `GSE133344_filtered_cell_identities.csv.gz`

Do not use GEARS/scPerturb processed copies for the canonical run.

Manifest inspection gives:

- 33,694 measured genes.
- 111,445 rows in the filtered identity file.
- **91,168 retained cells** after requiring `good_coverage=True` and `number_of_cells=1`.
- 105 gene-level single perturbations.
- **131 unordered double perturbations.**
- Every double has both corresponding singles.
- Every double occurs in all eight gemgroups.

### Preprocessing

1. Retain only the 91,168 high-confidence singlets above.
2. Collapse guide orientations and negative-control variants:

   - `gene + NegCtrl` → single `gene`.
   - `geneA + geneB` → sorted unordered pair.
   - `NegCtrl + NegCtrl` → control.

3. Per cell: library-size normalize to 10,000 counts, then `log1p`.
4. Select **5,000 HVGs** using only control and single-perturbation cells—never double-perturbation cells—with Scanpy `seurat_v3`, `batch_key=gemgroup`.
5. For condition \(c\) in gemgroup \(b\):

\[
\Delta_{c,b}=\mu_{c,b}-\mu_{\mathrm{control},b}
\]

6. Modeling unit: **condition × gemgroup pseudobulk**, not individual cells. All eight gemgroups belonging to a held-out pair travel together.

This is primarily a sparse CPU/RAM preprocessing job. The compressed expression matrix is about 1.1 GB; 16 GB system RAM may work carefully, while 32 GB is comfortable. After pseudobulking, the training data are tens of megabytes. The 168K-parameter model should use well under 1 GB VRAM; the RX 6800 XT is ample.

## 2. Split

Use **one-shot five-fold cross-fitting** over all 131 unordered pairs.

1. Canonical pair key: `min(geneA,geneB)+"+"+max(...)`.
2. Compute SHA-256 of `lexicon-v1|PAIR_KEY`.
3. Sort by digest and assign round-robin to five folds.
4. Fold sizes: **27, 26, 26, 26, 26**.
5. Each pair receives exactly one prediction from a model that saw:

   - neither its cells;
   - nor its pseudobulks;
   - nor its outcome;
   - but did see both constituent single perturbations.

All controls and all 105 singles are available in every fold. No pair may cross folds through gemgroups or guide orientation.

Within each outer-training fold, reserve 10% of training pairs by a second frozen hash for checkpoint selection only.

### Interaction strata

For pair \(ij\), gemgroup \(b\):

\[
r_{ij,b}=\Delta_{ij,b}-\Delta_{i,b}-\Delta_{j,b}
\]

Let \(\bar r_{ij}\) be its mean across eight gemgroups. Within outer fold \(f\), calculate the training-only shared interaction centroid:

\[
c_f=\operatorname{mean}_{ij\in train(f)}\bar r_{ij}
\]

and interaction magnitude:

\[
m_{ij}=\frac{\|\bar r_{ij}-c_f\|_2}{\sqrt{5000}}
\]

Freeze:

\[
T_f=Q_{0.80}\{m_{ij}:ij\in train(f)\}
\]

After all predictions are sealed, a held-out pair is:

- **High-interaction stratum:** \(m_{ij}>T_f\)
- **Additive stratum:** \(m_{ij}\le T_f\)

The test outcome is used only to assign its predeclared evaluation stratum, never for fitting, threshold selection, or model choice.

## 3. Model class

**32 learned response programs; exact additive spine plus learned interaction residual.**

Let:

- \(G=5000\) genes.
- \(K=32\) programs.
- \(B\in\mathbb R^{G\times K}\): learned program basis.

Single-perturbation activation:

\[
a_{i,b}=B^\top\Delta_{i,b}
\]

Symmetric pair features:

\[
h_{ij,b}=
[a_i+a_j,\ |a_i-a_j|,\ a_i\odot a_j]
\in\mathbb R^{96}
\]

Composition network:

\[
u=W_2\,\mathrm{GELU}(W_1h+b_1)+b_2
\]

with \(96\rightarrow64\rightarrow32\), followed by learned soft-thresholding:

\[
z_{ij,b}=\operatorname{sign}(u)\max(|u|-\mathrm{softplus}(\tau),0)
\]

Prediction:

\[
\widehat{\Delta}_{ij,b}
=
\Delta_{i,b}+\Delta_{j,b}+Bz_{ij,b}
\]

Thus the model cannot relearn the additive effect: it learns only reusable deviations from addition.

### Parameter count

- Program basis: \(5000\times32=160,000\)
- MLP and biases: 8,288
- Sparsity thresholds: 32
- **Total: 168,320 parameters**

No gene-ID lookup table, pathway annotation, graph database, or text embedding is allowed. A gene is represented only by its experimentally measured single-perturbation response.

### Frozen training recipe

- Loss:

\[
L=
\operatorname{MSE}(r,Bz)
+0.1\,\operatorname{MSE}(\Delta,BB^\top\Delta)
+10^{-3}\|z\|_1
+10^{-2}\|B^\top B-I\|_F^2
\]

- AdamW, learning rate \(10^{-3}\), weight decay \(10^{-4}\).
- Maximum 3,000 epochs.
- Check inner-validation loss every ten epochs.
- Patience: 200 checks.
- Five seeds: `17, 29, 43, 71, 101`.
- Average their held-out predictions.
- FP32 only for the canonical verdict.

Program labels and pathway interpretations occur only after the gate; they play no role in scoring.

## 4. Baseline ladder

Final baselines:

1. **No change:** \(\widehat{\Delta}_{ij}=0\)
2. **Perturbed mean:** mean double-perturbation response in the outer-training fold
3. **Additive singles:**  
   \[
   \widehat{\Delta}_{ij}=\Delta_i+\Delta_j
   \]
4. **Gene-marginal interaction:** mean training residual associated with gene \(i\) and gene \(j\), averaged; global training residual for an unseen marginal
5. **PCA-32 + ridge:** PCA learned on permitted training data; symmetric single-response features mapped linearly to interaction residuals. Ridge penalty selected from the frozen grid `{0.01, 0.1, 1, 10, 100}` using inner validation.

**The thesis-null baseline to beat is additive singles.** It represents “there is no reusable interaction law.”

The gene-marginal baseline is mandatory because otherwise a model could win by learning that certain genes are generally interaction-prone. PCA+ridge asks whether nonlinear sparse programs add anything beyond ordinary low-rank regression.

## 5. Metrics and gate

For held-out pair \(p\) in fold \(f\):

\[
q_p=\bar r_p-c_f,\qquad
\widehat q_p=\widehat{\bar r}_p-c_f
\]

Pair error:

\[
e_p=\frac{1}{5000}\|\widehat q_p-q_p\|_2^2
\]

### Primary: high-interaction error reduction

Against additive singles:

\[
I_{\mathrm{high}}
=
1-\operatorname{median}_{p\in high}
\left(\frac{e_{p,\mathrm{Lexicon}}}{e_{p,\mathrm{additive}}}\right)
\]

### Additive-stratum non-inferiority

\[
D_{\mathrm{add}}
=
\operatorname{median}_{p\in additive}
\left(\frac{e_{p,\mathrm{Lexicon}}}
{e_{p,\mathrm{additive}}}\right)-1
\]

### Perturbation discrimination

Within each outer fold, calculate L1 distance from each predicted \(\widehat q_p\) to every true held-out \(q_t\). Rank the correct pair using midranks for ties:

\[
PDS_p=1-\frac{\operatorname{rank}(p)-1}{N_f-1}
\]

Random/constant prediction has expectation approximately 0.5. Aggregate PDS over all 131 out-of-fold predictions.

### Uncertainty

Use **10,000 paired cluster-bootstrap resamples**, stratified by outer fold and interaction stratum. Resample whole gene pairs; their eight gemgroups remain together. RNG seed: `20260807`. Report percentile 95% intervals.

### PASS — all conditions required

1. At least **20** held-out pairs enter the high-interaction stratum.
2. \(I_{\mathrm{high}}\ge0.10\), and its 95% lower bound is above zero.
3. The 95% upper bound of \(D_{\mathrm{add}}\) is no greater than **0.05**.
4. Lexicon’s mean PDS exceeds additive PDS by at least **0.05**, with a 95% lower bound above zero.
5. No non-neural baseline has a statistically resolved error advantage over Lexicon on the high-interaction stratum.

### FAIL

FAIL if at least 20 high-interaction pairs exist and any of these is true:

- The 95% upper bound of \(I_{\mathrm{high}}\) is below **0.10**: the useful improvement is ruled out.
- The 95% lower bound of \(D_{\mathrm{add}}\) exceeds **0.05**: it materially damages additive cases.
- The 95% upper bound of PDS improvement is below **0.05**: useful discrimination is ruled out.

If PCA+ridge passes while Lexicon fails, record:

> **LEXICON MODEL FAIL; WEIGHTS THESIS ALIVE.**

That result redirects architecture rather than killing learned compression.

### INDETERMINATE

INDETERMINATE if:

- fewer than 20 held-out pairs enter the high-interaction stratum;
- any confidence interval straddles its pass and fail boundaries;
- strata are unstable under gemgroup leave-one-out analysis;
- preprocessing or manifest checks fail;
- different model seeds reverse the sign of the primary improvement.

No additional run may convert INDETERMINATE into PASS under this spec. A revised experiment requires `v1.1`, a new commit, and a new untouched dataset.

## 6. Predeclaration contents

Commit `SPEC.md` before the evaluation command can expose any held-out result. Freeze:

- scientific claim and prohibited claims;
- exact GEO filenames and SHA-256 hashes;
- expected manifest: 91,168 retained cells, 105 singles, 131 pairs, eight gemgroups, no missing constituent singles;
- condition parser and exclusion rules;
- normalization, batch centering, HVG procedure, and resulting gene-list hash;
- pair keys, fold algorithm, and complete fold manifest;
- inner-validation assignment;
- model equations, dimensions, loss coefficients, optimizer, seeds, stopping rule, and ensemble rule;
- all baseline implementations and ridge grid;
- stratum definition and 80th-percentile rule;
- metrics, bootstrap procedure, confidence intervals, and PASS/FAIL/INDETERMINATE logic;
- package versions and hardware;
- one-shot execution command;
- output schema and result-file hashes;
- synthetic-data unit tests;
- rule that implementation bugs invalidate the run rather than authorize selective reruns.

Operationally: preprocessing may create and hash the manifest, but no script may print pair outcomes by fold or model performance until the spec and code commit are frozen. The canonical evaluator should run all five folds before revealing any result.

> **Note added after the run (not part of the predeclaration):** the GEO SHA-256 hashes, the resulting gene-list hash, package versions/hardware and result-file hashes listed above were not frozen in practice and are not recorded in this repository. The fold algorithm is fixed in `lexicon/split.py`, but no complete fold manifest file was saved. The manifest check that was enforced covers the expected counts and pair completeness only. See the Limitations section of `README.md`.

## 7. Failure mode I most expect

**The eight gemgroups will masquerade as eight independent replicates, while the effective interaction sample remains only 131 pairs.**

Worse, each interaction residual subtracts three estimated pseudobulks:

\[
\Delta_{ij}-\Delta_i-\Delta_j
\]

so measurement error compounds. I expect the top-interaction membership to move when one gemgroup is removed. The model may appear convincing under cell-level or gemgroup-level statistics, then become INDETERMINATE when pairs are correctly treated as the independent units.

The darker biological version is that some “grammar” will describe CRISPRa delivery competition, fitness selection, or changing cell-state proportions—not reusable intracellular interaction law. The present gate can establish an assay grammar only. Cross-guide and cross-assay transfer must adjudicate that later.

## Name

Keep **Lexicon**. It is exact without overclaiming: a learned vocabulary may exist even if no universal biological language does.

Call this first artifact:

> **Lexicon L0: Syntax — Can observed single perturbations compose into unseen pair responses?**



---

# SPEC v1.1 — AMENDMENTS (frozen 2026-08-06, BEFORE any held-out biological result)

Every change below was triggered by a PLANTED CONTROL failing, never by a
biological result. No held-out pair outcome had been computed when these were
frozen. Each was reviewed and signed off by the reviewer; the ones the reviewer
initiated are marked.

**A1 — variance-standardised objectives.** v1.0's raw-MSE loss made the balance
depend on the arbitrary magnitude of a log1p residual: measured at init, the
orthogonality penalty was 17x the fit term and reconstruction 4.7x, so the
optimiser shaped B for orthogonality and never fit the interaction. Now
`MSE(r,Bz)/s_r^2 + 0.1 MSE(d,BB^T d)/s_d^2 + 1e-3|z|_1 + 1e-2||B^T B - I||_F^2`
with fold-level constants computed ONCE from outer-fit data, detached,
`var(correction=0)`, raising on degenerate variance rather than an epsilon
rescue. Inner-validation MSE normalised by the same training `s_r^2`.

**A2 — standardised model input/output.** `a = B^T(delta/s_d)`, `rhat = (zB^T)*s_r`,
with `s_d`/`s_r` as non-parameter buffers. A1 alone left the soft threshold, the
GELU regime and the L1 penalty scale-dependent; the 0.1x planted control caught
it. Measured after: ensemble unexplained variance range across 0.1x/1x/10x is
**0.000373 (CPU) / 0.000118 (ROCm)**. Parameter count unchanged at 168,320.

**A3 — `TAU_INIT` -2.0 -> -4.0.** A program thresholded to zero receives exactly
zero gradient and can never recover, so v1.0 started programs inside a dead
zone. Sweep (5 seeds, worst-seed unexplained): -2.0 -> 0.384, -3.0 -> 0.292,
-4.0 -> 0.275, -5.0 -> 0.278. -4.0 is the centre of the -3..-5 plateau, not the
best-looking endpoint. Frozen; no further tau/optimiser/LR/seed-count sweeps.

**A4 — relative early stopping.** Absolute `best - 1e-9` is scale-dependent;
now `1e-4` *relative* improvement on normalised validation MSE.

**A5 — condition-level training unit** (reviewer). Train on one gemgroup-mean
vector per condition; the eight gemgroups are used ONLY for stability and
uncertainty. Effective n stays 131 pairs. Avoids technical pseudoreplication.

**A6 — one bootstrap for every gate-bearing comparison** (reviewer). Per-baseline
ratio CIs now use the same frozen 10,000-draw fold/stratum-stratified cluster
bootstrap as the primary metrics. The 2,000-draw unstratified path is deleted —
a looser bootstrap for baselines would make the baseline test easier than the
primary test.

**A7 — standardised PCA-ridge** (reviewer). Training-only `StandardScaler` before
the frozen alpha grid, which is otherwise scale-sensitive and makes the baseline
artificially weak.

**A8 — corrected redirect logic** (reviewer). v1.0 emitted
`LEXICON MODEL FAIL; WEIGHTS THESIS ALIVE` when Lexicon BEAT PCA-ridge, which is
backwards. The redirect is now decided by whether PCA-ridge ITSELF clears the
additive-relative gates. Both directions are pinned by tests.

**A9 — high-set stability, not overall agreement** (reviewer). ~80% of pairs are
additive by construction, so overall membership agreement can hide total churn
in exactly the stratum the gate depends on. Now, for every omitted gemgroup:
`|H n H_-b|/|H| >= 0.90` AND `Jaccard(H,H_-b) >= 0.80`, else INDETERMINATE.

**A10 — planted-control suite replaces the per-seed bar** (reviewer's ruling). The
`every seed <= 0.25` rule tested an optimiser property of a model we never ship
(the canonical predictor is the five-seed ensemble), but a median-only bar could
hide dependence on one lucky seed. Eight frozen conditions, in
`lexicon/controls.py`:

1. Oracle unexplained `< 1e-4`
2. Shuffled-pair negative control unexplained `>= 0.90`
3. Five-seed ensemble unexplained `<= 0.10`
4. Median individual seed `<= 0.25`
5. Every individual seed `<= 0.50`
6. Every leave-one-seed-out four-model ensemble `<= 0.10`
7. Conditions 3 and 6 hold at scales 0.1, 1, 10
8. Full-ensemble unexplained variance range across scales `<= 0.02`

**A11 — controls must traverse the production backend** (reviewer). A positive
control that only ever runs on CPU does not validate the one-shot ROCm pipeline.
Both bundles are committed in `controls/`.

## Control results (final v1.1 code, both backends, all 8 conditions PASS)

| | CPU | ROCm |
|---|---|---|
| oracle | 6.233e-16 | 6.233e-16 |
| shuffled negative | 0.9981 | 0.9981 |
| ensemble @ 0.1x / 1x / 10x | 0.0733 / 0.0737 / 0.0737 | 0.0734 / 0.0735 / 0.0735 |
| median seed | 0.1565 | 0.1565 |
| max seed | 0.2746 | 0.2746 |
| leave-one-seed-out max | 0.0871 | 0.0867 |
| ensemble range across scales | 0.000373 | 0.000118 |

## Implementation note

`seurat_v3` HVG is defined on raw counts, so it is computed on counts for
control+single cells only, and the resulting 5,000-gene set is then used for the
lognorm pseudobulk. Doubles never inform gene selection.

## Manifest reconciliation

The reviewer's frozen expectation was derived independently of the parser that read the
files. They reconcile exactly: 91,168 retained cells, 105 singles, 131 pairs, 8
gemgroups, 0 unparsed, 0 pairs missing a constituent single, 9,702 control cells,
pseudobulk (1888, 5000) = 236 conditions x 8 gemgroups.
