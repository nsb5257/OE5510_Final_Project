# Two-Week Implementation Plan: Ship Maneuvering PINN Paper

**Project:** Reproduce the core implementation from *Identification modeling of ship maneuvering motion based on physics-informed neural network*.

**Week 1:** 16–22 October 2026 (project timeline used for the review milestone)  
**Week 2:** 23–29 October 2026  
**Two-week milestone / presentation:** Thursday, 29 October 2026, 4:30 PM  
**Final project review:** Thursday, 5 November 2026 — this is the following week and is outside the implementation scope of this plan.  
**End of this plan:** Thursday, 29 October 2026. **Do not add Week 3 implementation tasks to this plan.**

> **Time-critical note:** The current date is 27 October, so only about two calendar days remain before the Week 2 review. First inspect the workspace and identify work that is already complete. Attempt the remaining tasks in order, but never mark a task complete unless its acceptance criteria pass. If the project is starting from zero, completing every item by 29 October may not be realistic; make the actual status and blockers visible rather than claiming success. For the 29 October presentation, prioritize a verified dataset audit, an accurately explained method, tested equations/preprocessing where possible, and at least one genuine baseline or PINN diagnostic result. The list remains the intended two-week scope, not permission to fake or skip validation.

## 1. Goal and scope

Build the simplest defensible Python implementation that reproduces the paper's **core simulation experiments**. The deliverable is research code that a student can explain in a presentation, not production software.

### Week 1 target: functioning prototype (16–22 October; audit what already exists)

- Dataset has been inspected and the preprocessing is documented.
- Ship-motion equations used by the paper are implemented and tested.
- A data-only baseline neural network has been trained and evaluated.
- A PINN smoke-training run has been attempted, with data loss and each physics-loss component tracked separately.

### Week 2 target: core reproduction (complete by 29 October where feasible)

- PINN trains reproducibly using the paper's main loss design.
- The main simulation maneuvers available in the local dataset have been evaluated.
- Predicted-versus-reference plots and RMSE metrics are saved.
- PINN and data-only baseline are compared fairly.
- Hydrodynamic derivative estimates are reported and checked against the paper where comparable.
- At least one controlled ablation is completed, preferably dynamic loss weighting versus constant loss weighting, if the paper specification confirms that comparison and it is feasible.

### Explicitly out of scope for this two-week plan

- Full experimental-data reproduction unless it is already straightforward after the simulation milestone.
- Full sensitivity/generalization study beyond one small, clearly documented check if time permits.
- Production architecture, deployment, APIs, dashboards, configuration frameworks, elaborate logging, packaging, or performance optimization.
- Fetching data from the internet, downloading someone else's implementation, or copying a repository.
- Inventing missing data, results, coefficients, paper details, or unsupported explanations.

## 2. Source-of-truth rules

1. **The supplied paper is the source of truth for the method.** The local dataset is the source of truth for available data. Do not silently replace either with general knowledge.
2. Before implementing the model, create `notes/paper_spec.md`. Record the paper's exact input variables, output variables, equation numbers, physical residuals, data-loss definition, loss-weighting method, network layout, parameter initialization, optimizer, learning-rate schedule, iteration count, data maneuvers, preprocessing, and evaluation metrics. Every item must include the paper's printed page and equation/figure/table number where applicable.
3. The PDF may be image-based. If text extraction fails, inspect the rendered PDF pages. Do not guess at unreadable equations. Mark them `UNCLEAR — USER REVIEW REQUIRED` and stop the dependent task until clarified.
4. If the paper and local data appear inconsistent, write down the mismatch, show evidence, and ask the user. Do not silently change a sign convention, unit, column meaning, equation, or experiment.
5. Do not fetch data or code. Inspect only files inside the current workspace. Do not assume data column names or file formats before opening the files.
6. Never overwrite raw data. Save generated data, checkpoints, metrics, and figures under `outputs/`.
7. Clearly label every result as one of: **paper-reported**, **produced by our implementation**, or **planned/not yet run**. Never present a paper plot or table as our result.

## 3. Keep the code deliberately simple

- Use Python and PyTorch, matching the paper's reported implementation framework where feasible. Inspect the existing environment first; do not install or upgrade packages without the user's approval.
- Prefer a few short scripts and ordinary functions. Avoid unnecessary classes, decorators, abstractions, registries, complex configuration systems, CLI frameworks, experiment trackers, and design patterns.
- A small `torch.nn.Module` for the neural network is acceptable because PyTorch requires a model definition. Do not build a general-purpose PINN framework.
- Do not add features that are not needed to reproduce the paper's two-week scope.
- Use readable variable names and comments that explain the physics or a non-obvious choice. Code should be easy to explain line by line.
- Start with small smoke tests. Do not launch a long training run until shapes, units, loss values, gradients, and a short run have been checked.
- Keep baseline and PINN comparisons fair: same data split, target variables, units, evaluation procedure, and comparable training conditions unless the paper explicitly requires otherwise.
- Fix seeds where practical and record the seed. Report if results are still nondeterministic.

## 4. Suggested workspace layout

Preserve the existing repository if one exists. Do not move or rename user files unless explicitly approved. If starting from an empty folder, use this small structure:

```text
ship-pinn-reproduction/
├── plan.md                         # this file
├── references/
│   └── paper.pdf                   # supplied research paper
├── data/
│   └── raw/                        # copy the provided dataset here, unchanged
├── notes/
│   ├── workspace_inventory.md      # generated in Task T0
│   └── paper_spec.md               # generated in Task T0
├── src/
│   ├── inspect_data.py
│   ├── ship_dynamics.py
│   ├── prepare_data.py
│   ├── train_baseline.py
│   ├── model.py
│   ├── train_pinn.py
│   └── evaluate.py
├── tests/
│   └── test_ship_dynamics.py
└── outputs/
    ├── plots/
    ├── metrics/
    ├── processed/
    └── checkpoints/
```

Create only the files actually needed. If the project already has a simpler suitable structure, keep it.

## 5. Sequential task protocol — mandatory

**Execute exactly one task per agent run.** At the start of each run:

1. Read this plan and find the first task marked `[ ]` that is not blocked.
2. Read the relevant paper specification and existing code/data before editing.
3. Complete only that task and its acceptance checks.
4. Update its checkbox to `[x]` only when its acceptance criteria genuinely pass. Record any limitations or failed checks beside the task.
5. Stop. Do not begin the next task, perform opportunistic refactors, or silently expand scope.
6. Report: files changed, commands run, tests/results, unresolved issues, and the ID of the next task. Ask for user approval to continue.

If one task reveals a material ambiguity, do not guess. Record the exact question and evidence, mark the task `BLOCKED`, and stop. The user can answer before work continues.

## 6. Tasks and acceptance criteria

### [x] T0 — Workspace inventory and paper specification
**Completed:** 27 October 2026.
- Workspace and scanned PDF inspected; environment audited (PyTorch 2.13.0+cpu, Python 3.13.3).
- Created `notes/workspace_inventory.md` and `notes/paper_spec.md`.
- Derived and verified exact Prime-I ship dynamics equations matching training dataset accelerations within $< 10^{-5}$.
- All 36 trainable + 5 fixed hydrodynamic derivatives transcribed from Table 2 and Page 7.
- Loss function, dynamic weighting, standardization, and test RMSE specs documented with page references.
**Target:** 27 October. **Estimate:** 1–2 focused hours. **No model implementation in this task.**

Actions:
- Inspect existing files, Python environment and available packages without changing the environment.
- Find the local research PDF and dataset files. Do not assume their names or format.
- Create `notes/workspace_inventory.md` listing discovered files, formats, sizes, apparent dataset roles, environment details and unknowns.
- Create `notes/paper_spec.md` by inspecting the paper. Include the exact model inputs/outputs; ship equation transcription; definition of hydrodynamic derivatives; data-loss and all physics-loss terms; dynamic/constant weighting details; architecture; initialization; optimizer and learning-rate schedule; iteration counts; maneuvers; train/test setup; and reported metrics. Include page/equation/figure/table references.
- Where the paper does not specify a setting clearly, label it as unspecified rather than filling it in.

Acceptance criteria:
- Data files are identified without altering them.
- The method can be described from the paper-spec note without relying on guessed details.
- All unclear implementation-critical items are listed as blockers.

### [x] T1 — Dataset audit and exploratory plots
**Completed:** 27 October 2026.
- Created `src/inspect_data.py`.
- Profiled all 6 maneuvers (3 train, 3 test) in `outputs/data_profile.md`: 0 missing values, 0 duplicates, 1.0 s sampling rate.
- Mapped all columns to paper state variables and verified physical vs dimensionless units.
- Generated and saved 10 exploratory time-history and trajectory plots under `outputs/plots/`.
**Target:** 27 October. **Estimate:** 2–4 hours.

Actions:
- Inspect the actual column names, row counts, missing values, duplicate timestamps, units (if documented), sampling intervals, trajectories/maneuvers and ranges.
- Map the dataset columns to the paper's variables only when justified by evidence. Mark uncertain mappings.
- Produce a concise data profile and simple plots of available motion variables over time and/or trajectories.
- Save plots and summaries in `outputs/`; never alter raw files.

Suggested outputs:
- `src/inspect_data.py`
- `outputs/data_profile.md` (or CSV if more appropriate)
- `outputs/plots/` with clearly labelled plots

Acceptance criteria:
- The script runs against the real local dataset.
- The user can see which variables and maneuver records are available.
- No fabricated rows, assumed columns, or hidden data cleaning.

### [x] T2 — Implement and test the ship-motion equations
**Completed:** 27 October 2026.
- Created `src/ship_dynamics.py` implementing non-dimensional hydrodynamic forces (Eq. 4), accelerations (Eq. 5), and forward Euler integration (Eq. 6).
- Preserved Prime-I coordinate/sign conventions and verified exact numerical reproduction of training dataset accelerations.
- Created `tests/test_ship_dynamics.py` with 8 unit and regression tests (equilibrium, analytical hand-calculation, array vectorization, Euler stepping, trajectory simulation, and dataset accuracy checks across all 3 training maneuvers).
- All 8 unit tests passed via pytest.
**Target:** 27–28 October. **Estimate:** 3–5 hours.

Actions:
- Translate only the equations recorded in `notes/paper_spec.md` into small, readable Python functions.
- Preserve the paper's coordinate/sign convention and physical units.
- Add minimal tests for shapes, finite outputs, dimensional/limit checks that can be justified, and a hand-worked numerical example where possible.
- Keep the physical equations separate from neural-network training so they can be checked independently.

Suggested outputs:
- `src/ship_dynamics.py`
- `tests/test_ship_dynamics.py`

Acceptance criteria:
- Each implemented equation points back to a paper equation/page in comments or the paper-spec note.
- Tests pass.
- Any equation that could not be transcribed reliably remains unimplemented and is listed as a blocker.

### [x] T3 — Minimal preprocessing pipeline
**Completed:** 27 October 2026.
- Created `src/prepare_data.py` implementing trajectory-level loading, standardization (Eq. 19), and destandardization without altering raw data.
- Prevented data leakage by deriving normalization statistics ($\mu, \sigma$) strictly from the 2,997 training data points and saving them to `outputs/processed/normalization_stats.json`.
- Exported structured datasets `outputs/processed/train_data.npz` and `outputs/processed/test_data.npz`.
- Documented variable ordering, units, and loss scaling guidelines in `outputs/processed/preprocessing_metadata.md`.
- Created `tests/test_prepare_data.py` verifying file existence, shapes, standardization invertibility, and test dataset integrity. All 5 tests pass (13 tests pass overall).
**Target:** 27–28 October. **Estimate:** 2–4 hours.

Actions:
- Implement the smallest necessary processing steps supported by the dataset and paper.
- Document filtering, resampling, standardization and missing-data handling. Do not apply smoothing unless the chosen data/method requires it and the decision is documented.
- Prevent data leakage: calculate normalization statistics on the training data only and reuse them for validation/test data.
- Keep raw and normalized units distinct. Ensure physics residuals are calculated in physical units or correctly transformed if normalization is used.
- Save a simple processed dataset/metadata file under `outputs/processed/` without changing raw data.

Acceptance criteria:
- Prepared data can be loaded consistently by later scripts.
- Variable order and units are documented.
- The training/evaluation split is explicit and appropriate to the available trajectories. Avoid random row splitting of time-series data when it would leak near-identical time points across splits; follow the paper if its split is specified.

### [x] T4 — Train and evaluate a data-only baseline
**Completed:** 27 October 2026.
- Created `src/model.py` defining `ShipMotionNN` (4 inputs, 3 hidden layers of 10 neurons, Tanh, 3 outputs, Xavier uniform initialization).
- Created `src/evaluate.py` implementing recursive multi-step forward Euler time integration and RMSE metric computation.
- Created `src/train_baseline.py` training on all 2,997 training samples for 40,000 iterations using Adam with paper's learning-rate schedule ($10^{-3} \to 10^{-4}$).
- Training completed in 329.8 s reaching final normalized MSE $3.28 \times 10^{-5}$.
- Evaluated on all 3 held-out test maneuvers (`turning_circle_30`, `zigzag_20_10`, `zigzag_25_5`); saved model checkpoint, training loss curve, prediction-vs-reference plots, and RMSE metric tables under `outputs/`.
- Created `tests/test_train_baseline.py`. All 17 tests pass across the repository.
**Target:** 28 October. **Estimate:** 2–4 hours.

Actions:
- Implement the simplest standard feed-forward neural network baseline consistent with the paper's comparison. Verify the exact baseline settings from `paper_spec.md` rather than assuming them.
- Train on the prepared local simulation dataset.
- Save prediction-vs-reference plots, training history and RMSE metrics for the target variables.
- Report the data split and settings used.

Suggested output:
- `src/train_baseline.py`
- Reuse `src/evaluate.py` if it reduces duplicated code without overengineering.

Acceptance criteria:
- The baseline trains and produces finite predictions.
- Metrics are calculated in the correct units after reversing any normalization.
- Results can be reproduced by rerunning a documented command.

### [ ] T5 — Implement the minimal PINN and paper loss
**Target:** 28–29 October. **Estimate:** 3–5 hours.

Actions:
- Implement the neural-network layout and initialization specified by the paper. The paper describes a fully connected feed-forward network with three hidden layers of 10 neurons and Tanh activation; verify against the PDF before coding.
- Implement the paper's data loss and all three-degree-of-freedom physics residuals, using the precise definitions established in T0.
- Implement the paper's dynamic loss weighting only as described in the paper. Keep constant-weight loss available for the planned ablation only if that matches the reported comparison.
- Make the hydrodynamic derivatives trainable only where the paper treats them as trainable parameters. Apply the paper's stated initialization when confirmed.
- Log data loss, individual physics-loss terms, total loss and trainable derivative estimates separately.

Suggested outputs:
- `src/model.py`
- `src/train_pinn.py`

Acceptance criteria:
- Loss components are separately visible and finite on a tiny batch.
- A backward pass produces valid gradients.
- Inputs, outputs and physical residuals match `paper_spec.md`.

### [ ] T6 — PINN smoke training and debugging
**Target:** 28–29 October. **Estimate:** 2–4 hours.

Actions:
- Run a short training test first, not the full paper-length run.
- Check for NaN/Inf, exploding gradients, shape mismatches, sign/unit errors and loss components with wildly different scales.
- Change one thing at a time. Record the initial symptom, hypothesis, single change, and observed result in `notes/experiment_log.md`.
- Do not silently tune until a plot looks like the paper. Preserve failed runs' concise diagnostic logs.

Acceptance criteria:
- The training loop completes a short run and logs each loss component.
- At least one checkpoint can be saved and reloaded, if checkpointing is needed for the main run.
- Known issues are documented, not hidden.

### [ ] T7 — Main PINN simulation run
**Target:** 29 October, if the smoke test is stable. **Estimate:** 3–6 hours, depending on the machine.

Actions:
- Run the verified simulation experiment using the settings transcribed from the paper, adjusted only where the local environment requires a documented change.
- Record seed, device, iteration count, learning-rate schedule, data split and any deviations from the paper.
- Save the best/last checkpoint as appropriate, training history and all loss curves.
- If the exact paper run is too slow or unstable, complete a smaller stable run and clearly label it as a partial reproduction.

Acceptance criteria:
- The run is reproducible enough to rerun under documented settings.
- Losses and coefficient values are saved, not just printed to the console.
- Deviations from the paper are explicitly recorded.

### [ ] T8 — Evaluate maneuvers, plot predictions and compute RMSE
**Target:** 29 October, after a successful PINN run. **Estimate:** 2–4 hours.

Actions:
- Evaluate each core simulation maneuver actually present in the local dataset and specified by the paper.
- Create labelled reference-versus-PINN plots for the relevant state variables and trajectories.
- Compute RMSE in physical units with consistent sample alignment. Compare the same records/targets for the PINN and baseline.
- Include a small metrics table rather than relying on visual similarity alone.

Acceptance criteria:
- The plots come from our saved predictions and local dataset.
- Every metric states its variable, unit, split and model.
- The baseline/PINN comparison uses the same evaluation protocol.

### [ ] T9 — Hydrodynamic derivative analysis
**Target:** 29 October, if derivative estimates are available. **Estimate:** 2–4 hours.

Actions:
- Export the estimated hydrodynamic derivatives.
- Compare with paper-reported or reference values only when the parameter definitions and units are comparable.
- Explain large discrepancies honestly; check unit conversions, coefficient normalization, identifiability and training convergence before drawing conclusions.

Acceptance criteria:
- Estimates and comparison values are traceable to a run and a source.
- Incomparable or unavailable reference values are marked as such; do not invent replacements.

### [ ] T10 — One controlled ablation
**Target:** 29 October, only if the PINN and baseline both run. **Estimate:** 2–4 hours.

Actions:
- Prefer comparing the paper's dynamic loss weighting with constant weighting, provided T0 verifies the setup.
- Change only the weighting method; keep dataset, split, architecture, seed and other training settings the same as far as practical.
- Compare component/total losses, prediction plots and RMSE. Record the limitations of a single-seed experiment.

Acceptance criteria:
- The comparison is controlled and the changed setting is explicit.
- The result supports a limited conclusion; do not overclaim statistical significance.
- If this ablation cannot be completed, document why and report it as incomplete instead of substituting a different experiment without approval.

### [ ] T11 — Two-week checkpoint and reproducibility notes
**Target:** 29 October. **Estimate:** 1–2 hours.

Actions:
- Ensure all core plots and metrics are saved in `outputs/`.
- Write a short `README.md` with environment, exact commands to run, dataset paths, produced artifacts, known limitations and deviations from the paper.
- Summarize which Week 1 and Week 2 acceptance criteria passed or failed.
- Prepare concise result notes for the 29 October presentation and a clear status handoff.

Acceptance criteria:
- Another student can run the core pipeline by following a few documented commands.
- The code remains small and explainable.
- This task marks the end of this plan. Do not add Week 3 work.

## 7. Presentation checkpoint and scope boundary

### By Thursday, 29 October, 4:30 PM — end of Week 2

Use verified, real evidence: dataset profile/plots, the paper's method explained clearly, equation tests and preprocessing checks if completed, and a baseline or preliminary PINN result if available. If the PINN is not stable yet, show what was verified, the actual failed diagnostic and the next technical step. Clearly separate results from our code from paper-reported figures. Do not sacrifice correctness to force a premature full training run.

The final project review is on Thursday, 5 November, but it belongs to the following project week. **This plan does not schedule or define Week 3 implementation work.** The status recorded at the end of this plan should accurately state what is complete, partial, blocked, or not attempted by the 29 October milestone.

## 8. Risk register and response rules

| Risk | Response |
|---|---|
| Dataset column meanings or units are unclear | Inspect file metadata and values; ask the user instead of guessing. |
| Paper equation is unreadable or ambiguous | Mark the exact page/equation as unclear and block dependent code. |
| Physics loss is much larger/smaller than data loss | Log each term; check units and normalization first; alter weighting only according to the paper or as an explicitly labelled experiment. |
| Loss is NaN/Inf or predictions explode | Stop the run; inspect inputs, gradients, initialization, units and residual implementation before increasing iterations. |
| Derivatives fail to match paper | Check definitions, units, sign conventions, identifiability and training; never hand-edit results to match. |
| Model performs well on random rows but poorly on a new maneuver | Check for time-series leakage and evaluate a held-out trajectory/maneuver when available. |
| Full paper run cannot finish in time | Produce a stable, shorter run with clear labels and a documented deviation. Never claim full reproduction. |
| AI-generated code looks plausible but is unverified | Require an independent equation-to-code check, small numerical tests and observed execution output before marking a task complete. |

## 9. Definition of done by the Week 2 milestone — Thursday, 29 October

- [ ] The actual local dataset has been profiled and the preprocessing is reproducible.
- [ ] The paper's relevant ship equations are implemented and tested.
- [ ] A data-only baseline has predictions and RMSE metrics.
- [ ] A PINN implementation follows the verified paper specification and completes a reproducible training run or has a clearly documented blocker.
- [ ] The core available simulation maneuvers have prediction plots and comparable RMSE metrics.
- [ ] PINN-vs-baseline results are available.
- [ ] Hydrodynamic derivative estimates are saved and responsibly compared.
- [ ] One controlled ablation is completed, or its blocker is documented.
- [ ] Known limitations, failed experiments and deviations from the paper are recorded.
- [ ] The run instructions and outputs are documented.

**The quality goal is a small, scientifically honest reproduction that can be explained—not a large codebase or an artificially perfect match to the paper.**
