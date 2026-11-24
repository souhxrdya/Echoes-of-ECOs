# Validation

What was checked, how, and with what result. The method is described in [METHOD.md](METHOD.md) and the
results in [RESULTS.md](RESULTS.md). Every rule below was fixed in the code before the runs it judges;
no threshold, window or order was changed to obtain a pass, and failed checks are reported, not removed.

## 1. Summary

| Check | Scope | Result |
|---|---|---|
| Convergence at three resolutions, declared order confirmed | all 16 entries, ECO and BH reference | confirmed (p = 3.86–4.02; GS 2.01–2.03) |
| Error budget of the observable ≤ 0.1 of its norm (late, full) | all 16 | met by all 16; largest accepted 1.5e-4 |
| Controls (time step, domain, model settings) | all sequences | full coverage; non-negligible ones counted in the budget |
| Hard checks of every record | all records | production records pass; the failed checks, all reported, are in diagnostic jobs (section 2) |
| Independent frequency-domain reconstruction | GS, WH (8 entries) | pass; disagreement 0.40–0.85 of the combined budget |
| Independent causal solver | QH (4 entries) | pass at 10^8, 10^10, 10^12 (0.053%, 0.0083%, 0.0082% against 1%); fail at 10^6 (5.2%) |
| Independent method for the waveform | BS | none: stated gap |
| Reflection coefficients: variants and an independent solver | GS, WH | all 16 arrays agree to better than 1e-6 |
| Modes: spread under changed numerical settings | GS, WH | verified (σ_num 2.6e-13 to 1.2e-12) |
| BS modes | BS | withdrawn (section 5) |
| Status audit: every state re-derived from its evidence | catalogue and every result file | pass (`python -m eco_library audit`) |
| Full reproduction from the frozen equations | whole library | identical to the committed results (section 8) |

## 2. Convergence, controls and gates

Each sequence is computed at Δx = 0.2, 0.1 and 0.05 M. The observed order in the late window is 3.86–4.02
for the order-4 sequences and 2.01–2.03 for the gravastars (order 2 from the shell junction), within
the declared band of ±0.5. Every sequence has controls that change one setting at a time (time step
halved; left and right boundaries moved; for BS the background tolerance, profile sampling and domain;
for QH the Fourier settings), and the control coverage of every entry is complete.

The checks that fail are all in diagnostic jobs and are kept in the block reports: the replay of the
frozen notebook's BH record flags a causal-boundary warning (its domain is the notebook's own); the
published radius of mini boson star II does not match its own table (an inconsistency in print; the
mini BS I values match); and the BS turning-point check (B1) and the QH negative-lag screen (B2) are
the gates that make BS σ_c = 0.270963 and QH E_Pl M = 10^6 inconclusive.

The gates were audited on the actual waveforms of all 16 entries: no rule turned out too weak to keep,
and no run had to be repeated. The smallest margin between the end of an analysis window and the first
possible return of a signal from the computational boundary is 68.8 M.

## 3. Independent checks

**GS and WH: frequency-domain reconstruction.** The response is rebuilt from the BH signal and the
frequency-dependent reflection of the ECO interior, and compared with the direct evolution in the late
and full windows without fitting. The declared gate allows ten times the combined numerical budget of
the two constructions; the measured disagreements are 0.40 to 0.85 of the budget itself, so the gate
would pass even at factor 1. The two constructions share the potential, the initial data and the BH
source run, and differ in the method of solution.

**QH: causal solver.** An independent time-domain solver with a reflecting wall in the near-horizon
region reproduces the production correction to 0.053% (10^8), 0.0083% (10^10) and 0.0082% (10^12) in the
late window, against a 1% gate.

**BS: no independent method.** The boson-star waveforms rest on convergence and controls only; an
independent-method comparison is a stated gap. The equilibrium itself agrees with the published
table (RESULTS.md, section 5).

## 4. Reflection coefficients

The outgoing solution is started from the asymptotic series (truncation error ≤ 1.3e-13) instead of a
plane wave, which removes a barrier-coefficient error of up to 9.7e-5 in the frozen notebook's
shortcut. Two variants (half step; start radii ×1.5) and an independent adaptive solver agree with the
production arrays to 4.4e-11 (GS interface: 3.0e-10), far inside the 1e-6 rule. Delays are quoted only
where their state is `RESOLVED`. The frequencies where they are not are listed per entry in RESULTS.md
(section 3); they are mostly barrier coefficients above Mω ≈ 0.8–1.0, where |R_barrier| becomes small
and its phase can no longer be resolved or defined.

## 5. Boson-star modes withdrawn

The baseline notebook's BS poles were followed into the library and tested by changing the radius at
which the interior solution is matched to the Schwarzschild exterior. Between 1.4 and 2.6 R99 they move
by 19–29%, while the condition number of the matching grows; their small numerical spread measured only
the root solver. They are therefore not modes of the star and are listed as withdrawn
(`results/checks/bs-mode-matching-study.json`).

## 6. Quantum horizon at E_Pl M = 10^6

The 5.2% disagreement with the causal solver at E_Pl M = 10^6 does not shrink with resolution or with a
longer Fourier window. The reason is the prescription: its reflection kernel has a response before its
cause (negative-lag fraction ≥ 7.2e-4 at every near-horizon plane), which a causal solver cannot
represent, and the two differ by at least 1.49% wherever the wall is placed. The entry stays
inconclusive, the screen was not relaxed, and validated QH statements are limited to E_Pl M from 10^8 to
10^12 at γ = 1 (`results/checks/qh-1e6-study.json`).

## 7. Robustness

* **Source width:** at the B0 points (σ = 3, 4, 5 M) the residuals change by 28–48% per unit of width:
  results hold for the stated experiment only.
* **Observer position:** moving the observer from r* = 95 to 120 M changes the late residual at fixed
  retarded time by about 2%, at the B0 points and at the edge points GS 0.49 and WH 2.0005; orders and
  budgets are unchanged (`results/checks/edge-robustness-o120.json`).
* **Source position:** not studied.
* **Record length:** for each entry, the length a record would need to follow the slowest verified mode
  (`results/checks/record-length-requirements.json`); the slow WH decay, the GS 0.49 ringing beyond 1.3
  decay times and the later QH packets are outside the 520 M records.

## 8. Reproduction

`python -m eco_library reproduce` recomputes the whole library from the frozen notebook's equations,
and `python -m eco_library compare` checks the result against `results/`: every number, string and array
must agree, exactly on the same computer and within 1e-12 (relative) on another; only provenance fields
(file names, checksums, the producing commit, and the number of archive files hashed) are excluded,
listed in `eco_library/pipeline.py`.

The committed results were produced by such a reproduction with the code in this repository (Linux,
CPython 3.10.12, numpy 1.26.4, scipy 1.15.3, h5py 3.13.0). Two separate full reproductions on that
computer gave identical results, every number and every array bit for bit. Both are also identical to
the results of the original development runs, which used the same physics code before the repository
was reorganised; the only differences are file names and checksums. After the stage folders received
their final names (a change of file paths only), block B0 was computed once more: all its waveforms are
bitwise identical to the committed result files, and its block summary and transfer checks agree exactly.

The same code was checked on Windows (CPython 3.11.9) during development: the catalogue rebuilt there
differed from the Linux one in 41 derived numbers, by at most 7.9e-16 (relative), which is
floating-point rounding. This is why `compare` allows 1e-12 on another computer.

## 9. The supporting studies

The studies behind sections 3–7 are not part of `reproduce`; their results in `results/checks/` are
inputs of the catalogue. Their scripts are in `scripts/studies/` (each states its usage in its first
lines), and the robustness runs use `python -m eco_library robustness`. Some study files combine script
output with blocks written from it by hand; the scripts say which.

| File | Study | Script |
|---|---|---|
| `transfer-gate-review.json` | margins and independence of the GS/WH frequency-domain check | `scripts/studies/transfer_gate_review.py` |
| `bs-mode-matching-study.json` | BS poles against the matching radius | `scripts/studies/bs_mode_matching_study.py` |
| `qh-1e6-study.json` | QH kernel screen and causal solver at E_Pl M = 10^6 | `scripts/studies/qh_kernel_screen.py`, `scripts/studies/qh_pde_wall.py` |
| `waveform-audit.json` | gates on the waveforms of all entries | `scripts/studies/waveform_audit.py` |
| `edge-robustness-o120.json` | observer moved at the edge points | `scripts/studies/edge_robustness_report.py` |
| `record-length-requirements.json` | record length per entry | `scripts/studies/record_length_requirements.py` |
| `robustness-report-summary.json` | source width and observer at the B0 points | `python -m eco_library robustness`, `scripts/summarize_robustness.py` |
| `long-record-check-wh.json` | WH 2.005 recorded to 1000 M | `scripts/long_record_check.py` |
