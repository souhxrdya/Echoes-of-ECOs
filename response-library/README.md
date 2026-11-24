# Response library

The responses of the four ECO models, computed from the frozen equations of [stage 1](../baseline)
and checked until every number has a stated, controlled error. For each model there are four parameter
points spanning a declared domain; each entry is the axial l = 2 response to one fixed experiment (a
Gaussian pulse of width 4 M from r* = 55 M, observed at r* = 95 M for 520 M), computed at three
resolutions together with a matched black-hole reference.

## Start here

* **[walkthrough.ipynb](walkthrough.ipynb)**: a guided tour with the code, the numbers and the figures:
  how one entry is computed and checked, the whole library as a table and as figures, the reflection
  coefficients and modes, and one result file opened up. It opens on GitHub with all outputs.
* [docs/RESULTS.md](docs/RESULTS.md): the results and what can and cannot be concluded from them.
* [docs/METHOD.md](docs/METHOD.md): the equations, the computation and the acceptance rule.
* [docs/VALIDATION.md](docs/VALIDATION.md): every check and its outcome.

## Results

| Model | Parameter | State | Error budget / size (late, full) | Independent check | Modes |
|---|---|---|---|---|---|
| GS | M/a = 0.4 | accepted | 5e-05, 6e-05 | pass, 0.85 of budget | 1 verified |
| GS | M/a = 0.44 | accepted | 8e-05, 7e-05 | pass, 0.83 of budget | 1 verified |
| GS | M/a = 0.47 | accepted | 1e-04, 8e-05 | pass, 0.82 of budget | 1 verified |
| GS | M/a = 0.49 | accepted | 2e-04, 1e-04 | pass, 0.81 of budget | 1 verified |
| WH | r0/M = 2.0005 | accepted | 1e-07, 2e-07 | pass, 0.40 of budget | 4 verified |
| WH | r0/M = 2.005 | accepted | 6e-08, 2e-07 | pass, 0.56 of budget | 4 verified |
| WH | r0/M = 2.02 | accepted | 7e-08, 2e-07 | pass, 0.54 of budget | 4 verified |
| WH | r0/M = 2.05 | accepted | 9e-08, 2e-07 | pass, 0.69 of budget | 4 verified |
| BS | σ_c = 0.18 | accepted | 5e-08, 3e-07 | none (gap) | withdrawn |
| BS | σ_c = 0.22 | accepted | 5e-08, 5e-07 | none (gap) | withdrawn |
| BS | σ_c = 0.25 | accepted | 5e-08, 7e-07 | none (gap) | withdrawn |
| BS | σ_c = 0.270963 | **inconclusive** | 5e-08, 8e-07 | none (gap) | withdrawn |
| QH | E_Pl M = 10^6 | **inconclusive** | 2e-04, 2e-04 | fail, 5.2% | none |
| QH | E_Pl M = 10^8 | accepted | 4e-06, 4e-06 | pass, 0.053% | none |
| QH | E_Pl M = 10^10 | accepted | 3e-06, 4e-06 | pass, 0.0083% | none |
| QH | E_Pl M = 10^12 | accepted | 3e-06, 4e-06 | pass, 0.0082% | none |

14 entries are accepted and 2 are inconclusive; their reasons are in [docs/RESULTS.md](docs/RESULTS.md).
The error budget column is the numerical error of the observable (the BH-subtracted residual; for QH
the reflected correction) divided by its size, in the late and full windows; an entry needs at most
0.1. GS and WH have their own complex modes (`verified`); the boson-star poles of the baseline turned
out to be artefacts of the matching and are withdrawn; the quantum-horizon prescription has no modes.

## Use it

From this folder, with Python 3.10 or 3.11:

```
pip install -r requirements.txt
python -m eco_library audit          # a few seconds: checks every committed result against its evidence
jupyter notebook walkthrough.ipynb   # the guided tour
python -m pytest tests -q            # the test suite (about two minutes)
```

To recompute the whole library from the frozen equations and compare it with the committed results:

```
python -m eco_library reproduce --workers 4
python -m eco_library compare
```

On the computer that produced the committed results (2 cores), a full reproduction took about 2.2
CPU-hours: 1.7 for the 212 time-domain simulations, which run in parallel (`--workers`), and about
30 minutes for the transfer checks, modes and reflection coefficients, which run one after another.
With 4 workers expect roughly an hour. `reproduce` can be stopped at any time and continues where it stopped when run again
(`--budget SECONDS` stops it after a given time). Everything it writes goes to `runs/`, which Git
ignores. `compare` passes when every number and array agrees with `results/` (identical on the same
kind of computer, within 1e-12 relative elsewhere).

## Folder contents

| Path | Content |
|---|---|
| `walkthrough.ipynb` | the guided tour |
| `results/catalogue.json` | all 16 entries: state, convergence, error budgets, controls, independent checks, modes, reflection summary, reasons |
| `results/entries/` | one HDF5 file per entry: ECO and BH signals at three resolutions, residual, reflection coefficients and delays, modes, with checksums |
| `results/entries-index.json` | index of the entry files |
| `results/checks/` | the supporting studies (robustness, QH and BS studies, transfer review, record lengths) |
| `results/figures/` | the walkthrough's figures as PNG files |
| `eco_library/` | the code (`python -m eco_library` lists the commands) |
| `scripts/` | scripts the runs import, and the supporting studies in `scripts/studies/` |
| `tests/` | the test suite |
| `docs/` | method, results and validation |
| `requirements.txt` | the package versions used |

## Scope

Linear, nonrotating, axial perturbations, one experiment. The boson-star perturbation is a test field
on a fixed background, and the quantum horizon is a reflection prescription, not a spacetime. Echo
classification, distinguishability and detectability are later stages of the project.
