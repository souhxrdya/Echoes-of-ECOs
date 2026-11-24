> Validation record written when the notebook was frozen. It refers to the executed copy of the notebook
> and to the replay script `verify_freeze_release.py` used at the time; neither is part of this repository,
> which keeps only the frozen notebook and its outputs.

# ECO axial waveform release - validation and freeze record

## Decision

Ready to freeze this tested linear-axial release. The notebook was executed from
a fresh kernel, in order, and the separate verifier replayed exported data.
All 80 required regression checks passed. No fitted time, amplitude or
phase adjustment enters the transfer-reconstruction acceptance criteria.
In addition, 13 limiting-case checks,
9 independently regenerated baseline comparisons,
20 controlled-geometry checks, and 408 extra-bank
matrix/input replay checks passed. The separate verifier recalculates these
from the saved records; it does not merely trust the pass flags.

The scope is the five explicitly specified linear axial models and the recorded
parameter ranges. This is not a certificate for arbitrary parameters, polar or
rotating perturbations, nonlinear mergers, all ECO models in the bibliography,
or the absence of every possible software defect.

## Reconstructed versus independently evolved signals

Raw relative L2 on 190 <= t/M <= 500, divided by the direct correction norm:

| model   |   raw_relative_L2 |
|:--------|------------------:|
| GS      |       0.000322501 |
| WH      |       6.43457e-07 |
| QH      |       0.00141394  |

The acceptance threshold is 1%. Both sides of the wormhole and its separate
spatial-reflection families are additionally checked. The source is measured
as field plus spatial derivative, not inferred from a fitted output waveform.
All 44 analytic characteristic controls and 36 one-factor trials are retained.

## Independent literature tests

- Schwarzschild: three independently solved Leaver frequencies reproduce known
  reference values; free time-domain fits are separate evidence.
- Pani gravastar: 45 evaluations of the published analytic regular
  hypergeometric interior, real and complex frequencies, analytic derivatives,
  and metric-variable shell junctions. Maximum reflection-coefficient error:
  1.43228e-14. This is the zero-surface-energy
  shell subclass, not every gravastar family.
- Schwarzschild thin-shell axial wormhole: three published Table 6 frequencies
  reproduced with maximum relative complex discrepancy
  0.000150444. Complex outgoing zeros replace phase
  crossings. The parameter scan follows identified branches continuously and
  logs exclusions; it does not relabel a converged wrong overtone.
- Mini-boson star I: both published axial frequencies reproduced with maximum
  relative discrepancy 0.00103256. The paper's
  finite scalar-tail truncation is included explicitly. Its matching-radius
  sensitivity is as large as 4.5483%,
  separate from tightly converged numerical integration. It is not legitimate
  to report the untruncated star's frequencies at machine precision.
- OWA horizon: the published amplitude and phase, including the standard-r*
  lapse constant, are used. Its complex poles are derived effective-model
  predictions with residual/convergence tests, not claimed published frequency
  benchmarks.

Primary references: [Pani et al.](https://arxiv.org/abs/0909.0287),
[Yang, Xia and Miao](https://doi.org/10.1140/epjc/s10052-025-14446-w),
[Macedo et al.](https://arxiv.org/html/1307.4812),
[Oshita, Wang and Afshordi](https://arxiv.org/html/1905.00464), and
[Mark et al.](https://arxiv.org/abs/1706.06155).

The supplied Damour-Solodukhin and Qian et al. papers describe different
wormhole/scalar-potential problems; their results are not misrepresented as
axial thin-shell benchmarks. Rogue Echoes includes a more general gravastar
family than the massless-shell axial subclass here. The duplicate Pani PDF is
one reference, not two independent confirmations.

## Corrections that materially changed this release

1. The OWA coefficient is global. It is rephased to the computational plane
   before insertion into the local transfer formula, removing the previous
   double counting of propagation distance. Moving only that plane by 10M
   changes the physical waveform by 5.63771e-08 in
   relative norm. Independent PDE errors and negative-lag kernel truncation
   are recorded for both planes, not hidden.
2. Actual complex modes are computed for every model. WH scan branch identity
   is maintained by continuation. No FFT peak or real-axis phase crossing is
   accepted as an exact QNM.
3. Cross-model matrices use one bank with the same ADM mass, source/observer
   areal radii, initial strength and width. All 100 pair entries are replayed
   from the exported bank. Native reconstruction experiments remain separately
   identified, with matching BH controls, including the native BS experiment.
4. Wormhole timing refinement holds duration and sampling support fixed. Its
   peak-index timing uncertainty uses the observed grid spread and sampling
   floor, not an unjustified fourth-order Richardson divisor.
5. Published analytic/numerical benchmarks and saved-data replay complement
   internal TD/FD agreement, which alone cannot exclude a shared physics error.
6. The energy calculation distinguishes the evolved Regge-Wheeler field from
   the CPM strain master; finite-radius flux estimates use the RW normalization.
   The detector overlap converts the tapered RW spectra to strain shape in
   the detector band. Intrinsic mismatches compare the unconverted RW fields.
   A known-tone leakage calibration and Parseval identity accompany the PSDs.
   See [Martel and Poisson](https://doi.org/10.1103/PhysRevD.71.104003).

## Source and limiting-case tests

The complete five-model experiment is regenerated for the baseline, widths
3M and 5M, and a second parameter configuration. All 16 matrices (400 entries)
are recalculated from the exported waveforms. QH inputs are independently
measured in every source run. Geometry, normalization, boundary margins,
baseline agreement and the observed ranges are recorded in
`controlled_bank_*.csv` and `controlled_bank_exercises.h5`. These finite
experiments measure sensitivity; they do not assert source-independent model
identification or universal mismatch rankings.

Increasing GS compactness and decreasing WH throat offset yield a resolved
approach to the BH waveform in the fixed early observation window, tested at
two resolutions near the limit. Later-time differences remain. These checks
therefore do not imply that the ECO spectra converge to the BH spectrum.
Increasing the QH Planck hierarchy changes phase and delay, not the reflection
magnitude at fixed dimensionless frequency. The chosen mini-boson-star family
has no BH limit; none is manufactured for the comparison.

## Reproducibility

Upload `eco-waveform-final.ipynb` to Kaggle and Run All on CPU. Standard NumPy,
SciPy, Matplotlib, Pandas, h5py and mpmath are required; no data downloads occur.
The executed copy contains the local results. The HDF5 database includes both
native and controlled waveform records and the selected complex-mode inventory.

For independent replay, extract this archive and run:

`python verify_freeze_release.py --output-root .`

`verified_results.json` contains machine-readable measured checks. The archive's
SHA-256 manifest identifies the exact delivered notebook and evidence files.
Scientific claims should cite the stated model, mode subset, pulse, time window
and numerical/systematic error separately. Failure to identify clean repeated
echoes is a valid physical result, not a reason to insert or tune echo trains.
