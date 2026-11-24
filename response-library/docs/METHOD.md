# Method

How the response library is computed and when an entry is accepted. The results are in
[RESULTS.md](RESULTS.md) and the checks behind them in [VALIDATION.md](VALIDATION.md).

## 1. What an entry is

An entry is the axial l = 2 response of one model at one parameter value to one fixed experiment, E0:
a unit Gaussian pulse in Ψ of width σ = 4 M, centred at Schwarzschild tortoise coordinate r* = 55 M and
moving inwards, recorded at r* = 95 M for 520 M. Each model has four parameter points chosen to span a
declared domain (its two edges and two interior points, including the frozen notebook's own point):

| Model | Parameter | Points (role) |
|---|---|---|
| GS | compactness M/a | 0.40 (lower edge), 0.44, 0.47 (notebook point), 0.49 (upper edge) |
| WH | throat radius r0/M | 2.0005 (lower edge, notebook point), 2.005, 2.02, 2.05 (upper edge) |
| BS | central field σ_c | 0.18 (lower edge), 0.22, 0.25, 0.270963 (upper edge: mini boson star I, maximum mass) |
| QH | hierarchy E_Pl M at γ = 1 | 10^6 (lower edge), 10^8 (notebook point), 10^10, 10^12 (upper edge) |

The domains are the range covered by the library, not statements about where a model is physical.
The 16 points are computed in four blocks (B0–B3), each holding one point of every model:

| Block | GS | WH | BS | QH |
|---|---|---|---|---|
| B0 | 0.44 | 2.005 | 0.22 | 10^10 |
| B1 | 0.47 | 2.0005 | 0.270963 | 10^8 |
| B2 | 0.40 | 2.05 | 0.18 | 10^6 |
| B3 | 0.49 | 2.02 | 0.25 | 10^12 |

The analysis windows (time at the observer) are prompt [100, 190] M, ringdown [150, 190] M, late
[190, 500] M and full [100, 500] M.

## 2. Equations

The library does not re-implement the physics. It loads the declarations of the frozen notebook
(`../baseline/eco-waveform-final.ipynb`) by hash and calls them; `eco_library/baseline_runtime.py`
and `eco_library/frozen_declarations.py` refuse to run if the notebook's SHA-256 differs. The table
below maps each model's equations to the code.

Conventions for every model: linear, nonrotating, axial (odd-parity) gravitational perturbations,
l = 2; G = c = 1 and the ADM mass M = 1 (BS backgrounds are solved in units of the scalar mass mu and
rescaled by r/M = r_mu/M_mu, t/M = t_mu/M_mu, V M^2 = V_mu M_mu^2); time dependence e^{-i omega t},
damped modes Im omega < 0; physical spectrum X(omega) = int x(t) e^{+i omega t} dt; f = 1 - 2M/r,
Schwarzschild tortoise r* = r + 2M ln(r/2M - 1). "Notebook" is the frozen `../baseline/eco-waveform-final.ipynb`
(SHA-256 `64e873da...6513b9`), whose declarations the library loads by hash.

| Model | Equations implemented | Code | Parameter meaning | Literature or external check | Not included |
|---|---|---|---|---|---|
| BH | Regge-Wheeler: d^2Psi/dr*^2 + [omega^2 - V]Psi = 0, V = f [l(l+1)/r^2 - 6M/r^3] | notebook `rw_potential`, `r_from_schwarzschild_rstar` (cell 6); `evolve_wave` FD4/RK4 | none | Leaver QNMs (baseline benchmark) | rotation, polar sector |
| GS | interior de Sitter f_in = 1 - H^2 r^2 with H^2 = 2M/a^3 so that f is continuous at the shell (zero surface energy); interior axial potential f_in l(l+1)/r^2 (the -6m/r^3 and 4pi(rho - p) terms cancel for p = -rho); exterior Regge-Wheeler; junction: Psi and dPsi/dr* continuous (Pani et al. 2009 Eq. 3.29 with f and h continuous) | notebook `gravastar_grid` (cell 13: shell on a node, averaged shell potential, regular-centre stencil), `gs_regular_shell`/`gs_shell_reflectivity` (cell 27, closed-form angular-momentum ladder), `gravastar_regular_data` (cell 13, radial ODE) | compactness = M/a, 0 < M/a < 1/2; a = shell areal radius | Pani et al. arXiv:0909.0287 axial junction; FD-TD agreement (transfer gate review); independent radial ODE for R_interface | polar perturbations and shell dynamics, other gravastar classes |
| WH | two Schwarzschild exteriors joined at the throat r0; in the throat-centred tortoise distance x, V = f l(l+1)/r^2 - 6Mf/r^3 on both sides plus Lambda delta(x), Lambda = 2(r0 - M)/r0^2, i.e. Psi'(0+) - Psi'(0-) = Lambda Psi(0); throat r = Lambda/(2 i omega - Lambda), t = 2 i omega/(2 i omega - Lambda) | notebook `wormhole_grid` (cell 18), `throat_delta_coefficients`, `wh_seam_transfer` (cell 26), `wh_outgoing_cauchy` (cell 22) | r0_over_M = r0/M > 2 | Yang-Xia-Miao EPJC 85, 742 (2025): Table 6 values within 1.1e-4 (r0/M = 2.12) and 6.2e-5 (2.2) | Damour-Solodukhin geometry (a different spacetime; Damour-Solodukhin 2007 and Qian et al. 2024 are not benchmarks for this WH) |
| BS | mini boson star, no self-interaction; Einstein-Klein-Gordon with sigma = sqrt(8 pi G) phi = sqrt(2) phi_tilde: a'/a = (1-a^2)/(2r) + r/2[(omega_s^2/alpha^2 + 1) a^2 sigma^2 + sigma'^2], alpha'/alpha = (a^2-1)/(2r) + r/2[(omega_s^2/alpha^2 - 1) a^2 sigma^2 + sigma'^2], sigma'' = -(2/r + alpha'/alpha - a'/a) sigma' - a^2(omega_s^2/alpha^2 - 1) sigma; BCs sigma(0) = sigma_c, sigma'(0) = 0, a(0) = 1, sigma(r_max) = 0, a alpha = 1 at r_max; axial perturbation on the fixed background with c = alpha/a: Psi = r^{l+1} y, y'' + [2(l+1)/r + c'/c] y' + Q y = 0, Q = (4 - l^2)(a^2-1)/r^2 - (l+2) a^2 sigma^2 + omega^2/c^2 (reduces to Regge-Wheeler in vacuum) | notebook `solve_mini_boson_star` (cell 15), `BosonAxialModeProblem` (cell 24); `eco_library.boson_star` | sigma_c = central sigma; nodeless continuation from the declared anchor | Macedo et al. arXiv:1307.4812 Table I (mass, frequency; radius of mini BS II inconsistent in print), Table IV only at the shared 1.4 R99 truncation | coupled polar scalar-metric perturbations; radial stability; self-interacting stars |
| QH | Schwarzschild exterior with the Oshita-Wang-Afshordi reflectivity, global plane x = 0: R(omega) = e^{-pi|omega|/kappa} e^{-2 i Im log A}, log A = (i q/kappa) ln[gamma q/(4 kappa E_Pl M e^{-1/2})] + ln Gamma(-2iq/kappa) - ln Gamma(-iq/kappa) - ln Gamma(1 - iq/kappa), q = |omega|, kappa = 1/(4M), R(-omega) = conj R(omega); local plane R_local = R e^{2 i omega x_s} | notebook `quantum_reflection`, `quantum_local_reflection` (cell 20); `eco_library.quantum_horizon` (measured-source reconstruction, independent causal PDE) | EplM = E_Pl M (hierarchy), gamma = OWA parameter (1 in every point) | arXiv:1905.00464v3 PDF Eqs. 2.13-2.16 (HTML 13-16); hierarchy identity R(EplM) = R(1e8) e^{8 i omega ln(EplM/1e8)} checked to 1e-12 | a quantum-object spacetime; complex-frequency continuation (no QH modes); the prescription is acausal at the 1e-3 level for E_Pl M = 1e6 |

### Experiment, coordinates and initial data

* Source and observer are given as Schwarzschild tortoise coordinates r*_s = 55, r*_o = 95 (E0), i.e.
  areal radii 48.699 M and 87.490 M. For GS, WH (right exterior) and QH these are exterior points of
  Schwarzschild, so global and native coordinates agree (WH native coordinate is offset by the throat
  r*0). For BS, which has no exact Schwarzschild exterior, the global value fixes the target areal
  radius and the adapter maps it to the star's own tortoise coordinate (observer native 95.505 at
  sigma_c = 0.22), with a 1e-9 inverse-map tolerance; the result files check every areal radius
  back to its global coordinate to 1e-9.
* Initial data (notebook `initial_characteristics`): Psi = exp(-(x - x_s)^2/(2 sigma^2)) with unit
  amplitude and no carrier, Pi = D1_4(Psi) (fourth-order derivative on the native grid) for a
  left-going pulse; u+ = Pi + Psi_x is ingoing, so F_in = 1 to the last bit. Interface projections
  (GS shell, WH throat, BS regular centre) act once before the first step and are negligible here
  because the pulse support starts more than 20 M from any interface. Every record carries an
  `initial_data_audit` that recomputes this construction and the cubic-Lagrange extraction and
  passes at 1e-12.
* Windows: prompt [100, 190], ringdown [150, 190], late [190, 500], full [100, 500] (shifted with the
  source and observer for other experiments); records to 520 M.

### Known limits of the frozen code relevant here

* `r_from_schwarzschild_rstar` saturates at r = 1388.9 M for r* > 1402 M (Lambert-W argument clipped
  at exp(700)); no library product uses r* beyond 1300 M.
* The fixed-step RK4 Jost shooting is not norm conserving: T_inf drifts by up to 2e-8 at omega >= 0.15
  (step 0.0125); reflection ratios B/A are insensitive (1e-11).
* The plane-wave outer start of the frozen `make_seam_scattering` neglects the l(l+1)/r^2 tail
  (barrier coefficient error up to 9.7e-5); replaced by the asymptotic outgoing series in the library's reflection computation.
* The BS pole matching (`BosonAxialModeProblem`) is exponentially ill-conditioned for the strongly
  damped selected poles (VALIDATION.md, section 5).

## 3. Time-domain computation and acceptance

**Resolutions.** Every model is evolved at three grid spacings, Δx = 0.2, 0.1 and 0.05 M (refinement
ratio 2; the GS grid is adjusted so that the centre and the shell are grid nodes), with fourth-order
finite differences and RK4. Each ECO sequence has a matched BH reference sequence computed the same
way (for GS a separate BH reference on the gravastar's grid, "BHGS"), so that the BH-subtracted
residual is formed level by level.

**Convergence order and error estimate.** For a quantity measured at the three levels (coarse c,
medium m, fine f) in a window, with L2 norms over the window,

  p = log2( ‖c − m‖ / ‖m − f‖ ),  error estimate of the fine level = ‖m − f‖ / (2^p − 1),

and the declared order must be confirmed: p within ±0.5 of 4 (2 for GS, whose shell junction limits
the order). The error estimate is a Richardson-type estimate that holds when the sequence is in its
asymptotic range; it is not a rigorous bound.

**Controls.** Besides refinement, every sequence has controls that change one numerical setting at
a time: the time step halved, the boundaries of the computational domain moved, and model-specific
settings (BS background tolerance, profile sampling and domain; QH Fourier settings). A control whose
difference is not small compared with the refinement estimate is counted in the error budget.

**Acceptance.** The observable is the BH-subtracted residual (GS, WH, BS) or, for QH, the correction
added by the horizon reflection. In each of the late and full windows the budget

  B = (residual error estimate) + (differences of every non-negligible control)

must satisfy B ≤ 0.1 N, with N the L2 norm of the observable in that window. An entry is
`ACCEPTED_V1` if, in addition, every hard check of every record passes, the declared orders are
confirmed, the BH reference matches the ECO run level by level, and the model gates pass:

* BS: the star must be on the stable branch below the maximum mass (sampled mass strictly increasing
  up to the target), and the background must converge under denser profile sampling;
* QH: an independent causal time-domain solver with the reflecting wall at a near-horizon plane must
  agree with the production result in the late window.

Otherwise the entry is `INCONCLUSIVE`, with its reasons recorded. `REJECTED` would mean a failed hard
check of a record. No threshold, window or order is adjusted after the runs.

## 4. Independent frequency-domain check (GS and WH)

For GS and WH the ECO response is rebuilt a second way: the BH signal is propagated to the matching
plane and combined with the frequency-dependent reflection and transmission of the ECO interior
(`eco_library/transfer.py`). The rebuilt signal is compared with the directly evolved one in the late
and full windows, without fitting any time, amplitude or phase. The gate declared before the runs
allows a disagreement of up to ten times the combined numerical budget of the two constructions; the
check, its variants and the conditioning of the transforms are stored per entry.

## 5. Complex modes

`eco_library/modes.py`: modes are searched in 0.05 ≤ Re(Mω) ≤ 1.0, −0.6 ≤ Im(Mω) < 0. The frozen
notebook's modes at its own parameter point are used only as starting values; each root is solved
again from the exact outgoing condition and the interior matching and continued step by step in the
parameter to the entry's point (a step is accepted only if the residual stays below tolerance and the
root moves less than a third of the distance to the nearest other root). The spread of the root under
changed numerical settings gives σ_num and the label `VERIFIED_BY_VARIATION`. The lists are not
exhaustive, and no ECO branch is identified with a BH overtone. QH has no modes in this library: the
prescription is defined for real frequencies only.

## 6. Reflection coefficients and delays (GS and WH)

`eco_library/reflection.py`, on the grid Mω = 0.01 + 0.0025 k (k = 0…596): the outgoing solution is
started from the optimally truncated asymptotic series of the Regge–Wheeler equation and carried to
the matching plane with the frozen fourth-order integrator. In the basis U = A e^{iωX} + B e^{−iωX},
R_barrier = B/A; R_interface is the reflection of the interior (closed form for GS, throat plus the
other exterior for WH). Two variants (half step; start radii ×1.5) and an independent adaptive solver
must agree to 1e-6 for the label `VERIFIED_BY_VARIATION`. The group delay τ = dφ/dω is a local
derivative with its own uncertainty σ_τ; at each frequency its state is `RESOLVED` (|R| above a floor
and σ_τ ≤ 1e-6 max(|τ|, 1 M)), `UNRESOLVED`, `PHASE_ILL_CONDITIONED` (|R| too small for a phase) or
`UNDEFINED`, and a delay is quoted only where it is `RESOLVED`. The round-trip delay of the cavity is
τ_loop = τ_barrier + τ_interface.

## 7. Quantum horizon

The QH model is the Oshita–Wang–Afshordi reflection prescription applied to the Schwarzschild exterior
(`eco_library/quantum_horizon.py`). The production result is built from the measured BH source and the
reflection kernel; the observable is the correction it adds to the BH signal. An independent causal
partial-differential-equation solver with a reflecting wall checks it (`scripts/qh_causal_control.py`,
`scripts/qh_spectral_controls.py`). The validated domain is E_Pl M from 10^8 to 10^12 at γ = 1 (see
VALIDATION.md for E_Pl M = 10^6).

## 8. Result files

`results/catalogue.json` holds every entry with its state, convergence numbers, controls, checks,
modes, reflection summary, identities and the reasons for any non-accepted state. `results/entries/`
holds one HDF5 file per entry: the ECO and BH signals at all three resolutions (native grids and a
common grid t_k = 0.07 k), the residual, the reflection coefficients with their variants and delay
states, the modes, and a header with the same evidence; every array is checksummed in the file's
manifest. `eco_library.interface_export.audit_export` re-derives each file's statuses from its arrays
and its catalogue entry.

## 9. Code map

| Module | Role |
|---|---|
| `baseline_runtime.py`, `frozen_declarations.py` | load the frozen notebook's declarations by hash |
| `solvers.py`, `boson_star.py`, `quantum_horizon.py` | time-domain runs of GS/WH/BH, BS and QH |
| `offfiducial.py` | plans and runs a block, analyses refinement and controls, writes its report |
| `convergence.py` | observed order and error estimate |
| `transfer.py`, `modes.py`, `reflection.py` | frequency-domain check, complex modes, reflection coefficients |
| `catalogue_base.py`, `catalogue.py` | assemble the catalogue; `status_audit` re-derives every state |
| `interface_export.py` | write, index and audit the per-entry result files |
| `pipeline.py` | `reproduce` and `compare` |
| `verification.py` | comparison rule across computers (identical, or 1e-12 relative for derived numbers) |
| `storage.py`, `schema.py`, `adapter_archive.py`, `run.py`, `adapter_pilot.py`, `replay_policy.py` | record format, identities, archives and the underlying runner |
