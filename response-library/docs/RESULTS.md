# Results

The response library holds 16 entries: four parameter points for each of GS, WH, BS and QH, all for
the experiment E0 (method: [METHOD.md](METHOD.md); checks: [VALIDATION.md](VALIDATION.md)). The
numbers below are copied from `results/catalogue.json`; the waveforms, residuals and reflection
coefficients are in `results/entries/`.

**14 entries are accepted and 2 are inconclusive. None is rejected.**

## 1. Entries

| Model | Parameter | Role | State | p (late) | Budget/norm late | full | Independent check | Modes |
|---|---|---|---|---|---|---|---|---|
| GS | M/a = 0.4 | lower edge | accepted | 2.03 | 4.8e-05 | 5.7e-05 | pass, 0.85 of budget | 1 verified |
| GS | M/a = 0.44 | interior | accepted | 2.03 | 7.8e-05 | 6.9e-05 | pass, 0.83 of budget | 1 verified |
| GS | M/a = 0.47 | baseline notebook's point | accepted | 2.02 | 9.9e-05 | 8.2e-05 | pass, 0.82 of budget | 1 verified |
| GS | M/a = 0.49 | upper edge | accepted | 2.01 | 1.5e-04 | 1.2e-04 | pass, 0.81 of budget | 1 verified |
| WH | r0/M = 2.0005 | lower edge, baseline notebook's point | accepted | 4.01 | 9.6e-08 | 2.0e-07 | pass, 0.40 of budget | 4 verified |
| WH | r0/M = 2.005 | interior | accepted | 4.01 | 5.5e-08 | 2.1e-07 | pass, 0.56 of budget | 4 verified |
| WH | r0/M = 2.02 | interior | accepted | 3.99 | 6.6e-08 | 1.7e-07 | pass, 0.54 of budget | 4 verified |
| WH | r0/M = 2.05 | upper edge | accepted | 3.92 | 8.7e-08 | 2.0e-07 | pass, 0.69 of budget | 4 verified |
| BS | σ_c = 0.18 | lower edge | accepted | 4.02 | 5.0e-08 | 3.3e-07 | none (gap) | withdrawn |
| BS | σ_c = 0.22 | interior | accepted | 3.86 | 5.0e-08 | 5.0e-07 | none (gap) | withdrawn |
| BS | σ_c = 0.25 | interior | accepted | 3.91 | 5.1e-08 | 6.6e-07 | none (gap) | withdrawn |
| BS | σ_c = 0.270963 | upper edge (maximum-mass star) | **inconclusive** | 4.02 | 5.1e-08 | 7.8e-07 | none (gap) | withdrawn |
| QH | E_Pl M = 10^6 | lower edge | **inconclusive** | 4.00 | 2.2e-04 | 2.2e-04 | fail, 5.2% | none |
| QH | E_Pl M = 10^8 | baseline notebook's point | accepted | 4.00 | 4.3e-06 | 4.4e-06 | pass, 0.053% | none |
| QH | E_Pl M = 10^10 | interior | accepted | 4.00 | 3.5e-06 | 3.5e-06 | pass, 0.0083% | none |
| QH | E_Pl M = 10^12 | upper edge | accepted | 4.00 | 3.5e-06 | 3.5e-06 | pass, 0.0082% | none |

Columns: *p (late)* is the observed convergence order of the ECO sequence in the late window
(declared order 4; 2 for GS). *Budget/norm* is the error budget of the observable (BH-subtracted
residual; QH: the reflected correction) divided by the observable's norm, in the late and full
windows; the acceptance limit is 0.1, and the largest accepted value is 1.5e-4. *Independent check*:
for GS and WH the disagreement between the direct evolution and the frequency-domain reconstruction,
as a fraction of the combined numerical budget (pass below the declared gate); for QH the relative
late-window difference from an independent causal solver (gate 1%); BS has no independent method yet.

### The two inconclusive entries

* **BS σ_c = 0.270963** (mini boson star I, the maximum-mass star). Its numerics converge (p = 4.02,
  budget 5e-8 of the residual), but the model gate fails: this star sits exactly at the turning point
  between the stable and unstable branches, where the equilibrium sequence cannot confirm that the
  star is on the stable side, and its background fails the second-order convergence test under denser
  profile sampling. A linear response there is not a statement about a stable star.
* **QH E_Pl M = 10^6.** The independent causal solver differs from the production result by 5.2% in
  the late window against a 1% gate. The difference comes from the prescription itself: at this
  hierarchy the reflection kernel keeps a response before its cause (a negative-lag fraction of at
  least 7.2e-4 at every near-horizon plane), which a causal solver cannot reproduce. Validated QH
  results are therefore limited to E_Pl M from 10^8 to 10^12 (VALIDATION.md, section 6).

## 2. Complex modes

| Entry | Modes Mω (branch label) |
|---|---|
| GS M/a = 0.4 | 0.45020 − 1.11e-01i |
| GS M/a = 0.44 | 0.44042 − 6.36e-02i |
| GS M/a = 0.47 | 0.40510 − 2.50e-02i |
| GS M/a = 0.49 | 0.33692 − 3.51e-03i |
| WH r0/M = 2.0005 | 0.18692 − 6.62e-06i (even); 0.22813 − 4.76e-05i (odd); 0.34712 − 3.25e-03i (even); 0.38897 − 1.05e-02i (odd) |
| WH r0/M = 2.005 | 0.25023 − 1.38e-04i (even); 0.30905 − 1.35e-03i (odd); 0.42537 − 2.69e-02i (even); 0.47365 − 6.04e-02i (odd) |
| WH r0/M = 2.02 | 0.30604 − 1.33e-03i (even); 0.37061 − 1.00e-02i (odd); 0.47906 − 7.60e-02i (even); 0.53192 − 1.41e-01i (odd) |
| WH r0/M = 2.05 | 0.34946 − 5.89e-03i (even); 0.41130 − 2.96e-02i (odd); 0.51172 − 1.37e-01i (even); 0.56096 − 2.30e-01i (odd) |

Values are Mω (Im < 0: damped), each `VERIFIED_BY_VARIATION` (numerical spread 2.6e-13 to 1.2e-12).
The lists are the branches followed from the baseline notebook's modes; they are not complete, and no
branch is identified with a BH overtone. For comparison, the BH (Schwarzschild, l = 2) quasinormal
frequencies are 0.37367 − 0.08896i, 0.34671 − 0.27391i and 0.30105 − 0.47828i.

* **BS: withdrawn.** The boson-star poles of the baseline move by 19–29% when the radius at which the
  interior is matched to the exterior is changed from 1.4 to 2.6 R99, while the matching condition
  loses its conditioning; they are numerical artefacts of the matching, not modes of the star.
* **QH: none.** The prescription is defined for real frequencies only, so it has no poles to compute.

## 3. Reflection coefficients and delays (GS and WH)

| Entry | Coefficient | Evidence | Independent solver: max difference | Delay RESOLVED at | Not resolved |
|---|---|---|---|---|---|
| GS M/a = 0.4 | R_barrier | VERIFIED_BY_VARIATION | 4.6e-12 | 597 of 597 | — |
| GS M/a = 0.4 | R_interface | VERIFIED_BY_VARIATION | — | 597 of 597 | — |
| GS M/a = 0.44 | R_barrier | VERIFIED_BY_VARIATION | 4.8e-12 | 597 of 597 | — |
| GS M/a = 0.44 | R_interface | VERIFIED_BY_VARIATION | — | 597 of 597 | — |
| GS M/a = 0.47 | R_barrier | VERIFIED_BY_VARIATION | 3.3e-12 | 593 of 597 | unresolved at 4 frequencies in Mω 1.2975–1.4325 |
| GS M/a = 0.47 | R_interface | VERIFIED_BY_VARIATION | — | 597 of 597 | — |
| GS M/a = 0.49 | R_barrier | VERIFIED_BY_VARIATION | 5.6e-12 | 491 of 597 | unresolved at 106 frequencies in Mω 0.8825–1.5 |
| GS M/a = 0.49 | R_interface | VERIFIED_BY_VARIATION | — | 597 of 597 | — |
| WH r0/M = 2.0005 | R_barrier | VERIFIED_BY_VARIATION | 2.4e-11 | 324 of 597 | phase ill-conditioned in Mω 0.82–1.5 |
| WH r0/M = 2.0005 | R_interface | VERIFIED_BY_VARIATION | 3.2e-11 | 595 of 597 | unresolved at 2 frequencies in Mω 0.205–0.2075 |
| WH r0/M = 2.005 | R_barrier | VERIFIED_BY_VARIATION | 1.4e-11 | 309 of 597 | phase ill-conditioned in Mω 0.7825–1.5 |
| WH r0/M = 2.005 | R_interface | VERIFIED_BY_VARIATION | 1.3e-11 | 597 of 597 | — |
| WH r0/M = 2.02 | R_barrier | VERIFIED_BY_VARIATION | 8.7e-12 | 398 of 597 | unresolved at 137 frequencies in Mω 0.9175–1.345; phase ill-conditioned in Mω 1.348–1.5 |
| WH r0/M = 2.02 | R_interface | VERIFIED_BY_VARIATION | 8.6e-12 | 597 of 597 | — |
| WH r0/M = 2.05 | R_barrier | VERIFIED_BY_VARIATION | 4.7e-12 | 511 of 597 | unresolved at 86 frequencies in Mω 1.005–1.5 |
| WH r0/M = 2.05 | R_interface | VERIFIED_BY_VARIATION | 7.4e-12 | 597 of 597 | — |

Every array is `VERIFIED_BY_VARIATION`: its variants and an independent solver agree to better than
1e-6. Group delays are quoted only at frequencies whose delay state is `RESOLVED`. At low frequency
the round-trip delay of the cavity, τ_loop = τ_barrier + τ_interface, is about 9–16 M for the
gravastars and 12–29 M for the wormholes. The GS 0.47 barrier coefficient passes close to zero near
Mω ≈ 0.585, and the WH even/odd mode pairs sit on either side of the peaks of τ_loop (WH 2.005: peak at
Mω = 0.278, modes at 0.250 and 0.309).

## 4. What the signals look like

* **GS:** the residual rings at the gravastar's mode and decays with time constants of about 9, 16, 40
  and 285 M for M/a = 0.40, 0.44, 0.47 and 0.49; the most compact star rings beyond the record.
* **WH:** the throat traps waves between the two potential barriers; the slowest modes decay on 170 M
  (r0/M = 2.05) up to 1.5 × 10^5 M (2.0005), far longer than the 520 M record.
* **BS:** the residual is a short, quickly damped stellar response: about 3% of the BH prompt peak at the
  start of the late window, smaller by eight orders of magnitude by 400 M, and nearly the same for all
  four stars.
* **QH:** the correction (about 1e-3 of the BH prompt peak) arrives as a delayed packet after the BH
  ringdown, near 260, 300, 340 and
  375 M for E_Pl M = 10^6, 10^8, 10^10 and 10^12: the delay grows with the logarithm of the hierarchy.

None of this is an echo classification: deciding whether a late signal is a clean echo train, a
trapped resonance or a weak delayed correction is stage 3 of the project.

## 5. Comparisons with published values

| Model | Comparison | Result |
|---|---|---|
| BH | Schwarzschild l = 2 quasinormal frequencies (Leaver) | n = 0, 1, 2 within 1.6e-8 (relative); the time-domain extraction of n = 0 within 0.96% |
| WH | Yang–Xia–Miao, EPJC 85, 742 (2025), Table 6, at r0/M = 2.12 and 2.2 (outside this library's points) | even-parity poles within 1.1e-4 and 6.2e-5 |
| BS background | Macedo et al., PRD 88, 064046 (2013), Table I, mini BS I | mass 0.633001 (0.633), ω_s/μ 0.8530868 (0.853087), R99 μ 7.86158 (7.86149) |
| GS | Pani et al., arXiv:0909.0287, axial junction | equations matched; no tabulated values compared |
| QH | Oshita–Wang–Afshordi, arXiv:1905.00464v3, Eqs. 2.13–2.16 | formula matched; hierarchy identity to 1e-12 |

## 6. What can and cannot be concluded

For the 14 accepted entries: the axial l = 2 response to experiment E0 is numerically resolved
(Richardson-type estimates with controls, not rigorous bounds); the BH-subtracted residual (QH: the
correction) is resolved in the late and full windows; the GS and WH responses agree with an
independent frequency-domain construction and the QH responses with an independent causal solver; the
GS and WH reflection coefficients and their delays (at `RESOLVED` frequencies) are converged to better
than 1e-6; and the stated GS and WH modes are verified.

Not covered: echo classification, detectability, a mapping of ECO modes to BH overtones, rigorous error
bounds, any other experiment (the residuals change by 28–48% per unit of source width), polar
perturbations, rotation, BS modes, QH modes, QH below E_Pl M = 10^8, dynamics after 520 M (the slow
WH decay, the GS 0.49 ringing and later QH packets), and the completeness of the mode lists.
