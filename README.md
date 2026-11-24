# Echoes of ECOs

Gravitational-wave signatures of exotic compact objects (ECOs). The aim is not only to produce
waveforms that differ from a black hole's, but to find which physical properties of an ECO produce
observable differences, whether those differences are robust, and whether they can tell one ECO model
from another in realistic gravitational-wave observations.

## Models

| | Model | Description |
|---|---|---|
| BH | Black hole | Schwarzschild; the reference every ECO is compared with |
| GS | Gravastar | thin shell with a de Sitter interior and a Schwarzschild exterior |
| WH | Wormhole | two Schwarzschild exteriors joined at a thin-shell throat |
| BS | Boson star | mini boson star (a self-gravitating scalar field); axial test-field perturbation on the star |
| QH | Quantum horizon | Schwarzschild with a phenomenological, frequency-dependent reflection near the horizon |

Throughout: linear, nonrotating, axial (odd-parity) l = 2 perturbations, G = c = M = 1.

## Stages

| Stage | Folder | Content | Status |
|---|---|---|---|
| 1 | [`baseline`](baseline) | the equations of all models implemented and validated in one notebook | done |
| 2 | `response-library` | converged, independently checked responses of each model at four parameter points | in progress |
| 3 | [`signatures`](signatures) | ECO–BH differences in ringdown, echoes, spectra, phase and group delay | planned |
| 4 | [`signature-atlas`](signature-atlas) | model-by-model atlas of which features are generic and which are model specific | planned |
| 5 | [`distinguishability`](distinguishability) | degeneracies: noise-weighted mismatches between the BH and the ECOs and among ECOs | planned |
| 6 | [`observability`](observability) | whether the differences are measurable with real detector noise | planned |
| 7 | [`injection-recovery`](injection-recovery) | synthetic signals in detector noise, analysed under BH and ECO hypotheses | planned |
| 8 | [`extensions`](extensions) | rotation, tides and binary systems | later |

Each stage folder is self-contained: its README explains what it does, what it found and how to
reproduce it, and it holds its own code, tests, documents and results.

## Where to start

[`baseline/README.md`](baseline/README.md): what the validated notebook computes and how to view or run it.

## References

The papers the models are based on are listed, with links, in [`references/README.md`](references/README.md).
