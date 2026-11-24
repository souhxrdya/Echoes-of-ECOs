"""Predeclared off-fiducial axial response pilot (plan offfiducial-pilot/0.3).

One physical point per model away from the frozen fiducials: GS M/a=0.44,
WH r0/M=2.005, BS sigma_c=0.22 and QH E_Pl M=1e10 (gamma=1).  Each has three
refinement levels, matched-resolution BH references, separate systematic
controls with an explicit per-factor coverage table and explicit acceptance
rules (docs/METHOD.md, amendment 0.3).  Every waveform job
carries one explicit immutable ``Experiment``; the frozen-replay experiment is
a separate constant.

Run from a repository checkout:

  python -m eco_library.offfiducial plan      --output DIR
  python -m eco_library.offfiducial run       --output DIR [--budget S] [--workers N] [--only LABEL ...]
  python -m eco_library.offfiducial report    --output DIR [--historical --report-output DIR2]
  python -m eco_library.offfiducial verify    --output DIR [--read-only]
  python -m eco_library.offfiducial verify    --output DIR --mode historical [--manifest FILE] [--report-output DIR2]
  python -m eco_library.offfiducial reproduce --output DIR --labels LABEL ... [--report-output DIR2]

Four verification questions are kept apart: immutable archive integrity
(byte hashes against a published manifest), resume compatibility (exact
producer provenance, required only to add records to a run), analysis of
historical records (checked readback with the producer provenance reported
beside the analyzer's, never replaced), and cross-platform reproducibility
(recomputation classified against a derived roundoff bound).

``run`` starts pending jobs only while their cost estimate fits the wall-time
budget and may be repeated; existing archives are reused only after checked
readback against the request and runtime provenance.  A job with a recorded
failure is not retried unless ``--retry-failed`` is given.  Successful
execution is reported separately from record checks and from scientific
acceptance.  Nothing produced here is a total error bound, an echo
classification, a mode claim or an atlas.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback
import uuid

import h5py
import numpy as np
from scipy.integrate import trapezoid

from . import replay_policy
from .adapter_archive import read_archive, request_id, write_archive
from .baseline_runtime import load_frozen_runtime
from .boson_star import (FIDUCIAL_SIGMA_C, SIGMA_C_DOMAIN, _background_diagnostics,
                         load_boson_runtime, run_boson_star, solve_equilibrium)
from .convergence import WINDOWS, assess, common_series
from .quantum_horizon import _runtime as _qh_runtime
from .quantum_horizon import independent_td, reflection, run_quantum_horizon
from .run import BASELINE_H5_SHA256
from .schema import Experiment, RunSpec, canonical_json, digest
from .solvers import refinement_spacings, run_td
from .storage import (_check_resume_provenance, assert_output_path, read_record,
                      runtime_provenance, write_json, write_record)

PLAN_VERSION = "offfiducial-pilot/0.3"
BLOCK_PLAN_VERSION = "library-v1-block/0.1"   # catalogue blocks: the same rules at declared points
REPORT_VERSION = "offfiducial-pilot-report/0.3"
REPO = Path(__file__).resolve().parents[1]
LEVELS = (.2, .1, .05)
LEVEL_NAMES = ("coarse", "medium", "fine")
POINTS = {"GS": {"compactness": .44}, "WH": {"r0_over_M": 2.005},
          "BS": {"sigma_c": .22}, "QH": {"EplM": 1e10, "gamma": 1.}}
FIDUCIALS = {"GS": {"compactness": .47}, "WH": {"r0_over_M": 2.0005},
             "BS": {"sigma_c": FIDUCIAL_SIGMA_C}, "QH": {"EplM": 1e8, "gamma": 1.}}
DECLARED_ORDER = {"BH": 4, "GS": 2, "WH": 4, "BS": 4, "QH": 4}
# Numerical settings shared by every waveform job.  The physical experiment (source,
# observer, width, conventions) is never read from module state: each job carries an
# explicit immutable Experiment (audit finding A, 2026-11-23).
NUMERICS = {"t_end": 520., "cfl": .35, "x_right": 400., "sample_every": 2}
PRODUCTION_EXPERIMENT = Experiment()          # production experiment E0 of the pilot
FROZEN_REPLAY_EXPERIMENT = Experiment()       # fixed: the frozen controlled baseline replay
REFERENCE_SOURCE_RSTAR, REFERENCE_OBSERVER_RSTAR = 55., 95.   # geometry of the declared windows
ESTIMATE_KEYS = ("p_observed", "fine_error_estimate_L2", "fine_relative_error_estimate")
AREAL_RADIUS_RELATIVE_TOLERANCE = 1e-9        # the BS adapter's own inverse-map tolerance
LOAD_MODES = ("resume", "historical")
# Control factors per model of a sequence.  A factor is COVERED by at least one control of
# the point that varies it on that sequence, NOT_APPLICABLE with a stated reason, or MISSING.
SEQUENCE_FACTORS = {
    "BH": ("right_domain", "left_domain", "time_step"),
    "GS": ("right_domain", "left_domain", "time_step"),
    "WH": ("right_domain", "left_domain", "time_step"),
    "BS": ("right_domain", "left_domain", "time_step", "bvp_tolerance", "background_profile_sampling",
           "background_domain"),
    "QH": ("time_step", "right_domain", "left_and_source_domain", "jost_step", "plane", "frequency_grid",
           "fft_padding", "jost_outer_radius"),
}
NOT_APPLICABLE_FACTORS = {
    ("GS", "left_domain"): "left endpoint is the regular centre of the de Sitter interior, not a numerical boundary",
    ("BS", "left_domain"): "left endpoint is the regular centre of the boson star, not a numerical boundary",
}
# Plan 0.1/0.2 controls carried no factor key; their names map one-to-one.
LEGACY_FACTOR_KEYS = {
    "BH_RIGHT_DOMAIN": "right_domain", "BH_LEFT_DOMAIN": "left_domain", "BH_HALF_TIMESTEP": "time_step",
    "BHGS_HALF_TIMESTEP": "time_step", "GS_RIGHT_DOMAIN": "right_domain", "GS_HALF_TIMESTEP": "time_step",
    "WH_RIGHT_DOMAIN": "right_domain", "WH_LEFT_DOMAIN": "left_domain", "WH_HALF_TIMESTEP": "time_step",
    "BS_RIGHT_DOMAIN": "right_domain", "BS_HALF_TIMESTEP": "time_step", "BS_BVP_TOL": "bvp_tolerance",
    "BS_DEFAULT_PROFILE_TO_20001": "background_profile_sampling", "BS_DOMAIN_FIXED_SPACING": "background_domain",
    "QH_JOST_STEP": "jost_step", "QH_PLANE_50": "plane", "QH_GRID2": "frequency_grid",
    "QH_PADDING2": "fft_padding", "QH_RADIUS300": "jost_outer_radius",
}
BS_SETTINGS = {"equilibrium_rmax": 50., "equilibrium_n": 1200, "equilibrium_tol": 2e-7}
QH_SETTINGS = {"surface": -40., "x_left": -250., "source_t_end": 650.,
               "source_x_left": -400., "scattering_step": .02, "scattering_x_right": 200.}
BS_SEQUENCE = (.18, .20, .22, .24, .26, FIDUCIAL_SIGMA_C, .28, .30)
BS_TABLE_I = {  # Macedo et al. arXiv:1307.4812v3 Table I (same in the v3 PDF and HTML), tilde (mu) units, as printed.
    "mini BS I": {"phi_tilde_c": .1916, "omega_s_over_mu": .853087, "mass_mu": .63300,
                  "radius_99_mu": 7.86149, "branch_statement": "maximum total mass; stable/unstable critical point"},
    "mini BS II": {"phi_tilde_c": .4101, "omega_s_over_mu": .773453, "mass_mu": .53421,
                   "radius_99_mu": 4.52825, "branch_statement": "maximum compactness; usually unstable branch"},
}
TABLE_HALF_UNIT = {"phi_tilde_c": 5e-5, "omega_s_over_mu": 5e-7, "mass_mu": 5e-6, "radius_99_mu": 5e-6}
PDE_LEVELS = (.05, .025, .0125)
PDE_T_END = 520.2
IMPORTED_SCRIPTS = ("scripts/qh_spectral_controls.py", "scripts/qh_causal_control.py")
EXTRA_RESUME_KEYS = ("plan_sha256", "imported_scripts")
# Two separately computed norm ratios of a bitwise-identical kernel may differ in
# the last bits (BLAS summation order); v0.1 demanded exact equality and failed
# at 8e-15 relative. Arrays are compared by SHA-256; the ratio by this tolerance.
NORM_RATIO_RELATIVE_TOLERANCE = 1e-12
QUANTIFIED = ("QUANTIFIED_TD_DOMINANT", "MULTI_COMPONENT_MEASURED")
POLICY = {
    "common_grid_dt": .07, "windows_M": WINDOWS, "window_masks": "inclusive sampled masks",
    "norm": "integral L2 by trapezoid on the common grid",
    "interpolation": "cubic spline on native samples, no extrapolation",
    "order_band": .5, "richardson_safety_factor": 1.25, "agreement_floor_relative": 1e-12,
    "control_budget_fraction": .1, "acceptance_windows": ["late", "full"],
    "qh_independent_late_gate": .01, "qh_negative_lag_screen": 1e-3, "qh_pde_expected_order": 2,
    "qh_hierarchy_identity_tolerance": 1e-12, "bs_profile_expected_order": 2,
    "gs_metric_continuity_tolerance": 1e-12, "wh_symmetry_tolerance": 1e-12,
    "wh_offset_tolerance": 1e-12, "nested_ratio_tolerance": 1e-12,
    "runtime_regression_relative_max": 1e-10,
    "runtime_regression_expected_failed_checks": ["boundary_causal"],
    "kernel_norm_ratio_relative_tolerance": 1e-12,
    "bs_table_numerical_allowance": {"mass_mu": 1e-6, "omega_s_over_mu": 1e-6,
                                     "radius_99_mu": "one dense radial sample spacing"},
    "threshold_provenance": ("existing pilot policy axial-td-pilot/0.2 and inherited notebook/adapter "
                             "gates; not calibrated physical thresholds"),
    "estimates_are_error_bounds": False,
    "estimate_keys_cleared_without_support": list(ESTIMATE_KEYS),
    "initial_data_audit_relative_tolerance": 1e-12,
    "areal_radius_relative_tolerance": AREAL_RADIUS_RELATIVE_TOLERANCE,
    "experiment_window_rule": ("windows and record length shift by (r*_s-55)+(r*_o-95): unit-speed propagation "
                               "in Schwarzschild r* for the barrier-reflected left-going pulse; a declared "
                               "convention, not a fitted alignment"),
    "control_coverage_required": True,
    "invalid_comparison_input": ("rejected: complex, nonfinite, bool, string or object data, malformed or "
                                 "nonmonotonic grids, shape mismatches, no common support, too few samples"),
}


# --------------------------------------------------------------------- experiments

def as_experiment(value):
    """An immutable Experiment from an Experiment or its dictionary form."""
    if isinstance(value, Experiment):
        return value
    if isinstance(value, Mapping):
        return Experiment(**value)
    raise TypeError("experiment must be an Experiment or its dictionary form")


def experiment_settings(experiment):
    """Adapter keyword values that carry the experiment's source, observer and width."""
    e = as_experiment(experiment)
    return {"source_global": e.source_global_rstar, "observer_global": e.observer_global_rstar, "sigma": e.sigma}


def window_shift(experiment):
    """Declared propagation shift of the windows relative to r*_s=55, r*_o=95 (POLICY)."""
    e = as_experiment(experiment)
    return (e.source_global_rstar-REFERENCE_SOURCE_RSTAR)+(e.observer_global_rstar-REFERENCE_OBSERVER_RSTAR)


def windows_for(experiment):
    shift = window_shift(experiment)
    return {w: [lo+shift, hi+shift] for w, (lo, hi) in WINDOWS.items()}


def record_t_end(experiment):
    return NUMERICS["t_end"]+window_shift(experiment)


def request_experiment(request):
    """The experiment of a BS/QH request; plan 0.1/0.2 requests predate the explicit field."""
    if "experiment" in request:
        return as_experiment(request["experiment"])
    s = request["settings"]
    return Experiment(source_global_rstar=s["source_global"], observer_global_rstar=s["observer_global"],
                      sigma=s["sigma"])


def _windows(entry):
    return {w: tuple(v) for w, v in entry.get("windows", WINDOWS).items()}


def validate_request_experiment(job):
    """Refuse a request whose numerical settings contradict its own experiment."""
    request = job.get("request", {})
    kind = job["kind"]
    if kind == "qh_control":
        request = request["base_request"]
    if kind not in ("bs", "qh", "qh_control", "qh_pde") or "experiment" not in request:
        return
    e = as_experiment(request["experiment"])
    s = request["settings"]
    if kind == "qh_pde":
        got = {"source_global": s.get("source_global"), "observer_global": s.get("observer"), "sigma": s.get("sigma")}
    else:
        got = {k: s.get(k) for k in ("source_global", "observer_global", "sigma")}
    if got != experiment_settings(e):
        raise ValueError(f"Request settings contradict its experiment: {got} != {experiment_settings(e)}")


# --------------------------------------------------------------------------- plan

def _td_seconds(spec):
    """Scheduling estimate from measured BH dx=0.1 (8.5 s for 650 M x 520 M)."""
    left = spec.x_left if spec.x_left is not None else {"WH": -400., "GS": -6., "BH": -250.}[spec.model]
    return 8.5*((spec.x_right-left)/650.)*(.1/spec.dx)**2*(.35/spec.cfl)*(spec.t_end/520.)+1.


def _bs_seconds(settings):
    return 8.5*(settings["x_right"]/650.)*(.1/settings["dx"])**2*(.35/settings["cfl"])+3.


def _qh_seconds(settings):
    bh = 8.5*((settings["x_right"]-settings["x_left"])/650.)*(.1/settings["dx"])**2*(.35/settings["cfl"])
    source = bh*((settings["x_right"]-settings["source_x_left"])/(settings["x_right"]-settings["x_left"]))*(
        settings["source_t_end"]/settings["t_end"])
    return bh+source+6.*(.02/settings["scattering_step"])


def _td_job(spec, role, point, purpose, **extra):
    return {"kind": "td", "role": role, "point": point, "spec": spec.to_dict(),
            "identity": spec.record_id, "parents": [], "purpose": purpose,
            "estimated_seconds": round(_td_seconds(spec), 1), **extra}


def _adapter_job(kind, request, role, point, purpose, seconds, parents=()):
    return {"kind": kind, "role": role, "point": point, "request": request,
            "identity": request_id(request), "parents": list(parents), "purpose": purpose,
            "estimated_seconds": round(seconds, 1)}


def _json_job(kind, request, role, point, purpose, seconds):
    return {"kind": kind, "role": role, "point": point, "request": request,
            "identity": "bsbg-"+digest(request)[:24], "parents": [], "purpose": purpose,
            "estimated_seconds": seconds}


def build_plan(experiment=None, points=None, block=None):
    """Return the complete immutable plan for one experiment and one point per model; no result.

    ``points`` (default POINTS) gives one parameter dictionary per model; a catalogue block
    (docs/METHOD.md) names itself with ``block``.

    ``experiment`` defaults to PRODUCTION_EXPERIMENT.  It is passed explicitly to every
    BH/GS/WH spec, BS/QH request, QH control and independent PDE; the frozen-replay job
    always uses FROZEN_REPLAY_EXPERIMENT.  Windows and record length follow the declared
    propagation shift of the experiment (POLICY["experiment_window_rule"]).
    """
    experiment = PRODUCTION_EXPERIMENT if experiment is None else as_experiment(experiment)
    points = {m: dict(POINTS[m]) for m in POINTS} if points is None else {m: dict(points[m]) for m in POINTS}
    if set(points) != set(POINTS) or (block is None) != (points == {m: dict(POINTS[m]) for m in POINTS}):
        raise ValueError("a non-default point set must be declared as a named catalogue block, and vice versa")
    exp_dict = asdict(experiment)
    t_end = record_t_end(experiment)
    windows = windows_for(experiment)
    numerics = {**NUMERICS, "t_end": t_end, **experiment_settings(experiment)}
    jobs, sequences, controls = {}, {}, {}

    def td_sequence(name, model, params, hs, role, point, x_left=None, cells=None):
        labels = []
        for k, (level, h) in enumerate(zip(LEVEL_NAMES, hs)):
            spec = RunSpec(model, params, dx=h, t_end=t_end, cfl=NUMERICS["cfl"], x_left=x_left,
                           x_right=NUMERICS["x_right"], sample_every=NUMERICS["sample_every"],
                           experiment=experiment)
            extra = {"expected_interior_cells": cells*2**k} if cells else {}
            label = f"{name}_{level}"
            jobs[label] = _td_job(spec, role, point, f"{name} refinement level {level}", **extra)
            labels.append(label)
        return labels

    gs_hs = refinement_spacings("GS", points["GS"])
    rt = load_frozen_runtime()
    _, _, _, gs_meta = rt["gravastar_grid"](points["GS"]["compactness"], cfg=rt["Config"](dx=gs_hs[0]), dx=gs_hs[0])
    gs_cells = int(gs_meta["shell_index"])
    bh = td_sequence("BH", "BH", {}, LEVELS, "reference", "BH", x_left=-250.)
    bhgs = td_sequence("BHGS", "BH", {}, gs_hs, "reference", "BH", x_left=-250.)
    gs = td_sequence("GS", "GS", points["GS"], gs_hs, "production", "GS", cells=gs_cells)
    wh = td_sequence("WH", "WH", points["WH"], LEVELS, "production", "WH")
    def sequence(model, labels, field, h, order, reference):
        return {"model": model, "labels": labels, "field": field, "h": h, "expected_order": order,
                "bh_reference": reference, "windows": windows}

    sequences["BH"] = sequence("BH", bh, "psi", "actual_dx", 4, None)
    sequences["BHGS"] = sequence("BH", bhgs, "psi", "actual_dx", 4, None)
    sequences["GS"] = sequence("GS", gs, "psi", "actual_dx", 2, "BHGS")
    sequences["WH"] = sequence("WH", wh, "psi", "actual_dx", 4, "BH")

    bs = []
    for level, h in zip(LEVEL_NAMES, LEVELS):
        settings = {**numerics, "dx": h, **BS_SETTINGS}
        request = {"model": "BS", "params": dict(points["BS"]), "settings": settings, "experiment": exp_dict}
        jobs[f"BS_{level}"] = _adapter_job("bs", request, "production", "BS",
            f"BS fresh equilibrium and TD level {level}", _bs_seconds(settings))
        bs.append(f"BS_{level}")
    sequences["BS"] = sequence("BS", bs, "psi", "actual_dx", 4, "BH")
    qh = []
    for level, h in zip(LEVEL_NAMES, LEVELS):
        settings = {**numerics, "dx": h, **QH_SETTINGS}
        request = {"model": "QH", "params": dict(points["QH"]), "settings": settings, "experiment": exp_dict}
        jobs[f"QH_{level}"] = _adapter_job("qh", request, "production", "QH",
            f"QH measured-source OWA response level {level}", _qh_seconds(settings))
        qh.append(f"QH_{level}")
    sequences["QH"] = sequence("QH", qh, "psi", "actual_dx", 4, "INTERNAL")
    sequences["QH_CORRECTION"] = sequence("QH", qh, "correction", "actual_dx", 4, None)

    def td_control(name, base_label, point, factor, **changes):
        base = RunSpec.from_dict(jobs[base_label]["spec"])
        variant = replace(base, **changes)
        label = f"{name}"
        # v0.2 fix: a control on the same mesh inherits the base's declared GS cell count.
        extra = ({"expected_interior_cells": jobs[base_label]["expected_interior_cells"]}
                 if "expected_interior_cells" in jobs[base_label] else {})
        jobs[label] = _td_job(variant, "control", point, f"control: {factor}", **extra)
        return label

    def add_control(name, point, base, variant, field, estimate, factor, factor_key):
        if factor_key not in SEQUENCE_FACTORS[sequences[estimate]["model"]]:
            raise ValueError(f"Undeclared control factor {factor_key} for {estimate}")
        controls[name] = {"point": point, "base": base, "variant": variant, "field": field,
                          "estimate_sequence": estimate, "factor": factor, "factor_key": factor_key,
                          "windows": windows}

    gs_left = -250.-round(50./gs_hs[-1])*gs_hs[-1]   # integer GS-spacing cells, as BH_LEFT_DOMAIN
    for name, base, seq, point, changes, factor, key in (
            ("BH_RIGHT_DOMAIN", "BH_fine", "BH", "BH", {"x_right": 450.}, "right boundary 400 -> 450",
             "right_domain"),
            ("BH_LEFT_DOMAIN", "BH_fine", "BH", "BH", {"x_left": -250.-round(50./LEVELS[-1])*LEVELS[-1]},
             "left boundary -250 -> -300 (integer cells)", "left_domain"),
            ("BH_HALF_TIMESTEP", "BH_fine", "BH", "BH", {"cfl": .175, "sample_every": 4},
             "CFL 0.35 -> 0.175 at fixed output cadence", "time_step"),
            ("BHGS_HALF_TIMESTEP", "BHGS_fine", "BHGS", "BH", {"cfl": .175, "sample_every": 4},
             "CFL 0.35 -> 0.175 at fixed output cadence", "time_step"),
            ("BHGS_RIGHT_DOMAIN", "BHGS_fine", "BHGS", "BH", {"x_right": 450.}, "right boundary 400 -> 450",
             "right_domain"),
            ("BHGS_LEFT_DOMAIN", "BHGS_fine", "BHGS", "BH", {"x_left": gs_left},
             "left boundary -250 -> about -300 (integer GS-spacing cells)", "left_domain"),
            ("GS_RIGHT_DOMAIN", "GS_fine", "GS", "GS", {"x_right": 450.}, "right boundary 400 -> 450",
             "right_domain"),
            ("GS_HALF_TIMESTEP", "GS_fine", "GS", "GS", {"cfl": .175, "sample_every": 4},
             "CFL 0.35 -> 0.175 at fixed output cadence", "time_step"),
            ("WH_RIGHT_DOMAIN", "WH_fine", "WH", "WH", {"x_right": 450.}, "right boundary 400 -> 450",
             "right_domain"),
            ("WH_LEFT_DOMAIN", "WH_fine", "WH", "WH", {"x_left": -450.}, "left boundary -400 -> -450",
             "left_domain"),
            ("WH_HALF_TIMESTEP", "WH_fine", "WH", "WH", {"cfl": .175, "sample_every": 4},
             "CFL 0.35 -> 0.175 at fixed output cadence", "time_step")):
        label = td_control(name, base, point, factor, **changes)
        add_control(name, point, base, label, "psi", seq, factor, key)

    def bs_variant(name, base_label, factor, seconds=None, **changes):
        base = jobs[base_label]["request"]
        request = {**base, "settings": {**base["settings"], **changes}}
        jobs[name] = _adapter_job("bs", request, "control", "BS", f"control: {factor}",
                                  seconds if seconds is not None else _bs_seconds(request["settings"]))
        return name

    bs_variant("BS_RIGHT_DOMAIN", "BS_fine", "right boundary 400 -> 450", x_right=450.)
    add_control("BS_RIGHT_DOMAIN", "BS", "BS_fine", "BS_RIGHT_DOMAIN", "psi", "BS", "TD right boundary 400 -> 450",
                "right_domain")
    bs_variant("BS_HALF_TIMESTEP", "BS_fine", "CFL 0.35 -> 0.175", cfl=.175, sample_every=4)
    add_control("BS_HALF_TIMESTEP", "BS", "BS_fine", "BS_HALF_TIMESTEP", "psi", "BS", "TD CFL 0.35 -> 0.175",
                "time_step")
    bs_variant("BS_BVP_TOL", "BS_medium", "BVP tolerance 2e-7 -> 2e-8", equilibrium_tol=2e-8)
    add_control("BS_BVP_TOL", "BS", "BS_medium", "BS_BVP_TOL", "psi", "BS", "BVP tolerance 2e-7 -> 2e-8",
                "bvp_tolerance")
    profile = []
    for samples in (5001, 10001, 20001):
        profile.append(bs_variant(f"BS_PROFILE_{samples}", "BS_medium",
                                  f"dense background samples {samples} at rmax=50", background_samples=samples))
    sequences["BS_PROFILE"] = sequence("BS", profile, "psi", "radial_spacing_mu", 2, None)
    add_control("BS_DEFAULT_PROFILE_TO_20001", "BS", "BS_medium", "BS_PROFILE_20001", "psi", "BS",
                "default 5000-sample dense profile -> nested 20001 samples", "background_profile_sampling")
    extended = 1e-5+6000*((50.-1e-5)/5000)  # as bs_background_controls.py: same mu spacing, 24001 vs 20001
    bs_variant("BS_DOMAIN_24001", "BS_medium", "rmax 50 -> 59.999998 at fixed fine radial spacing",
               equilibrium_rmax=extended, background_samples=24001)
    add_control("BS_DOMAIN_FIXED_SPACING", "BS", "BS_PROFILE_20001", "BS_DOMAIN_24001", "psi", "BS",
                "finite background radius 50 -> 59.999998 at fixed radial spacing", "background_domain")
    jobs["BS_BACKGROUND_SEQUENCE"] = _json_job("bs_sequence", {
        "model": "BS", "kind": "background_sequence",
        "sigma_c": sorted(set(BS_SEQUENCE) | {points["BS"]["sigma_c"]}),
        **({} if block is None else {"target_sigma_c": points["BS"]["sigma_c"]}),
        "settings": {**BS_SETTINGS, "background_samples": 5000}}, "control", "BS",
        "fresh equilibria across sigma_c: side of the mass maximum", 20.)
    jobs["BS_TABLE_BENCHMARK"] = _json_job("bs_benchmark", {
        "model": "BS", "kind": "literature_background_benchmark", "table": BS_TABLE_I,
        "phi_tilde_offsets": [-5e-5, 0., 5e-5], "settings": {**BS_SETTINGS, "background_samples": 5000},
        "source": "Macedo et al. arXiv:1307.4812v3 Table I"}, "diagnostic", "BS",
        "frozen BVP solver against published mini BS I/II backgrounds (II is not a library point)", 20.)

    medium = jobs["QH_medium"]
    for name, control, seconds, factor, key in (
            ("QH_JOST_STEP", {"kind": "fresh", "scattering_step": .01}, 32., "Jost step 0.02 -> 0.01", "jost_step"),
            ("QH_PLANE_50", {"kind": "fresh", "surface": -50.}, 28., "computational plane -40 -> -50, source remeasured",
             "plane"),
            ("QH_GRID2", {"kind": "scattering", "grid_factor": 2}, 8., "nested frequency grid x2", "frequency_grid"),
            ("QH_PADDING2", {"kind": "reconstruct", "padding_factor": 2}, 4., "inverse-FFT padding x2", "fft_padding"),
            ("QH_RADIUS300", {"kind": "scattering", "scattering_x_right": 300.}, 8., "Jost outer radius 200 -> 300",
             "jost_outer_radius")):
        request = {"model": "QH", "kind": "spectral_control", "base_request_id": medium["identity"],
                   "base_request": medium["request"], "control": control}
        jobs[name] = _adapter_job("qh_control", request, "control", "QH", f"control: {factor}",
                                  seconds, parents=["QH_medium"])
        for field, seq in (("psi", "QH"), ("correction", "QH_CORRECTION")):
            add_control(name+("" if field == "psi" else "_CORRECTION"), "QH", "QH_medium", name,
                        field, seq, factor, key)

    def qh_variant(name, base_label, factor, key, **changes):
        """Fresh QH evolution (BH stage, measured source, scattering, reconstruction) with one changed setting."""
        base = jobs[base_label]["request"]
        request = {**base, "settings": {**base["settings"], **changes}}
        jobs[name] = _adapter_job("qh", request, "control", "QH", f"control: {factor}",
                                  _qh_seconds(request["settings"]))
        for field, seq in (("psi", "QH"), ("correction", "QH_CORRECTION")):
            add_control(name+("" if field == "psi" else "_CORRECTION"), "QH", base_label, name,
                        field, seq, factor, key)

    # v0.3 (audit finding C).  The halved time step runs at the medium level: at dx=0.05 it would
    # exceed one 180 s execution slice; RK4 time error at dx=0.1 is about 16x its dx=0.05 value,
    # so comparing it with the fine-level refinement estimate is conservative.
    qh_variant("QH_HALF_TIMESTEP", "QH_medium", "CFL 0.35 -> 0.175 at fixed output cadence (BH stage and "
               "measured-source run)", "time_step", cfl=.175, sample_every=4)
    qh_variant("QH_RIGHT_DOMAIN", "QH_fine", "TD right boundary 400 -> 450 (BH stage and measured-source run)",
               "right_domain", x_right=450.)
    qh_variant("QH_LEFT_DOMAIN", "QH_fine", "TD left boundaries: BH stage -250 -> -300, measured-source run "
               "-400 -> -450", "left_and_source_domain", x_left=-300., source_x_left=-450.)
    pde = []
    for h in PDE_LEVELS:
        settings = {"dx": h, "surface": QH_SETTINGS["surface"], "observer": experiment.observer_global_rstar,
                    "t_end": PDE_T_END+window_shift(experiment), "source_global": experiment.source_global_rstar,
                    "sigma": experiment.sigma, "x_right": NUMERICS["x_right"]}
        label = f"QH_PDE_{h:g}"
        request = {"model": "QH", "kind": "independent_pde", "params": dict(points["QH"]), "settings": settings,
                   "experiment": exp_dict}
        jobs[label] = _adapter_job("qh_pde", request, "control", "QH",
            "independent paired diamond PDE with causal wall history", 3.*(.05/h)**2, parents=[])
        pde.append(label)
        clabel = f"QH_CAUSAL_{h:g}"
        crequest = {"model": "QH", "kind": "causal_wall_diagnostic",
                    "base_request_id": jobs["QH_fine"]["identity"], "pde_request_id": request_id(request),
                    "wall": {"h": h, "t_end": PDE_T_END, "surface": QH_SETTINGS["surface"]},
                    "params": dict(points["QH"])}
        jobs[clabel] = _adapter_job("qh_causal", crequest, "diagnostic", "QH",
            "DIAGNOSTIC causal-kernel FD wall matched to the PDE; not the production law",
            8.*(.05/h), parents=["QH_fine", label])
    sequences["QH_PDE"] = sequence("QH", pde, "psi", "pde_h", 2, None)
    replay = RunSpec("BH", dx=.1, t_end=520., x_left=-140., experiment=FROZEN_REPLAY_EXPERIMENT)
    jobs["BH_RUNTIME_REPLAY"] = _td_job(replay, "regression", "BH",
        "runtime regression against the frozen controlled BH record (not a library point)")

    points = {
        "GS": {"params": points["GS"], "fiducial": FIDUCIALS["GS"], "sequence": "GS", "bh_sequence": "BHGS",
               "controls": ["GS_RIGHT_DOMAIN", "GS_HALF_TIMESTEP", "BHGS_HALF_TIMESTEP", "BHGS_RIGHT_DOMAIN",
                            "BHGS_LEFT_DOMAIN"],
               "model": "Zero-surface-energy thin-shell gravastar, axial sector (Pani et al. arXiv:0909.0287)"},
        "WH": {"params": points["WH"], "fiducial": FIDUCIALS["WH"], "sequence": "WH", "bh_sequence": "BH",
               "controls": ["WH_RIGHT_DOMAIN", "WH_LEFT_DOMAIN", "WH_HALF_TIMESTEP",
                            "BH_RIGHT_DOMAIN", "BH_LEFT_DOMAIN", "BH_HALF_TIMESTEP"],
               "model": "Thin-shell Schwarzschild wormhole with axial delta throat (Yang-Xia-Miao EPJC 85, 742); not Damour-Solodukhin"},
        "BS": {"params": points["BS"], "fiducial": FIDUCIALS["BS"], "sequence": "BS", "bh_sequence": "BH",
               "controls": ["BS_RIGHT_DOMAIN", "BS_HALF_TIMESTEP", "BS_BVP_TOL", "BS_DEFAULT_PROFILE_TO_20001",
                            "BS_DOMAIN_FIXED_SPACING", "BH_RIGHT_DOMAIN", "BH_LEFT_DOMAIN", "BH_HALF_TIMESTEP"],
               "model": "Mini boson star, fixed-background decoupled axial sector (Macedo et al. arXiv:1307.4812); not polar"},
        "QH": {"params": points["QH"], "fiducial": FIDUCIALS["QH"], "sequence": "QH",
               "correction_sequence": "QH_CORRECTION", "bh_sequence": None,
               "controls": [c for c in controls if c.startswith("QH_")],
               "model": "Schwarzschild axial with OWA effective boundary reflectivity (arXiv:1905.00464); not a spacetime"},
    }
    plan = {"plan_version": PLAN_VERSION if block is None else BLOCK_PLAN_VERSION,
            "document": "docs/METHOD.md" if block is None else "docs/METHOD.md",
            **({} if block is None else {"block": block}),
            "scope": ("one off-fiducial point per model; three refinement levels, matched BH references, "
                      "separate controls; not a sweep, validated library, atlas or detectability claim"),
            "experiment": exp_dict, "experiment_fingerprint": experiment.fingerprint,
            "frozen_replay_experiment": asdict(FROZEN_REPLAY_EXPERIMENT),
            "common_experiment": {"M_ADM": 1., "ell": 2, "polarisation": "axial", "field": "Q_RW",
                "time_convention": "exp(-i omega t)", "initial_data": "Psi=exp(-(x-xs)^2/(2 sigma^2)), Pi=D1_4(Psi)",
                "sigma": experiment.sigma, "source_schwarzschild_rstar": experiment.source_global_rstar,
                "observer_schwarzschild_rstar": experiment.observer_global_rstar,
                "areal_radius_rule": "areal radii fixed by Schwarzschild r*; BS maps them through its equilibrium",
                "t_end": t_end, "sample_every": NUMERICS["sample_every"], "cfl": NUMERICS["cfl"],
                "windows_M": windows, "window_shift_M": window_shift(experiment)},
            "points": points, "jobs": jobs, "sequences": sequences, "controls": controls, "policy": POLICY,
            "gs_expected_interior_cells": [gs_cells, 2*gs_cells, 4*gs_cells],
            "imported_scripts": list(IMPORTED_SCRIPTS)}
    return plan


def plan_sha256(plan):
    return digest(plan)


# --------------------------------------------------------------------- provenance

def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported_script_hashes(repo=REPO):
    return {name: _sha256_file(Path(repo)/name) for name in IMPORTED_SCRIPTS}


def _import_script(name, expected_sha256, repo=REPO):
    """Load an existing diagnostic script by path after checking its hash."""
    path = Path(repo)/name
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"Imported diagnostic script changed: {name}")
    if str(Path(repo)) not in sys.path:
        sys.path.insert(0, str(Path(repo)))
    module_name = "_offfid_"+Path(name).stem+"_"+expected_sha256[:12]
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def make_provenance(plan, repo=REPO):
    provenance = runtime_provenance(repo)
    provenance.update(plan_sha256=plan_sha256(plan), imported_scripts=imported_script_hashes(repo),
                      runner="eco_library.offfiducial", plan_version=PLAN_VERSION,
                      cpu_count=os.cpu_count(), machine=platform.machine(),
                      threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                                "MKL_NUM_THREADS")},
                      python_packages=sorted(f"{d.metadata['Name']}=={d.version}"
                                             for d in importlib.metadata.distributions()
                                             if d.metadata["Name"]))
    return provenance


def check_provenance(stored, expected):
    """Exact resume compatibility: inherited keys plus plan and imported scripts."""
    _check_resume_provenance(stored, expected)
    for key in EXTRA_RESUME_KEYS:
        if key not in stored or key not in expected:
            raise ValueError(f"Missing resume provenance: {key}")
        if stored[key] != expected[key]:
            raise ValueError(f"Resume provenance mismatch: {key}")


# ------------------------------------------------------------------ job storage

def output_path(root, job):
    root = Path(root)
    if job["kind"] in ("bs_sequence", "bs_benchmark"):
        return root/"background"/(job["identity"]+".json")
    return root/"records"/(job["identity"]+".h5")


def _payload_digest(data):
    return digest({k: v for k, v in data.items() if k != "payload_sha256"})


def load_job(root, job, provenance, mode="resume"):
    """Checked readback; raises on identity, provenance, checksum or schema mismatch.

    ``mode="resume"`` (default) is scientific reuse inside one run: the stored producer
    provenance must equal ``provenance`` on every resume key, the plan hash and the imported
    scripts.  ``mode="historical"`` analyses an archive produced elsewhere or earlier: identity,
    schema and every checksum are still enforced, the producer provenance is returned exactly as
    stored, and no compatibility with the analyzer is claimed.
    """
    if mode not in LOAD_MODES:
        raise ValueError(f"Unknown load mode {mode!r}; expected one of {LOAD_MODES}")
    resume = mode == "resume"
    if resume and provenance is None:
        raise ValueError("Resume readback requires the current runtime provenance")
    expected = provenance if resume else None
    path = output_path(root, job)
    if job["kind"] == "td":
        spec = RunSpec.from_dict(job["spec"])
        if spec.record_id != job["identity"]:
            raise ValueError("Plan record identity mismatch")
        record = read_record(path, expected_spec=spec, expected_provenance=expected)
        if resume:
            check_provenance(record["provenance"], provenance)
        return {"kind": "td", "spec": spec, "t": record["t"], "psi": record["psi"], "x": record["x"],
                "r": record["r"], "V": record["V"], "metadata": record["metadata"],
                "checks": record["checks"], "provenance": record["provenance"]}
    if job["kind"] in ("bs_sequence", "bs_benchmark"):
        data = json.loads(path.read_text(encoding="utf8"))
        canonical_json(data)
        if data.get("payload_sha256") != _payload_digest(data):
            raise ValueError("Background payload checksum mismatch")
        if data["request"] != job["request"] or data["identity"] != job["identity"]:
            raise ValueError("Background request identity mismatch")
        if resume:
            check_provenance(data["provenance"], provenance)
        return {"kind": job["kind"], **data}
    if request_id(job["request"]) != job["identity"]:
        raise ValueError("Plan request identity mismatch")
    archive = read_archive(path, expected_request=job["request"], expected_provenance=expected)
    if resume:
        check_provenance(archive["provenance"], provenance)
    result = archive["result"]
    return {"kind": job["kind"], "t": result["t"], "psi": result["psi"], "x": result["x"],
            "r": result["r"], "V": result["V"], "metadata": result["metadata"],
            "checks": result["checks"], "result": result, "provenance": archive["provenance"]}


# ------------------------------------------------------------- model-level checks

def td_model_checks(job, result):
    """Structural off-fiducial checks recorded inside the archive (pass/fail)."""
    spec = RunSpec.from_dict(job["spec"])
    meta, x, V = result["metadata"], np.asarray(result["x"]), np.asarray(result["V"])
    checks = {}
    if spec.model == "GS":
        a = float(meta["a"])
        continuity = abs((1-2*a*a/a**3)-(1-2/a))
        checks["gs_metric_continuity"] = {"passed": bool(continuity < POLICY["gs_metric_continuity_tolerance"]),
            "abs_f_in_minus_f_out_at_shell": continuity, "tolerance": POLICY["gs_metric_continuity_tolerance"],
            "scope": "Sigma=0 background identity f_in(a)=f_out(a), Pani Eq. (2.3) with M=M_v"}
        expected = job.get("expected_interior_cells")
        ratio_error = abs(meta["actual_dx"]/spec.dx-1)
        checks["gs_nested_cells"] = {"passed": bool(expected is not None and int(meta["shell_index"]) == expected
                                                   and ratio_error < POLICY["nested_ratio_tolerance"]),
            "shell_index": int(meta["shell_index"]), "expected": expected,
            "actual_over_requested_dx_minus_1": ratio_error}
    elif spec.model == "WH":
        r0 = spec.params["r0_over_M"]
        e = spec.experiment  # the record's own specification, never module state (finding A)
        i0 = int(np.argmin(np.abs(x)))
        n = min(i0, len(x)-1-i0)
        k = np.arange(1, n+1)
        mirror = float(np.max(np.abs(V[i0+k]-V[i0-k]))) if n > 0 else None
        lam_error = abs(meta["throat_Lambda"]-2*(r0-1)/r0**2)
        rstar0 = float(load_frozen_runtime()["schwarzschild_rstar"](r0))
        offset_error = max(abs(meta["rstar0"]-rstar0),
                           abs(meta["source_native"]-(e.source_global_rstar-rstar0)),
                           abs(meta["observer_native"]-(e.observer_global_rstar-rstar0)))
        checks["wh_throat_node"] = {"passed": bool(x[i0] == 0.), "throat_coordinate": float(x[i0])}
        checks["wh_lambda"] = {"passed": bool(lam_error <= 1e-15), "abs_error": lam_error,
                               "formula": "Lambda=2(r0-M)/r0^2 (Yang-Xia-Miao Eqs. 19/28/37 in M units)"}
        checks["wh_symmetry"] = {"passed": bool(mirror is not None and mirror < POLICY["wh_symmetry_tolerance"]),
                                 "max_abs_mirror_potential_difference": mirror, "mirrored_nodes": n}
        checks["wh_coordinate_offset"] = {"passed": bool(offset_error < POLICY["wh_offset_tolerance"]),
                                          "max_abs_error": offset_error,
                                          "rule": "native = global Schwarzschild r* - r*(r0), global from the record spec"}
    return checks


def experiment_consistency(experiment, settings, metadata):
    """The adapter's recorded source, observer, width, areal radii and extraction match the experiment."""
    e = as_experiment(experiment)
    expected = experiment_settings(e)
    mismatches = {}
    if settings is not None:
        for key, value in expected.items():
            if settings.get(key) != value:
                mismatches["settings."+key] = [settings.get(key), value]
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches["metadata."+key] = [metadata.get(key), value]
    radii = load_frozen_runtime()["r_from_schwarzschild_rstar"](
        np.array([e.source_global_rstar, e.observer_global_rstar]))
    for key, radius in zip(("source_areal_radius_over_M", "observer_areal_radius_over_M"), radii):
        value = metadata.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                or abs(value-float(radius)) > AREAL_RADIUS_RELATIVE_TOLERANCE*max(1., float(radius))):
            mismatches["metadata."+key] = [value, float(radius)]
    native = metadata.get("observer_native")
    points = metadata.get("extraction_points") or {}
    if native is None or metadata.get("x_extract") != native or points.get("observer") != native:
        mismatches["extraction"] = {"observer_native": native, "x_extract": metadata.get("x_extract"),
                                    "extraction_points.observer": points.get("observer")}
    return {"passed": not mismatches, "mismatches": mismatches, "experiment_fingerprint": e.fingerprint,
            "scope": ("requested and recorded source, observer and width; areal radii r(r*) from the frozen "
                      "Schwarzschild map; cubic extraction at the native observer")}


def initial_data_audit(x, recorded_t0, source_native, observer_native, sigma, runtime=None):
    """Recompute the recorded t=0 observer value from the frozen initial-data and extraction rules.

    The frozen ``initial_characteristics`` builds Psi on the record's own grid for the given
    source and width; the same four-point Lagrange stencil as ``evolve_wave`` extracts it at
    the observer.  Because exp(-(x_o-x_s)^2/(2 sigma^2)) changes by orders of magnitude when
    the width, source or observer change, agreement identifies the initial data actually
    evolved.  Not a waveform accuracy statement.
    """
    rt = load_frozen_runtime() if runtime is None else runtime
    tolerance = POLICY["initial_data_audit_relative_tolerance"]
    out = {"source_native": float(source_native), "observer_native": float(observer_native),
           "sigma": float(sigma), "tolerance": tolerance, "expected_psi_t0": None, "relative_difference": None,
           "rule": "frozen initial_characteristics on the stored grid + evolve_wave cubic Lagrange extraction"}
    recorded = float(recorded_t0)
    out["recorded_psi_t0"] = recorded if math.isfinite(recorded) else None
    x = np.asarray(x, dtype=float)
    if (x.ndim != 1 or len(x) < 8 or not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0)
            or not x[1] <= observer_native <= x[-2] or not math.isfinite(recorded)):
        return {**out, "passed": False, "reason": "record grid, observer or t=0 value unusable for the audit"}
    psi0 = rt["initial_characteristics"](x, float(source_native), float(sigma))[0]
    i = int(np.searchsorted(x, observer_native))
    start = min(max(i-2, 0), len(x)-4)
    inds = np.arange(start, start+4)
    nodes, weights = x[inds], np.ones(4)
    for j in range(4):
        for k in range(4):
            if j != k:
                weights[j] *= (observer_native-nodes[k])/(nodes[j]-nodes[k])
    expected = float(weights@psi0[inds])
    if not (math.isfinite(expected) and expected > 0):
        return {**out, "passed": False, "reason": "expected Gaussian tail underflows at the observer"}
    relative = abs(recorded-expected)/expected
    return {**out, "passed": bool(relative <= tolerance), "expected_psi_t0": expected,
            "relative_difference": relative if math.isfinite(relative) else None}


def experiment_checks(job, result):
    """Experiment consistency and initial-data audits for a waveform record (pass/fail)."""
    kind = job["kind"]
    if kind == "td":
        spec = RunSpec.from_dict(job["spec"])
        e, settings, model = spec.experiment, None, spec.model
    else:
        request = job["request"]
        e = request_experiment(request["base_request"] if kind == "qh_control" else request)
        settings = (request["base_request"] if kind == "qh_control" else request)["settings"]
        model = "BS" if kind == "bs" else "QH"
    meta = result["metadata"]
    checks = {"experiment_consistency": experiment_consistency(e, settings, meta)}
    if kind == "qh_control":
        control = job["request"]["control"]
        if control.get("kind") != "fresh":
            return checks  # BH stage and measured source reused from the base archive, audited there
        settings = {**settings, **{k: v for k, v in control.items() if k != "kind"}}
    rt = load_frozen_runtime()
    if model == "WH":
        offset = float(rt["schwarzschild_rstar"](spec.params["r0_over_M"]))
        source, observer = e.source_global_rstar-offset, e.observer_global_rstar-offset
    elif model == "BS":
        # BS native positions follow from the equilibrium tortoise map; the adapter's own
        # geometry check and the areal-radius comparison above tie them to the experiment.
        source, observer = meta.get("source_native"), meta.get("observer_native")
    else:
        source, observer = e.source_global_rstar, e.observer_global_rstar
    stage = result.get("bh_reference", result) if model == "QH" else result
    if source is None or observer is None:
        checks["initial_data_audit"] = {"passed": False, "reason": "native positions missing", "tolerance":
                                        POLICY["initial_data_audit_relative_tolerance"], "relative_difference": None}
    else:
        checks["initial_data_audit"] = initial_data_audit(stage["x"], np.asarray(stage["psi"])[0], source,
                                                          observer, e.sigma, rt)
    if model == "QH":
        s = settings
        xs = np.arange(s["source_x_left"], s["x_right"]+.5*s["dx"], s["dx"])  # the adapter's monitor grid
        checks["initial_data_audit_source_monitor"] = initial_data_audit(
            xs, np.asarray(result["source"]["native_field"])[0], e.source_global_rstar, s["surface"], e.sigma, rt)
    return checks


def pde_experiment_consistency(request, result):
    e = as_experiment(request["experiment"]) if "experiment" in request else None
    s = request["settings"]
    if e is None:
        return {"passed": True, "scope": "plan 0.1/0.2 request without an explicit experiment"}
    got = {"source_global": s["source_global"], "observer_global": s["observer"], "sigma": s["sigma"]}
    ok = got == experiment_settings(e) and result["metadata"].get("observer") == s["observer"]
    return {"passed": bool(ok), "settings": got, "expected": experiment_settings(e),
            "scope": "independent PDE driven by the requested experiment"}


def qh_model_checks(params, result):
    """Production uses the analytic OWA law; hierarchy is a pure phase factor."""
    omega = np.asarray(result["response"]["omega"])
    stored = np.asarray(result["response"]["R_horizon_global"])
    fresh = reflection(omega, **params)
    reference = reflection(omega, EplM=FIDUCIALS["QH"]["EplM"], gamma=params["gamma"])
    phase = np.exp(8j*omega*math.log(params["EplM"]/FIDUCIALS["QH"]["EplM"]))
    identity = float(np.max(np.abs(stored-reference*phase)))
    return {
        "qh_analytic_production_law": {"passed": bool(np.array_equal(stored, fresh)),
            "scope": "stored R_horizon_global is bitwise the analytic OWA coefficient at the requested parameters"},
        "qh_hierarchy_identity": {"passed": bool(identity <= POLICY["qh_hierarchy_identity_tolerance"]),
            "max_abs": identity, "tolerance": POLICY["qh_hierarchy_identity_tolerance"],
            "identity": "R(EplM)=R(1e8)*exp(8 i omega ln(EplM/1e8)) for kappa=1/4, gamma fixed"},
    }


# ---------------------------------------------------------------- job execution

def _ratio_difference(a, b):
    scale = max(abs(a), abs(b))
    return abs(a-b)/scale if scale > 0 else 0.


def _same_ratio(a, b):
    return _ratio_difference(a, b) <= NORM_RATIO_RELATIVE_TOLERANCE


def _parent(root, plan_jobs, label, provenance):
    return load_job(root, plan_jobs[label], provenance)


def compute_job(label, job, root, provenance, plan_jobs):
    start = time.perf_counter()
    kind = job["kind"]
    validate_request_experiment(job)  # refuse contradictory requests before any solve
    if kind == "td":
        spec = RunSpec.from_dict(job["spec"])
        e = spec.experiment
        result = run_td(spec.model, spec.params, dx=spec.dx, t_end=spec.t_end, cfl=spec.cfl,
                        source_global=e.source_global_rstar, observer_global=e.observer_global_rstar,
                        sigma=e.sigma, x_right=spec.x_right, x_left=spec.x_left,
                        sample_every=spec.sample_every)
        result["checks"].update(_safe_checks(td_model_checks, job, result))
        result["checks"].update(_safe_checks(experiment_checks, job, result))
    elif kind == "bs":
        request = job["request"]
        result = run_boson_star(request["params"], **request["settings"])
        fresh = result["metadata"].get("equilibrium_solve") == "FRESH_BVP_CONTINUATION_NO_SAVED_PROFILE"
        result["checks"]["bs_fresh_equilibrium"] = {
            "passed": bool(fresh and result["metadata"]["sigma_c"] == request["params"]["sigma_c"]),
            "equilibrium_solve": result["metadata"].get("equilibrium_solve")}
        result["checks"].update(_safe_checks(experiment_checks, job, result))
    elif kind == "qh":
        request = job["request"]
        result = run_quantum_horizon(request["params"], **request["settings"])
        result["checks"].update(qh_model_checks(request["params"], result))
        result["checks"].update(_safe_checks(experiment_checks, job, result))
    elif kind == "qh_control":
        request = job["request"]
        base = _parent(root, plan_jobs, job["parents"][0], provenance)
        spectral = _import_script(IMPORTED_SCRIPTS[0], provenance["imported_scripts"][IMPORTED_SCRIPTS[0]])
        result = spectral.make_control(base["result"], request["base_request"], request["control"])
        base_file = output_path(root, plan_jobs[job["parents"][0]])
        result["metadata"].update(diagnostic_control=request["control"], base_label=job["parents"][0],
                                  base_archive_sha256=_sha256_file(base_file),
                                  control_input_policy="measured source/scattering of the base archive; "
                                                       "fresh evolution for 'fresh' controls")
        result["checks"].update(qh_model_checks(request["base_request"]["params"], result))
        result["checks"].update(_safe_checks(experiment_checks, job, result))
    elif kind == "qh_pde":
        request = job["request"]
        s, p = request["settings"], request["params"]
        direct = independent_td(dx=s["dx"], surface=s["surface"], observer=s["observer"], t_end=s["t_end"],
                                source_global=s["source_global"], sigma=s["sigma"], x_right=s["x_right"], **p)
        causal = _import_script(IMPORTED_SCRIPTS[1], provenance["imported_scripts"][IMPORTED_SCRIPTS[1]])
        wall = causal.wall_kernel(s["dx"], t_end=s["t_end"], surface=s["surface"], **p)
        rt = _qh_runtime()
        nx = int(round((s["x_right"]-s["surface"])/s["dx"]))+1
        x = s["surface"]+s["dx"]*np.arange(nx)
        r = rt["r_from_schwarzschild_rstar"](x)
        V = rt["rw_potential"](r, 2, 1.)
        margin = direct["right_boundary_return_M"]-float(direct["time"][-1])
        result = {"t": direct["time"], "psi": direct["correction"], "x": x, "r": r, "V": V,
                  "kernel_causal_prefix": wall["impulse"],
                  "metadata": {k: v for k, v in direct.items() if k not in ("time", "correction")},
                  "checks": {
                      "finite": {"passed": bool(np.isfinite(direct["correction"]).all())},
                      "negative_lag_screen": {"passed": bool(direct["negative_lag_kernel_L2_fraction"]
                                                              < POLICY["qh_negative_lag_screen"]),
                          "value": direct["negative_lag_kernel_L2_fraction"], "tolerance": POLICY["qh_negative_lag_screen"],
                          "scope": "screen on discarded kernel norm; not a waveform error bound"},
                      "boundary_causal": {"passed": bool(margin > 0), "margin_M": margin},
                      "kernel_identity": {"passed": bool(wall["nfft"] == direct["kernel_nfft"] and _same_ratio(
                              wall["negative_lag_L2_fraction"], direct["negative_lag_kernel_L2_fraction"])),
                          "relative_difference_of_negative_lag_ratio": _ratio_difference(
                              wall["negative_lag_L2_fraction"], direct["negative_lag_kernel_L2_fraction"]),
                          "tolerance": NORM_RATIO_RELATIVE_TOLERANCE,
                          "scope": "stored causal prefix is the PDE's own kernel construction"}}}
        result["checks"]["experiment_consistency"] = pde_experiment_consistency(request, result)
        result["metadata"].update(full_kernel_sha256=wall["full_kernel_sha256"], kernel_period_M=wall["period_M"],
                                  omitted_positive_lag_L2_fraction=wall["omitted_positive_lag_L2_fraction"],
                                  grid="x=surface+h*arange(nx), x[0] is the wall plane",
                                  numerics_evidence="UNQUANTIFIED",
                                  model_scope="independent second-order PDE with causal discrete wall; diagnostic")
    elif kind == "qh_causal":
        request = job["request"]
        fine = _parent(root, plan_jobs, job["parents"][0], provenance)
        pde = _parent(root, plan_jobs, job["parents"][1], provenance)
        causal = _import_script(IMPORTED_SCRIPTS[1], provenance["imported_scripts"][IMPORTED_SCRIPTS[1]])
        w = request["wall"]
        wall = causal.wall_kernel(w["h"], t_end=w["t_end"], surface=w["surface"], **request["params"])
        source, scattering = fine["result"]["source"], fine["result"]["scattering"]
        response = causal.response_with_wall(source, fine["t"], scattering, wall,
                                             nfft=int(fine["result"]["response"]["nfft"]))
        same_analytic = bool(np.array_equal(response["analytic_OWA_correction"],
                                            fine["result"]["response"]["correction"]))
        result = {"t": fine["t"], "psi": response["correction"], "x": fine["x"], "r": fine["r"], "V": fine["V"],
                  "analytic_OWA_correction": response["analytic_OWA_correction"],
                  "wall_omega": response["wall_omega"], "R_local_causal": response["R_local_causal"],
                  "K_causal": response["K"], "kernel_causal_prefix": wall["impulse"],
                  "metadata": {"label": "DIAGNOSTIC_CAUSAL_KERNEL_FD_NOT_PRODUCTION_LAW", "wall": {
                                   k: v for k, v in wall.items() if k != "impulse"},
                               "coefficient_action": response["coefficient_action"],
                               "frequency_support": response["frequency_support"],
                               "DC_policy": response["DC_policy"], "base_label": job["parents"][0],
                               "base_archive_sha256": _sha256_file(output_path(root, plan_jobs[job["parents"][0]])),
                               "pde_label": job["parents"][1],
                               "numerics_evidence": "UNQUANTIFIED"},
                  "checks": {"finite": {"passed": bool(np.isfinite(response["correction"]).all())},
                             "reproduces_production_correction": {"passed": same_analytic,
                                 "scope": "analytic-law reconstruction equals the archived production correction bitwise"},
                             "kernel_matches_pde": {"passed": bool(
                                 wall["full_kernel_sha256"] == pde["metadata"]["full_kernel_sha256"]
                                 and _same_ratio(wall["negative_lag_L2_fraction"],
                                                 pde["metadata"]["negative_lag_kernel_L2_fraction"])),
                                 "full_kernel_sha256_equal": wall["full_kernel_sha256"] == pde["metadata"]["full_kernel_sha256"],
                                 "relative_difference_of_negative_lag_ratio": _ratio_difference(
                                     wall["negative_lag_L2_fraction"], pde["metadata"]["negative_lag_kernel_L2_fraction"]),
                                 "tolerance": NORM_RATIO_RELATIVE_TOLERANCE,
                                 "scope": "identical discrete wall in both calculations"}}}
    elif kind == "bs_sequence":
        rows = []
        for sc in job["request"]["sigma_c"]:
            try:
                bg = solve_equilibrium({"sigma_c": sc}, **{k: v for k, v in job["request"]["settings"].items()})
                m = bg["metadata"]
                rows.append({"sigma_c": sc, "state": "SOLVED", "mass_mu": m["raw_mass_mu_units"],
                             "omega_s_over_mu": m["omega_s_over_mu"], "radius_99_mu": m["radius_99_mu_units"],
                             "R99_over_M": m["R99_over_M"], "EKG_residual": m["EKG_residual"],
                             "checks": {k: v["passed"] for k, v in bg["checks"].items()}})
            except Exception as exc:  # retained, never replaced by an invented background
                rows.append({"sigma_c": sc, "state": "FAILED", "exception": type(exc).__name__, "message": str(exc)})
        target = job["request"].get("target_sigma_c", POINTS["BS"]["sigma_c"])  # plan 0.1-0.3: the pilot point
        result = {"rows": rows, "turning_point": turning_point_side(rows, target)}
    elif kind == "bs_benchmark":
        result = {"configurations": {}}
        settings = job["request"]["settings"]
        rt = load_boson_runtime(background_samples=settings["background_samples"])
        for name, table in job["request"]["table"].items():
            rows = []
            for offset in job["request"]["phi_tilde_offsets"]:
                phi = table["phi_tilde_c"]+offset
                try:
                    bg = rt["solve_mini_boson_star"](sigma_c=math.sqrt(2.)*phi, rmax=settings["equilibrium_rmax"],
                                                    n=settings["equilibrium_n"], tol=settings["equilibrium_tol"])
                    _, meta, checks = _background_diagnostics(bg, settings["equilibrium_tol"])
                    rows.append({"phi_tilde_c": phi, "state": "SOLVED", "mass_mu": float(bg.mass),
                                 "omega_s_over_mu": float(bg.omega_s), "radius_99_mu": float(bg.radius_99),
                                 "EKG_residual": float(bg.residual),
                                 "radial_spacing_mu": float((bg.r[-1]-bg.r[0])/(len(bg.r)-1)),
                                 "checks": {k: v["passed"] for k, v in checks.items()}})
                except Exception as exc:
                    rows.append({"phi_tilde_c": phi, "state": "FAILED", "exception": type(exc).__name__,
                                 "message": str(exc)})
            result["configurations"][name] = {"rows": rows, "comparison": table_comparison(rows, table)}
    else:
        raise ValueError(f"Unknown job kind {kind}")
    if kind in ("bs_sequence", "bs_benchmark"):
        result["wall_seconds"] = time.perf_counter()-start
    else:
        result["metadata"]["wall_seconds"] = time.perf_counter()-start
    return result


def _safe_checks(function, job, result):
    """A structural check that cannot be evaluated fails closed instead of aborting the job."""
    try:
        return function(job, result)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return {function.__name__: {"passed": False, "reason": f"check not evaluable: {type(exc).__name__}: {exc}"}}


def persist(root, job, result, provenance):
    path = output_path(root, job)
    if job["kind"] == "td":
        write_record(path, RunSpec.from_dict(job["spec"]), result, provenance, root)
    elif job["kind"] in ("bs_sequence", "bs_benchmark"):
        data = {"identity": job["identity"], "request": job["request"], "result": result,
                "provenance": provenance, "schema": "offfiducial-background/0.1"}
        data["payload_sha256"] = _payload_digest(data)
        write_json(path, data, root)
    else:
        write_archive(path, job["request"], result, provenance, root)
    return path


def failed_checks(payload):
    if payload["kind"] == "bs_sequence":
        rows = payload["result"]["rows"]
        failed = [f"sigma_c={r['sigma_c']}" for r in rows
                  if r["state"] != "SOLVED" or not all(r["checks"].values())]
        if payload["result"]["turning_point"]["state"] != "LOW_CENTRAL_FIELD_SIDE":
            failed.append("turning_point")
        return failed
    if payload["kind"] == "bs_benchmark":
        return [f"{name}:{q}" for name, c in payload["result"]["configurations"].items()
                for q, v in c["comparison"].items() if not v.get("passed")]
    return [k for k, v in payload["checks"].items() if not v["passed"]]


def execute_job(label, job, root_string, provenance, plan_jobs):
    root = Path(root_string)
    start = time.perf_counter()
    try:
        path = output_path(root, job)
        reused = path.exists()
        if not reused:
            result = compute_job(label, job, root, provenance, plan_jobs)
            persist(root, job, result, provenance)
        payload = load_job(root, job, provenance)  # always read back the published bytes
        wall = payload["result"]["wall_seconds"] if job["kind"] in ("bs_sequence", "bs_benchmark") \
            else payload["metadata"]["wall_seconds"]
        failed = failed_checks(payload)
        # Execution success is not acceptance: record checks are reported separately and the
        # scientific decision is made only by the report.
        return {"label": label, "execution_state": "COMPLETE", "reused": reused, "identity": job["identity"],
                "file_sha256": _sha256_file(path), "wall_seconds": wall,
                "elapsed_seconds": time.perf_counter()-start, "failed_checks": failed,
                "record_check_state": "FAILED" if failed else "PASSED",
                "scientific_acceptance": "NOT_EVALUATED_AT_EXECUTION"}
    except Exception as exc:
        failure = {"label": label, "execution_state": "FAILED", "job": job, "exception": type(exc).__name__,
                   "message": str(exc), "traceback": traceback.format_exc(), "provenance": provenance,
                   "utc": datetime.now(timezone.utc).isoformat()}
        write_json(root/"failures"/f"{label}-{uuid.uuid4().hex}.json", failure, root)
        return {"label": label, "execution_state": "FAILED", "identity": job["identity"],
                "exception": type(exc).__name__, "message": str(exc)}


def select_next(pending, plan_jobs, complete, running, remaining):
    """First pending job whose parents are complete and whose estimate fits."""
    for label in pending:
        job = plan_jobs[label]
        if label in running or any(p not in complete for p in job["parents"]):
            continue
        if job["estimated_seconds"] <= remaining:
            return label
    return None


# --------------------------------------------------------------- report helpers

def turning_point_side(rows, target, fiducial=FIDUCIAL_SIGMA_C):
    """Is target on the increasing low-central-field side of the sampled maximum?

    Declared test: every background solved and passed; the sampled mass is
    strictly increasing up to its maximum; target lies below that maximum; and
    the maximum sits at the sample nearest the frozen mini BS I value or one of
    its two neighbours (Macedo et al. place mini BS I at the maximum mass).
    """
    solved = [r for r in rows if r["state"] == "SOLVED" and all(r["checks"].values())]
    if len(solved) != len(rows) or len(rows) < 3:
        return {"state": "UNQUANTIFIED", "reason": "not every declared background solved and passed"}
    sc = np.array([r["sigma_c"] for r in rows])
    mass = np.array([r["mass_mu"] for r in rows])
    order = np.argsort(sc)
    sc, mass = sc[order], mass[order]
    peak = int(np.argmax(mass))
    nearest = int(np.argmin(np.abs(sc-fiducial)))
    increasing = bool(np.all(np.diff(mass[:peak+1]) > 0))
    near_fiducial = bool(abs(peak-nearest) <= 1)
    side = bool(increasing and near_fiducial and target < sc[peak] and np.any(np.isclose(sc, target)))
    return {"state": "LOW_CENTRAL_FIELD_SIDE" if side else "NOT_ESTABLISHED",
            "sampled_peak_sigma_c": float(sc[peak]), "sampled_peak_mass_mu": float(mass[peak]),
            "mass_strictly_increasing_up_to_peak": increasing, "peak_within_one_sample_of_fiducial": near_fiducial,
            "target_sigma_c": target, "sigma_c": sc.tolist(), "mass_mu": mass.tolist(),
            "scope": ("location relative to the sampled mass maximum only; radial stability is inferred from "
                      "the turning-point statement of Macedo et al., not computed")}


def table_comparison(rows, table):
    """Published-table agreement at table precision, bracketing input rounding."""
    solved = [r for r in rows if r["state"] == "SOLVED" and all(r["checks"].values())]
    out = {}
    if len(solved) != len(rows):
        return {q: {"passed": False, "state": "UNQUANTIFIED", "reason": "a bracketing solve failed"}
                for q in ("mass_mu", "omega_s_over_mu", "radius_99_mu")}
    for q in ("mass_mu", "omega_s_over_mu", "radius_99_mu"):
        values = np.array([r[q] for r in rows])
        allowance = (max(r["radial_spacing_mu"] for r in rows) if q == "radius_99_mu"
                     else POLICY["bs_table_numerical_allowance"][q]*float(np.max(np.abs(values))))
        lo = float(values.min())-TABLE_HALF_UNIT[q]-allowance
        hi = float(values.max())+TABLE_HALF_UNIT[q]+allowance
        central = float(values[len(values)//2])
        out[q] = {"passed": bool(lo <= table[q] <= hi), "table": table[q], "central": central,
                  "bracket_min": float(values.min()), "bracket_max": float(values.max()),
                  "relative_difference_central": (central-table[q])/table[q],
                  "allowance": allowance, "table_half_unit": TABLE_HALF_UNIT[q],
                  "scope": "agreement with rounded published background values; not an error bound"}
    return out


class EvidenceInputError(ValueError):
    """Data that cannot support the requested comparison; never converted into a measurement."""


def _real_series(t, y, name):
    """Validate before casting: a real waveform comparison accepts only real numeric data."""
    t_raw, y_raw = np.asarray(t), np.asarray(y)
    if t_raw.dtype.kind not in "iuf" or y_raw.dtype.kind not in "iuf":
        raise EvidenceInputError(f"{name}: time and field must be real numeric arrays, got {t_raw.dtype} and "
                                 f"{y_raw.dtype} (complex data need complex_difference)")
    if t_raw.ndim != 1 or y_raw.shape != t_raw.shape:
        raise EvidenceInputError(f"{name}: time and field must be 1-D arrays of equal length, got "
                                 f"{t_raw.shape} and {y_raw.shape}")
    if len(t_raw) < 3:
        raise EvidenceInputError(f"{name}: fewer than three samples")
    t_out, y_out = t_raw.astype(float), y_raw.astype(float)
    if not (np.all(np.isfinite(t_out)) and np.all(np.isfinite(y_out))):
        raise EvidenceInputError(f"{name}: nonfinite time or field values")
    if not np.all(np.diff(t_out) > 0):
        raise EvidenceInputError(f"{name}: time grid is not strictly increasing")
    return t_out, y_out


def series(payload, field):
    if field == "psi":
        return {"t": payload["t"], "psi": payload["psi"]}
    if field == "correction":
        return {"t": payload["t"], "psi": payload["result"]["response"]["correction"]}
    raise ValueError(field)


def _complex_series(item, name):
    w_raw, z_raw = np.asarray(item["omega"]), np.asarray(item["value"])
    if w_raw.dtype.kind not in "iuf":
        raise EvidenceInputError(f"{name}: frequency grid must be real numeric")
    if z_raw.dtype.kind != "c":
        raise EvidenceInputError(f"{name}: scattering data must be complex; a real array would discard the phase")
    if w_raw.ndim != 1 or z_raw.shape != w_raw.shape or len(w_raw) < 3:
        raise EvidenceInputError(f"{name}: frequency grid and values must be equal 1-D arrays of at least 3 samples")
    w, z = w_raw.astype(float), z_raw.astype(complex)
    if not (np.all(np.isfinite(w)) and np.all(np.isfinite(z.real)) and np.all(np.isfinite(z.imag))):
        raise EvidenceInputError(f"{name}: nonfinite frequency or value")
    if not np.all(np.diff(w) > 0):
        raise EvidenceInputError(f"{name}: frequency grid is not strictly increasing")
    return w, z


def complex_difference(a, b):
    """Complex scattering comparison on one identical frequency grid (no interpolation, no casting).

    Reports the complex L2 difference (trapezoid in omega), its ratio to the base norm, the
    largest modulus and magnitude differences and the largest wrapped phase difference
    arg(b/a) where both are nonzero.  A measured difference, not an error bound.
    """
    wa, za = _complex_series(a, "base")
    wb, zb = _complex_series(b, "variant")
    if wa.shape != wb.shape or not np.array_equal(wa, wb):
        raise EvidenceInputError("complex comparisons require an identical frequency grid; complex data are "
                                 "not interpolated")
    d = zb-za
    reference = float(np.sqrt(trapezoid(np.abs(za)**2, wa)))
    l2 = float(np.sqrt(trapezoid(np.abs(d)**2, wa)))
    both = (np.abs(za) > 0) & (np.abs(zb) > 0)
    phase = np.angle(zb[both]/za[both])
    return {"grid": "identical frequency grid", "samples": int(len(wa)), "L2": l2, "reference_L2": reference,
            "relative_L2": l2/reference if reference > 0 else None,
            "max_abs_difference": float(np.max(np.abs(d))),
            "max_abs_magnitude_difference": float(np.max(np.abs(np.abs(zb)-np.abs(za)))),
            "max_abs_phase_difference": float(np.max(np.abs(phase))) if phase.size else None,
            "phase_definition": "arg(b/a) wrapped to (-pi, pi] where both are nonzero",
            "is_error_bound": False}


def spacing(payload, rule):
    if rule == "actual_dx":
        return float(payload["metadata"]["actual_dx"])
    if rule == "radial_spacing_mu":
        return float(payload["metadata"]["background_reconstruction"]["radial_spacing_mu"])
    if rule == "pde_h":
        return float(payload["metadata"]["dx"])
    raise ValueError(rule)


def _windows_l2(t, delta, reference, windows=WINDOWS):
    t, delta = _real_series(t, delta, "difference")
    _, reference = _real_series(t, reference, "reference")
    out = {}
    for name, (lo, hi) in windows.items():
        if t[0] > lo or t[-1] < hi:
            out[name] = {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"}
            continue
        mask = (t >= lo) & (t <= hi)
        if np.count_nonzero(mask) < 3:
            out[name] = {"state": "UNQUANTIFIED", "reason": "TOO_FEW_SAMPLES"}
            continue
        d = float(np.sqrt(trapezoid(delta[mask]**2, t[mask])))
        n = float(np.sqrt(trapezoid(reference[mask]**2, t[mask])))
        out[name] = {"state": "MEASURED", "difference_L2": d, "reference_L2": n,
                     "relative_difference": d/n if n > 0 else None, "is_error_bound": False}
    return out


def pair_difference(a, b, field, windows=WINDOWS):
    """b-a on a common supported grid, normalised by a; no fit or alignment.

    Raises EvidenceInputError for anything but finite real data on strictly increasing
    grids (finding B: the identical-grid path used to cast complex data to float).
    """
    sa, sb = series(a, field), series(b, field)
    ta, ya = _real_series(sa["t"], sa["psi"], "base")
    tb, yb = _real_series(sb["t"], sb["psi"], "variant")
    if ta.shape == tb.shape and np.array_equal(ta, tb):
        t, grid = ta, "identical native time grid"
    else:
        try:
            t, (ya, yb) = common_series([{"t": ta, "psi": ya}, {"t": tb, "psi": yb}], dt=POLICY["common_grid_dt"])
        except ValueError as exc:
            raise EvidenceInputError(f"no adequate common support: {exc}") from exc
        grid = "cubic-spline common grid dt=0.07 M"
    return {"grid": grid, "windows": _windows_l2(t, yb-ya, ya, windows)}


def control_evidence(control, payloads, sequences):
    base, variant = payloads.get(control["base"]), payloads.get(control["variant"])
    window_set = _windows(control)
    if base is None or variant is None:
        return {**control, "state": "INCOMPLETE", "windows": {w: {"state": "UNQUANTIFIED", "reason": "MISSING"}
                                                              for w in window_set}}
    checks_pass = not failed_checks(base) and not failed_checks(variant)
    try:
        diff = pair_difference(base, variant, control["field"], window_set)
    except EvidenceInputError as exc:
        reason = f"INVALID_INPUT: {exc}"
        return {**control, "state": "INVALID_INPUT", "reason": reason, "checks_pass": checks_pass,
                "windows": {w: {"state": "UNQUANTIFIED", "reason": reason,
                                "fraction_of_conditional_refinement_estimate": None} for w in window_set}}
    reference = sequences[control["estimate_sequence"]].get("refinement", {})
    windows = {}
    for name in window_set:
        row = dict(diff["windows"][name])
        est = reference.get(name, {}).get("fine_error_estimate_L2")
        if row["state"] != "MEASURED" or not checks_pass or est is None or est <= 0:
            row.update(state="UNQUANTIFIED", fraction_of_conditional_refinement_estimate=None,
                       reason=(row.get("reason") if row["state"] != "MEASURED" else
                               "RECORD_CHECKS_FAILED" if not checks_pass else
                               "NO_CONDITIONAL_REFINEMENT_ESTIMATE"))
        else:
            fraction = row["difference_L2"]/est
            row.update(fraction_of_conditional_refinement_estimate=fraction,
                       state=("BELOW_DECLARED_REFINEMENT_BUDGET"
                              if fraction <= POLICY["control_budget_fraction"] else "SYSTEMATIC_NOT_SUBDOMINANT"))
        windows[name] = row
    drift = None
    if base["kind"] == "td" and variant["kind"] == "td":
        xb, xv = np.asarray(base["x"]), np.asarray(variant["x"])
        start = int(np.argmin(np.abs(xv-xb[0])))
        overlap = xv[start:start+len(xb)]
        drift = float(np.max(np.abs(overlap-xb))) if len(overlap) == len(xb) else None
    return {**control, "state": "COMPLETE", "checks_pass": checks_pass, "grid": diff["grid"],
            "windows": windows, "max_overlap_grid_drift_M": drift,
            "scope": "MEASURED_SENSITIVITY_NOT_SYSTEMATIC_ERROR_BOUND"}


def _unquantified(table, reason):
    """Clear every estimate a failed or unsupported parent can no longer support (finding B)."""
    return {w: {**v, "state": "UNQUANTIFIED", "reason": reason, **{k: None for k in ESTIMATE_KEYS}}
            for w, v in table.items()}


BVP_KEYS = ("raw_mass_mu_units", "omega_s_over_mu", "EKG_residual")


def _equilibrium_digest(payload):
    equilibrium = (payload.get("result") or {}).get("equilibrium")
    if not isinstance(equilibrium, Mapping) or not equilibrium:
        return None
    h = hashlib.sha256()
    for key in sorted(equilibrium):
        array = np.ascontiguousarray(np.asarray(equilibrium[key]))
        h.update(f"{key}|{array.dtype.str}|{array.shape}|".encode())
        h.update(array.tobytes())
    return h.hexdigest()


def _same(metas, key):
    values = [canonical_json(m[key]) if key in m else None for m in metas]
    return None not in values and len(set(values)) == 1


def bs_background_gate(seq, payloads):
    """What a BS level sequence varies, and whether it varies only that (finding B).

    TD refinement (h=actual_dx) is a fixed-background claim: the BVP solution, the dense
    equilibrium arrays and the native source/observer must be identical on every level.
    Profile sampling (h=radial_spacing_mu) varies the dense background reconstruction at one
    TD grid and one BVP solution; it is not a fixed-background TD refinement.
    """
    metas = [p["metadata"] for p in payloads]
    if seq["h"] == "actual_dx":
        digests = [_equilibrium_digest(p) for p in payloads]
        compared = {k: _same(metas, k) for k in BVP_KEYS+("source_native", "observer_native")}
        compared["equilibrium_arrays_sha256"] = None not in digests and len(set(digests)) == 1
        return {"kind": "TD_GRID_AT_FIXED_BACKGROUND", "identical": all(compared.values()), "compared": compared,
                "equilibrium_sha256": digests, "failure_state": "BACKGROUND_NOT_FIXED"}
    if seq["h"] == "radial_spacing_mu":
        td = {k: _same(metas, k) for k in ("actual_dx", "cfl")}
        bvp = {k: _same(metas, k) for k in BVP_KEYS}
        failure = None if all(td.values()) else "TD_GRID_NOT_FIXED"
        failure = failure or (None if all(bvp.values()) else "BACKGROUND_NOT_FIXED")
        return {"kind": "BACKGROUND_PROFILE_SAMPLING", "identical": failure is None,
                "compared": {**td, **bvp}, "failure_state": failure,
                "scope": "dense equilibrium reconstruction varied at one TD grid and one BVP solution"}
    return {"kind": "UNDECLARED_BS_SEQUENCE", "identical": False, "compared": {},
            "failure_state": "UNDECLARED_REFINEMENT"}


def refinement_kind(seq):
    if seq["model"] == "BS":
        return {"actual_dx": "TD_GRID_AT_FIXED_BACKGROUND", "radial_spacing_mu": "BACKGROUND_PROFILE_SAMPLING"}.get(
            seq["h"], "UNDECLARED")
    return {"actual_dx": "TD_GRID", "pde_h": "INDEPENDENT_PDE_GRID"}.get(seq["h"], "UNDECLARED")


def sequence_evidence(name, seq, payloads, jobs):
    labels = seq["labels"]
    if any(label not in payloads for label in labels):
        return {"state": "INCOMPLETE", "labels": labels,
                "missing": [label for label in labels if label not in payloads]}
    rs = [payloads[label] for label in labels]
    windows = _windows(seq)
    try:
        t, ys = common_series([series(r, seq["field"]) for r in rs], dt=POLICY["common_grid_dt"])
    except ValueError as exc:
        return {"state": "INVALID_INPUT", "labels": labels, "reason": str(exc),
                "refinement": {w: {"state": "UNQUANTIFIED", "reason": f"INVALID_INPUT: {exc}",
                                   **{k: None for k in ESTIMATE_KEYS}} for w in windows}}
    hs = [spacing(r, seq["h"]) for r in rs]
    failed = {label: failed_checks(payloads[label]) for label in labels}
    state = "CHECKS_FAILED" if any(failed.values()) else "NUMERICALLY_ASSESSED"
    raw = assess(t, ys, hs, expected_order=seq["expected_order"], order_tolerance=POLICY["order_band"],
                 windows=windows)
    out = {"labels": labels, "field": seq["field"], "actual_h": hs, "failed_checks": failed,
           "expected_order": seq["expected_order"], "h_rule": seq["h"], "refinement_kind": refinement_kind(seq),
           "windows_M": {w: list(v) for w, v in windows.items()}}
    if seq["model"] == "BS":
        gate = bs_background_gate(seq, rs)
        out["fixed_background"] = gate
        if state == "NUMERICALLY_ASSESSED" and not gate["identical"]:
            state = gate["failure_state"]
    if state != "NUMERICALLY_ASSESSED":
        raw = _unquantified(raw, "RECORD_CHECKS_FAILED" if state == "CHECKS_FAILED" else state)
    out.update(state=state, refinement=raw)
    return out


def residual_evidence(name, seq, sequences, payloads, plan):
    ref = seq["bh_reference"]
    if ref in (None, "INTERNAL"):
        return None
    labels, bh_labels = seq["labels"], plan["sequences"][ref]["labels"]
    if any(label not in payloads for label in labels+bh_labels):
        return {"state": "INCOMPLETE"}
    windows = _windows(seq)
    rs = [series(payloads[label], "psi") for label in labels+bh_labels]
    try:
        t, ys = common_series(rs, dt=POLICY["common_grid_dt"])
    except ValueError as exc:
        empty = {w: {"state": "UNQUANTIFIED", "reason": f"INVALID_INPUT: {exc}", **{k: None for k in ESTIMATE_KEYS}}
                 for w in windows}
        return {"state": "INVALID_INPUT", "bh_reference": ref, "residual_refinement": empty,
                "paired_budget": {w: {"state": "UNQUANTIFIED", "reason": f"INVALID_INPUT: {exc}"} for w in windows}}
    residuals = [ys[k]-ys[k+3] for k in range(3)]
    hs = [spacing(payloads[label], "actual_dx") for label in labels]
    checks_ok = sequences[name]["state"] == "NUMERICALLY_ASSESSED" and sequences[ref]["state"] == "NUMERICALLY_ASSESSED"
    raw = assess(t, residuals, hs, expected_order=seq["expected_order"], order_tolerance=POLICY["order_band"],
                 windows=windows)
    budget = {}
    for w, (lo, hi) in windows.items():
        a = sequences[name]["refinement"][w].get("fine_error_estimate_L2") if checks_ok else None
        b = sequences[ref]["refinement"][w].get("fine_error_estimate_L2") if checks_ok else None
        if a is None or b is None or t[0] > lo or t[-1] < hi:
            budget[w] = {"state": "UNQUANTIFIED", "reason": "BOTH_WAVEFORM_ESTIMATES_REQUIRED"}
            continue
        mask = (t >= lo) & (t <= hi)
        norm = float(np.sqrt(trapezoid(residuals[-1][mask]**2, t[mask])))
        budget[w] = {"state": "CONDITIONAL_ESTIMATE", "error_estimate_L2": a+b, "fine_residual_L2": norm,
                     "relative_error_estimate": (a+b)/norm if norm > 0 else None,
                     "is_error_bound": False, "includes_independent_systematics": False}
    if not checks_ok:
        raw = _unquantified(raw, f"PARENT_SEQUENCE_NOT_ASSESSED: {sequences[name]['state']}/"
                                 f"{sequences[ref]['state']}")
    return {"state": "NUMERICALLY_ASSESSED" if checks_ok else "CHECKS_FAILED", "bh_reference": ref,
            "bh_reference_mode": "MATCHED_RESOLUTION_NOT_IDENTICAL_GRID",
            "residual_refinement": raw, "paired_budget": budget}


def qh_independent_evidence(payloads, sequences):
    labels = [f"QH_PDE_{h:g}" for h in PDE_LEVELS]
    windows = _windows({"windows": sequences.get("QH", {}).get("windows_M", WINDOWS)})
    out = {"levels": {}, "pde_refinement": sequences.get("QH_PDE", {}).get("refinement"),
           "scope": "raw unfitted comparison; neither calculation is a total error bound"}
    fine = payloads.get("QH_fine")
    for h, label in zip(PDE_LEVELS, labels):
        pde, causal = payloads.get(label), payloads.get(f"QH_CAUSAL_{h:g}")
        row = {"state": "INCOMPLETE"}
        if pde is not None and fine is not None:
            row = {"state": "MEASURED", "negative_lag_kernel_L2_fraction":
                   pde["metadata"]["negative_lag_kernel_L2_fraction"],
                   "analytic_FD_minus_PDE": _against_pde(pde, series(fine, "correction"), windows)}
            if causal is not None:
                row["causal_kernel_FD_minus_PDE"] = _against_pde(pde, series(causal, "psi"), windows)
                row["causal_label"] = "DIAGNOSTIC_NOT_PRODUCTION_LAW"
                analytic = np.asarray(causal["result"]["analytic_OWA_correction"])
                row["analytic_minus_causal_FD"] = _windows_l2(np.asarray(causal["t"]),
                    analytic-np.asarray(causal["psi"]), analytic, windows)
                row["analytic_minus_causal_scope"] = ("wall-prescription difference at fixed source/Jost data; "
                                                     "diagnostic only, never part of the acceptance gate")
        out["levels"][f"{h:g}"] = row
    finest = out["levels"][f"{PDE_LEVELS[-1]:g}"]
    late = finest.get("analytic_FD_minus_PDE", {}).get("late", {})
    pde_late = (sequences.get("QH_PDE", {}).get("refinement") or {}).get("late", {})
    checks = {label: failed_checks(payloads[label]) for label in labels if label in payloads}
    gate = (finest.get("state") == "MEASURED" and late.get("relative_L2") is not None
            and late["relative_L2"] < POLICY["qh_independent_late_gate"]
            and finest["negative_lag_kernel_L2_fraction"] < POLICY["qh_negative_lag_screen"]
            and pde_late.get("state") == "OBSERVED_ORDER_CONSISTENT" and not any(checks.values()))
    out["gate"] = {"passed": bool(gate), "late_relative_L2_finest": late.get("relative_L2"),
                   "gate": POLICY["qh_independent_late_gate"], "pde_late_state": pde_late.get("state"),
                   "failed_checks": checks}
    return out


def _against_pde(pde, fd, windows=WINDOWS):
    t, ys = common_series([series(pde, "psi"), fd], dt=POLICY["common_grid_dt"])
    out = {}
    for name, item in _windows_l2(t, ys[1]-ys[0], ys[0], windows).items():
        if item["state"] == "MEASURED":
            item = {**item, "relative_L2": item["relative_difference"],
                    "normalisation": "independent PDE correction norm in the window"}
        out[name] = item
    return out


def runtime_regression(payloads, repo=REPO):
    payload = payloads.get("BH_RUNTIME_REPLAY")
    from .paths import BASELINE_OUTPUTS
    path = BASELINE_OUTPUTS/"ECO_waveforms.h5"
    if _sha256_file(path) != BASELINE_H5_SHA256:
        raise ValueError("Frozen HDF5 checksum changed; regression refused")
    if payload is None:
        return {"state": "UNAVAILABLE"}
    if "spec" in payload and payload["spec"].experiment != FROZEN_REPLAY_EXPERIMENT:
        return {"state": "FAIL", "reason": "REPLAY_RECORD_NOT_AT_THE_FROZEN_EXPERIMENT"}
    with h5py.File(path, "r") as f:
        ts, ys = f["controlled_comparison"]["BH"]["time"][:], f["controlled_comparison"]["BH"]["waveform_real"][:]
    if not np.array_equal(ts, payload["t"]):
        return {"state": "FAIL", "reason": "NATIVE_TIME_GRID_DIFFERS"}
    error = float(np.max(np.abs(ys-payload["psi"])))
    scale = float(np.max(np.abs(ys)))
    failed = failed_checks(payload)
    expected = POLICY["runtime_regression_expected_failed_checks"]
    ok = error <= POLICY["runtime_regression_relative_max"]*scale and failed == expected
    return {"state": "PASS" if ok else "FAIL",
            "max_abs_error": error, "relative_max_error": error/scale, "bit_identical": error == 0.,
            "samples": int(len(ts)), "record_failed_checks": failed, "expected_failed_checks": expected,
            "expected_failure_reason": "frozen x_left=-140 domain: practical left-boundary return near 398 M",
            "purpose": "RUNTIME_REGRESSION_NOT_CONTINUUM_VALIDATION"}


def control_factor_key(name, control):
    return control.get("factor_key") or LEGACY_FACTOR_KEYS.get(name.removesuffix("_CORRECTION"))


def control_coverage(plan):
    """Explicit per-point, per-sequence control coverage (finding C).

    For each point, its production sequence(s) and its BH reference sequence, every factor
    declared for the sequence's model is COVERED (names of the point's controls that vary it
    on that sequence), NOT_APPLICABLE (with the reason) or MISSING.  A control on another
    sequence, model or experiment is never counted.
    """
    out = {}
    for point_name, point in plan["points"].items():
        names = [point["sequence"]]+[point[k] for k in ("correction_sequence", "bh_sequence") if point.get(k)]
        rows = {}
        for seq_name in names:
            model = plan["sequences"][seq_name]["model"]
            factors = {}
            for factor in SEQUENCE_FACTORS[model]:
                reason = NOT_APPLICABLE_FACTORS.get((model, factor))
                if reason:
                    factors[factor] = {"status": "NOT_APPLICABLE", "reason": reason}
                    continue
                covering = sorted(c for c in point["controls"] if c in plan["controls"]
                                  and plan["controls"][c]["estimate_sequence"] == seq_name
                                  and control_factor_key(c, plan["controls"][c]) == factor)
                factors[factor] = {"status": "COVERED" if covering else "MISSING", "controls": covering}
            rows[seq_name] = factors
        out[point_name] = rows
    return out


def decide_point(name, plan, payloads, sequences, residuals, controls, extra):
    """Apply the predeclared acceptance rules; returns states and reasons only."""
    point = plan["points"][name]
    seq_name, bh_name = point["sequence"], point["bh_sequence"]
    labels = plan["sequences"][seq_name]["labels"]+(plan["sequences"][bh_name]["labels"] if bh_name else [])
    hard = {label: ("MISSING" if label not in payloads else failed_checks(payloads[label])) for label in labels}
    hard = {k: v for k, v in hard.items() if v}
    result = {"params": point["params"], "fiducial": point["fiducial"], "model": point["model"],
              "hard_failures": hard, "windows": {}, "reasons": []}
    if any(v == "MISSING" for v in hard.values()):
        result.update(outcome="INCONCLUSIVE", reasons=["production/reference record missing"])
        return result
    if hard:
        result.update(outcome="REJECTED", reasons=["production/reference hard check failed"])
        return result
    point_controls = {c: controls[c] for c in point["controls"]}
    incomplete = [c for c, v in point_controls.items() if v["state"] != "COMPLETE" or not v.get("checks_pass")]
    for w in WINDOWS:
        eco = sequences[seq_name]["refinement"][w]
        parts = {"eco": eco["state"]}
        ok = eco["state"] == "OBSERVED_ORDER_CONSISTENT"
        row = {"eco_relative_estimate": eco.get("fine_relative_error_estimate")}
        if name == "QH":
            corr = sequences[point["correction_sequence"]]["refinement"][w]
            parts["correction"] = corr["state"]
            ok = ok and corr["state"] == "OBSERVED_ORDER_CONSISTENT"
            row["correction_relative_estimate"] = corr.get("fine_relative_error_estimate")
        else:
            bh = sequences[bh_name]["refinement"][w]
            budget = residuals[seq_name]["paired_budget"][w]
            parts.update(bh=bh["state"], residual_budget=budget["state"])
            ok = ok and bh["state"] == "OBSERVED_ORDER_CONSISTENT" and budget["state"] == "CONDITIONAL_ESTIMATE"
            row.update(bh_relative_estimate=bh.get("fine_relative_error_estimate"),
                       residual_relative_budget=budget.get("relative_error_estimate"),
                       residual_refinement_state=residuals[seq_name]["residual_refinement"][w]["state"])
        states = {c: v["windows"][w]["state"] for c, v in point_controls.items()}
        dominant = {c: v["windows"][w].get("relative_difference") for c, v in point_controls.items()
                    if states[c] == "SYSTEMATIC_NOT_SUBDOMINANT"}
        if not ok:
            state = "UNQUANTIFIED"
        elif any(s == "UNQUANTIFIED" for s in states.values()) or incomplete:
            state = "UNQUANTIFIED"
        elif dominant:
            state = "MULTI_COMPONENT_MEASURED"
        else:
            state = "QUANTIFIED_TD_DOMINANT"
        row.update(state=state, evidence=parts, control_states=states,
                   non_subdominant_controls_relative_difference=dominant)
        result["windows"][w] = row
    reasons = []
    for w in POLICY["acceptance_windows"]:
        if result["windows"][w]["state"] not in QUANTIFIED:
            reasons.append(f"{w} window not quantified")
    if incomplete:
        reasons.append("controls incomplete or failed: "+", ".join(incomplete))
    coverage = control_coverage(plan)[name]
    missing = [f"{seq}:{factor}" for seq, factors in coverage.items()
               for factor, row in factors.items() if row["status"] == "MISSING"]
    if missing:
        reasons.append("control coverage missing: "+", ".join(missing))
    result["control_coverage"] = coverage
    gate = extra.get(name, {"passed": True, "reason": "structural checks are record-level hard checks"})
    if not gate["passed"]:
        reasons.append("model gate failed: "+gate.get("reason", ""))
    result["model_gate"] = gate
    result["outcome"] = "ACCEPTED_PILOT_SCOPE" if not reasons else "INCONCLUSIVE"
    result["reasons"] = reasons
    result["scope"] = ("numerically assessed for this point, experiment and the late/full windows only; "
                       "no total error bound, neighbouring parameter, mode, reflectivity or echo claim")
    return result


def provenance_summary(payloads):
    """Distinct producer provenance dictionaries, as stored, with the labels that carry each."""
    out = {}
    for label, payload in sorted(payloads.items()):
        key = digest(payload["provenance"])
        out.setdefault(key, {"provenance": payload["provenance"], "labels": []})["labels"].append(label)
    return out


def _resume_compatible(stored, current):
    try:
        check_provenance(stored, current)
    except ValueError:
        return False
    return True


def post_hoc_experiment_checks(plan, payloads):
    """Analyzer-computed experiment checks for archives that predate them (never stored back)."""
    out = {}
    for label, payload in payloads.items():
        job = plan["jobs"][label]
        if job["kind"] not in ("td", "bs", "qh") or "experiment_consistency" in payload["checks"]:
            continue
        result = payload if job["kind"] == "td" else payload["result"]
        checks = _safe_checks(experiment_checks, job, result)
        out[label] = {"post_hoc_"+k: v for k, v in checks.items()}
    return out


def make_report(plan, root, provenance, execution_logs, repo=REPO, mode="resume"):
    """Evidence report.  ``mode="historical"`` analyses archives produced elsewhere or earlier."""
    payloads, unavailable = {}, {}
    for label, job in plan["jobs"].items():
        try:
            payloads[label] = load_job(root, job, provenance, mode=mode)
        except (OSError, ValueError, KeyError) as exc:
            unavailable[label] = f"{type(exc).__name__}: {exc}"
    post_hoc = post_hoc_experiment_checks(plan, payloads)
    for label, checks in post_hoc.items():  # analyzer-side hard checks; the archive is not modified
        payloads[label] = {**payloads[label], "checks": {**payloads[label]["checks"], **checks}}
    sequences = {name: sequence_evidence(name, seq, payloads, plan["jobs"])
                 for name, seq in plan["sequences"].items()}
    residuals = {}
    for name, seq in plan["sequences"].items():
        if sequences[name]["state"] == "INCOMPLETE":
            continue
        item = residual_evidence(name, seq, sequences, payloads, plan)
        if item is not None:
            residuals[name] = item
    controls = {name: control_evidence(c, payloads, sequences) for name, c in plan["controls"].items()}
    extra = {}
    bs_seq, bs_bench = payloads.get("BS_BACKGROUND_SEQUENCE"), payloads.get("BS_TABLE_BENCHMARK")
    profile = sequences.get("BS_PROFILE", {})
    bs_gate = (bs_seq is not None and bs_seq["result"]["turning_point"]["state"] == "LOW_CENTRAL_FIELD_SIDE"
               and profile.get("state") == "NUMERICALLY_ASSESSED"
               and all(profile["refinement"][w]["state"] == "OBSERVED_ORDER_CONSISTENT"
                       for w in POLICY["acceptance_windows"]))
    extra["BS"] = {"passed": bool(bs_gate), "reason": "turning-point side and order-2 dense-profile sequence",
                   "turning_point": None if bs_seq is None else bs_seq["result"]["turning_point"]}
    qh_ind = qh_independent_evidence(payloads, sequences)
    qh_identity = all(not failed_checks(payloads[label]) for label in plan["sequences"]["QH"]["labels"]
                      if label in payloads)
    extra["QH"] = {"passed": bool(qh_ind["gate"]["passed"] and qh_identity),
                   "reason": "independent PDE late gate, kernel screen, PDE order and hierarchy identity",
                   "independent": qh_ind["gate"]}
    points = {}
    for name in plan["points"]:
        if sequences[plan["points"][name]["sequence"]]["state"] == "INCOMPLETE" or (
                plan["points"][name]["bh_sequence"] and
                sequences[plan["points"][name]["bh_sequence"]]["state"] == "INCOMPLETE"):
            points[name] = {"outcome": "INCONCLUSIVE", "reasons": ["sequence incomplete"],
                            "params": plan["points"][name]["params"]}
            continue
        points[name] = decide_point(name, plan, payloads, sequences, residuals, controls, extra)
    background = {"sequence": None if bs_seq is None else bs_seq["result"],
                  "table_benchmark": None if bs_bench is None else bs_bench["result"],
                  "dense_profile_sequence": profile}
    file_hashes = {}
    for label, job in plan["jobs"].items():
        path = output_path(root, job)
        if label in payloads:
            file_hashes[label] = {"file": str(path.relative_to(root)), "sha256": _sha256_file(path)}
    return {"report_version": REPORT_VERSION, "plan_version": plan["plan_version"],
            "plan_sha256": plan_sha256(plan), "analysis_mode": mode, "analyzer_provenance": provenance,
            "producer_provenance": provenance_summary(payloads),
            "resume_compatible_with_analyzer": (None if provenance is None else
                all(_resume_compatible(p["provenance"], provenance)
                    for p in provenance_summary(payloads).values())),
            "execution_logs": execution_logs,
            "unavailable": unavailable, "job_failed_checks": {k: failed_checks(v) for k, v in payloads.items()},
            "post_hoc_experiment_checks": post_hoc, "control_coverage": control_coverage(plan),
            "sequences": sequences, "residuals": residuals, "controls": controls,
            "bs_background": background, "qh_independent": qh_ind,
            "runtime_regression": runtime_regression(payloads, repo), "points": points,
            "file_hashes": file_hashes, "policy": POLICY,
            "statements": {"estimates_are_error_bounds": False, "library_complete": False,
                           "modes": "NOT_COMPUTED", "reflectivity_GS_WH": "NOT_COMPUTED",
                           "echo_classification": "NOT_COMPUTED", "detectability": "NOT_COMPUTED"}}


def manifest_hashes(value):
    """File hashes from a runner manifest ('files_sha256') or a published summary ('archive_sha256')."""
    for key in ("files_sha256", "archive_sha256"):
        if isinstance(value.get(key), Mapping):
            return dict(value[key])
    raise ValueError("Manifest holds neither files_sha256 nor archive_sha256")


def integrity_check(root, manifest):
    """Byte-level immutability of an archive tree against a published hash manifest."""
    root = Path(root)
    actual = {p.relative_to(root).as_posix(): _sha256_file(p) for p in sorted(root.rglob("*"))
              if p.is_file() and not p.name.endswith(".tmp")}
    mismatched = sorted(k for k in manifest if k in actual and actual[k] != manifest[k])
    missing = sorted(k for k in manifest if k not in actual)
    return {"state": "PASS" if not (mismatched or missing) else "FAIL", "checked": len(manifest),
            "matched": len(manifest)-len(mismatched)-len(missing), "mismatched": mismatched, "missing": missing,
            "unlisted": sorted(k for k in actual if k not in manifest),
            "scope": "SHA-256 of every listed file; unlisted files are reported, not failed"}


def verify_archive(root, plan, *, manifest=None, analyzer_provenance=None, labels=None):
    """Read-only historical verification: integrity, checked readback and both provenances.

    Nothing is written under ``root``.  The producer provenance is reported as stored and the
    analyzer provenance beside it; ``resume_compatible`` says whether new records could be
    added to this run by the analyzer, which historical analysis never requires.
    """
    root = Path(root).resolve()
    labels = list(plan["jobs"]) if labels is None else list(labels)
    integrity = (integrity_check(root, manifest) if manifest is not None else
                 {"state": "NOT_CHECKED", "reason": "no immutable manifest supplied"})
    readback, payloads = {}, {}
    for label in labels:
        try:
            payload = load_job(root, plan["jobs"][label], None, mode="historical")
        except Exception as exc:  # any unreadable archive is a verification failure, reported per label
            readback[label] = {"archive_state": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
            continue
        payloads[label] = payload
        failed = failed_checks(payload)
        readback[label] = {"archive_state": "VERIFIED", "record_check_state": "FAILED" if failed else "PASSED",
                           "failed_checks": failed}
    producers = provenance_summary(payloads)
    compatible = (None if analyzer_provenance is None or not producers else
                  all(_resume_compatible(p["provenance"], analyzer_provenance) for p in producers.values()))
    return {"mode": "historical-read-only", "integrity": integrity, "readback": readback,
            "producer_provenance": producers, "analyzer_provenance": analyzer_provenance,
            "resume_compatible": compatible}


def _step_count(job, payload):
    steps = payload["metadata"].get("steps")
    if job["kind"] == "qh":
        steps = (steps or 0)+int(payload["result"]["source"]["metadata"].get("steps", 0))
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("archived record carries no time-step count; reproduction bound undefined")
    return steps


def reproduce_job(root, plan, label):
    """Recompute one archived waveform record in memory and classify the difference.

    BITWISE_IDENTICAL, WITHIN_ROUNDOFF_BOUND (max |psi difference| at most
    replay_policy.roundoff_bound(steps, amplitude) and the time grids within the same bound
    scaled by the record end) or EXCEEDS_ROUNDOFF_BOUND.  Nothing is written.
    """
    job = plan["jobs"][label]
    if job["kind"] not in ("td", "bs", "qh"):
        raise ValueError("reproduction is defined for TD, BS and QH waveform records")
    stored = load_job(root, job, None, mode="historical")
    fresh = compute_job(label, job, Path(root), None, plan["jobs"])
    steps = _step_count(job, stored)
    experiment = (RunSpec.from_dict(job["spec"]).experiment if job["kind"] == "td"
                  else request_experiment(job["request"]))
    bound = replay_policy.roundoff_bound(steps, experiment.amplitude)
    out = {"label": label, "identity": job["identity"], "steps": steps, "roundoff_bound": bound,
           "bound_rule": "replay_policy.roundoff_bound(steps, initial amplitude); not a continuum error bound"}
    a_t, b_t = np.asarray(stored["t"]), np.asarray(fresh["t"])
    a_y, b_y = np.asarray(stored["psi"]), np.asarray(fresh["psi"])
    if a_t.shape != b_t.shape or a_y.shape != b_y.shape:
        return {**out, "state": "SHAPE_DIFFERS", "stored_samples": int(a_t.size), "fresh_samples": int(b_t.size)}
    dt_dev, dy = float(np.max(np.abs(a_t-b_t))), float(np.max(np.abs(a_y-b_y)))
    out.update(max_abs_time_deviation=dt_dev, max_abs_psi_deviation=dy,
               bitwise_identical=bool(np.array_equal(a_t, b_t) and np.array_equal(a_y, b_y)))
    if out["bitwise_identical"]:
        state = "BITWISE_IDENTICAL"
    elif dy <= bound and dt_dev <= bound*max(1., float(np.max(np.abs(a_t)))):
        state = "WITHIN_ROUNDOFF_BOUND"
    else:
        state = "EXCEEDS_ROUNDOFF_BOUND"
    return {**out, "state": state}


def report_exit_code(report):
    if report["unavailable"]:
        return 1
    if report["runtime_regression"].get("state") != "PASS":
        return 2
    if any(p.get("outcome") == "REJECTED" for p in report["points"].values()):
        return 2
    return 0


# ----------------------------------------------------------------------- commands

def _root(path):
    root = Path(path).resolve()
    assert_output_path(root/"plan.json", root)
    return root


def parse_experiment(items):
    """``sigma=3 observer=120 source=55`` -> Experiment; unknown keys are refused."""
    names = {"sigma": "sigma", "observer": "observer_global_rstar", "source": "source_global_rstar"}
    values = {}
    for item in items or ():
        key, _, value = item.partition("=")
        if key not in names or not value:
            raise ValueError(f"experiment items are sigma=, observer= or source=; got {item!r}")
        values[names[key]] = float(value)
    return Experiment(**values)


def rebuild_plan(stored):
    """Rebuild a stored plan from its own declared experiment, points and block name."""
    if stored.get("block") is None:
        return build_plan(stored.get("experiment"))
    return build_plan(stored.get("experiment"), {m: p["params"] for m, p in stored["points"].items()}, stored["block"])


def ensure_plan(root, experiment=None, points=None, block=None):
    """The root's plan: the stored one (rebuilt and compared exactly) or a new one."""
    path = root/"plan.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf8"))
        stored_experiment = stored.get("experiment")
        if experiment is not None and stored_experiment != asdict(as_experiment(experiment)):
            raise ValueError("Output directory holds a plan for a different experiment; choose a new directory")
        if block is not None and stored.get("block") != block:
            raise ValueError("Output directory holds a different catalogue block")
        plan = rebuild_plan(stored)
        if canonical_json(stored) != canonical_json(plan):
            raise ValueError("Output directory holds a different plan; choose a new directory")
    else:
        plan = build_plan(experiment, points, block)
        write_json(path, plan, root)
    return plan


def _execution_logs(root):
    logs = []
    for path in sorted((root/"execution").glob("*.json")):
        logs.append(json.loads(path.read_text(encoding="utf8")))
    return logs


def command_run(args):
    root = _root(args.output)
    plan = ensure_plan(root)
    return execute_plan(root, plan, make_provenance(plan), budget=args.budget, workers=args.workers,
                        only=args.only, retry_failed=args.retry_failed)


def execute_plan(root, plan, provenance, *, budget, workers, only=None, retry_failed=False):
    """Budgeted, resumable execution of a plan's jobs (shared by every runner built on this one)."""
    jobs = plan["jobs"]
    labels = list(jobs) if not only else only
    unknown = [label for label in labels if label not in jobs]
    if unknown:
        raise ValueError(f"Unknown job labels: {unknown}")
    failed_before = {p.name.rsplit("-", 1)[0] for p in (root/"failures").glob("*.json")}
    complete = {label for label in jobs if output_path(root, jobs[label]).exists()}
    pending = [label for label in labels if label not in complete
               and (retry_failed or label not in failed_before)]
    start = time.perf_counter()
    rows, running = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while True:
            remaining = budget-(time.perf_counter()-start)
            while len(running) < workers:
                label = select_next(pending, jobs, complete, running.values(), remaining)
                if label is None:
                    break
                pending.remove(label)
                running[pool.submit(execute_job, label, jobs[label], str(root), provenance, jobs)] = label
            if not running:
                break
            done, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for future in done:
                label = running.pop(future)
                try:
                    row = future.result()
                except Exception as exc:
                    row = {"label": label, "execution_state": "WORKER_FAILED", "message": repr(exc)}
                rows.append(row)
                if row["execution_state"] == "COMPLETE":
                    complete.add(label)
                print(canonical_json(row), flush=True)
    if make_provenance(plan)["source_sha256"] != provenance["source_sha256"]:
        raise ValueError("Executable source changed during the run; outputs need a new directory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
    write_json(root/"execution"/(stamp+".json"), {"utc": stamp, "budget_seconds": budget,
               "workers": workers, "rows": rows,
               "pending_not_started": pending}, root)
    left = [label for label in jobs if not output_path(root, jobs[label]).exists()]
    executed = [r for r in rows if r["execution_state"] == "COMPLETE"]
    print(f"Executed this call: {len(executed)} (record checks failed: "
          f"{sum(r.get('record_check_state') == 'FAILED' for r in executed)}); outstanding archives: {len(left)}; "
          "scientific acceptance is decided only by 'report'", flush=True)
    return 1 if any(r["execution_state"] != "COMPLETE" for r in rows) else 0


def _stored_plan(root, manifest=None):
    """The plan an archive was produced with, checked against its published hash when given."""
    path = Path(root)/"plan.json"
    if manifest is not None and manifest.get("plan.json") != _sha256_file(path):
        raise ValueError("plan.json does not match the immutable manifest")
    plan = json.loads(path.read_text(encoding="utf8"))
    canonical_json(plan)
    return plan


def _external_output(path, root):
    """A report path for read-only modes: never inside the analysed archive."""
    out = Path(path).resolve()
    if out == root or root in out.parents:
        raise ValueError("read-only modes write nothing inside the analysed archive")
    return out


def _load_manifest(path):
    return None if path is None else manifest_hashes(json.loads(Path(path).read_text(encoding="utf8")))


def _write_external(directory, name, value):
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
    path = directory/f"{stamp}-{name}.json"
    write_json(path, value, directory)
    return path


def command_report(args):
    if args.historical:
        root = Path(args.output).resolve()
        if args.report_output is None:
            raise ValueError("--historical needs --report-output outside the archive")
        target = _external_output(args.report_output, root)
        manifest = _load_manifest(args.manifest)
        plan = _stored_plan(root, manifest)
        analyzer = make_provenance(plan)
        report = make_report(plan, root, analyzer, _execution_logs(root), mode="historical")
        report["integrity"] = (integrity_check(root, manifest) if manifest is not None else
                               {"state": "NOT_CHECKED", "reason": "no immutable manifest supplied"})
        report["archive_root_name"] = root.name
        path = _write_external(target, "historical-report", report)
    else:
        root = _root(args.output)
        plan = ensure_plan(root)
        provenance = make_provenance(plan)
        report = make_report(plan, root, provenance, _execution_logs(root))
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
        path = root/"reports"/(stamp+".json")
        write_json(path, report, root)
    print(f"Report: {path}", flush=True)
    for name, point in report["points"].items():
        windows = {w: v["state"] for w, v in point.get("windows", {}).items()}
        print(name, point["outcome"], windows, point.get("reasons"), flush=True)
    return report_exit_code(report)


def command_verify(args):
    """Re-read every planned archive and hash every file under the root."""
    root = Path(args.output).resolve()
    if args.mode == "historical":
        manifest = _load_manifest(args.manifest)
        plan = _stored_plan(root, manifest)
        out = verify_archive(root, plan, manifest=manifest, analyzer_provenance=make_provenance(plan),
                             labels=args.labels)
        out.update(utc=datetime.now(timezone.utc).isoformat(), archive_root_name=root.name,
                   plan_sha256=plan_sha256(plan), plan_version=plan.get("plan_version"))
        if args.report_output is not None:
            path = _write_external(_external_output(args.report_output, root), "historical-verify", out)
            print(f"Verification: {path}", flush=True)
        else:
            print(canonical_json(out))
        bad = [k for k, v in out["readback"].items() if v["archive_state"] != "VERIFIED"]
        print(f"integrity {out['integrity']['state']}; verified {len(out['readback'])-len(bad)}/"
              f"{len(out['readback'])} archives; producer provenance sets {len(out['producer_provenance'])}; "
              f"resume compatible with this analyzer: {out['resume_compatible']}", file=sys.stderr, flush=True)
        return 1 if bad or out["integrity"]["state"] == "FAIL" else 0
    plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
    if canonical_json(plan) != canonical_json(rebuild_plan(plan)):
        raise ValueError("Stored plan differs from this code's plan")
    provenance = make_provenance(plan)
    readback = {}
    for label, job in plan["jobs"].items():
        try:
            payload = load_job(root, job, provenance)
            failed = failed_checks(payload)
            readback[label] = {"archive_state": "VERIFIED", "record_check_state": "FAILED" if failed else "PASSED",
                               "failed_checks": failed}
        except Exception as exc:
            readback[label] = {"archive_state": "FAILED", "reason": f"{type(exc).__name__}: {exc}"}
    files = {str(p.relative_to(root)): _sha256_file(p) for p in sorted(root.rglob("*")) if p.is_file()
             and not p.name.endswith(".tmp")}
    from .paths import PROJECT_ROOT
    baseline = {name: _sha256_file(PROJECT_ROOT/name) for name in (
        "baseline/eco-waveform-final.ipynb", "baseline/eco_outputs_final/ECO_waveforms.h5")}
    manifest = {"utc": datetime.now(timezone.utc).isoformat(), "root_name": root.name,
                "plan_sha256": plan_sha256(plan), "readback": readback, "files_sha256": files,
                "baseline_sha256": baseline, "provenance": provenance}
    if args.read_only:
        print(canonical_json(manifest))
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid.uuid4().hex[:8]
        write_json(root/"manifests"/(stamp+".json"), manifest, root)
        print(f"Manifest: {root/'manifests'/(stamp+'.json')}", flush=True)
    bad = {k: v for k, v in readback.items() if v["archive_state"] != "VERIFIED"}
    print(f"verified {len(readback)-len(bad)}/{len(readback)} planned archives; files hashed {len(files)}",
          file=sys.stderr, flush=True)
    return 1 if bad else 0


def command_reproduce(args):
    root = Path(args.output).resolve()
    manifest = _load_manifest(args.manifest)
    plan = _stored_plan(root, manifest)
    rows = {label: reproduce_job(root, plan, label) for label in args.labels}
    out = {"utc": datetime.now(timezone.utc).isoformat(), "archive_root_name": root.name,
           "plan_sha256": plan_sha256(plan), "analyzer_environment": replay_policy.environment(),
           "analyzer_provenance": make_provenance(plan),
           "producer_provenance": provenance_summary({k: load_job(root, plan["jobs"][k], None, mode="historical")
                                                     for k in args.labels}),
           "rows": rows}
    if args.report_output is not None:
        path = _write_external(_external_output(args.report_output, root), "reproduction", out)
        print(f"Reproduction: {path}", flush=True)
    else:
        print(canonical_json(out))
    for label, row in rows.items():
        print(label, row["state"], row.get("max_abs_psi_deviation"), row["roundoff_bound"], file=sys.stderr)
    return 0 if all(r["state"] in ("BITWISE_IDENTICAL", "WITHIN_ROUNDOFF_BOUND") for r in rows.values()) else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "report", "verify", "reproduce"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        if name == "plan":
            p.add_argument("--experiment", nargs="*", metavar="KEY=VALUE",
                           help="sigma=, observer=, source= (Schwarzschild r*); default: the production experiment")
            p.add_argument("--block", help="declared catalogue block (eco_library.catalogue_base.BLOCKS), e.g. B1")
        if name == "run":
            p.add_argument("--budget", type=float, default=150.)
            p.add_argument("--workers", type=int, default=2)
            p.add_argument("--only", nargs="*")
            p.add_argument("--retry-failed", action="store_true")
        if name == "report":
            p.add_argument("--historical", action="store_true")
            p.add_argument("--manifest", type=Path)
            p.add_argument("--report-output", type=Path)
        if name == "verify":
            p.add_argument("--read-only", action="store_true")
            p.add_argument("--mode", choices=("resume", "historical"), default="resume")
            p.add_argument("--manifest", type=Path)
            p.add_argument("--report-output", type=Path)
            p.add_argument("--labels", nargs="*")
        if name == "reproduce":
            p.add_argument("--labels", nargs="+", required=True)
            p.add_argument("--manifest", type=Path)
            p.add_argument("--report-output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        points = None
        if args.block:
            from .catalogue_base import BLOCKS
            if args.block not in BLOCKS or args.block == "B0":
                parser.error("--block must name a declared new block (B0 is the executed pilot root)")
            points = BLOCKS[args.block]
        plan = ensure_plan(_root(args.output), parse_experiment(args.experiment) if args.experiment else None,
                           points, args.block)
        print(f"Plan {PLAN_VERSION}: {len(plan['jobs'])} jobs; sha256 {plan_sha256(plan)}", flush=True)
        return 0
    if args.command == "run":
        if not 1 <= args.workers <= (os.cpu_count() or 1):
            parser.error("workers must be between 1 and the logical CPU count")
        if not math.isfinite(args.budget) or args.budget <= 0:
            parser.error("budget must be positive")
        return command_run(args)
    if args.command == "report":
        return command_report(args)
    if args.command == "reproduce":
        return command_reproduce(args)
    return command_verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
