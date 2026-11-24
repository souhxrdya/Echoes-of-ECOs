"""One checksummed HDF5 result file per catalogue entry (format ``eco-analysis-export/0.2``, the format
the signature analysis of stage 3 reads).  Each file carries:

1. the entry's production sequence and its BH reference, coarse to fine: every level's native ``t`` and
   ``psi`` with checksums, ``actual_dx``, ``dt``, ``sample_every``, the measured refinement ratio, the
   declared order with the library's stated basis, and coverage of [0, 500] from t0 = 0;
2. the experiment in full with the metadata areal radii and coordinates; the library identities,
   recomputed and checked; the analysis identities computed by the request's mapping (6.2, 6.3);
3. the BH-reference mode and the five match items decided from record metadata (catalogue MATCH_RULES);
4. every named record check, and the rule that makes a record unusable;
5. the library's per-window report entries verbatim, with the library norm convention named;
6. the initial-data audit and initial-data metadata of every record, and the parameter meanings;
7. modes (rows of the verified modes file, with continuation paths and variant values; no BS rows,
   withdrawn in catalogue 1.1) and, for GS and WH, the reflection 0.2 interface and barrier coefficients
   on a stated grid with plane, variant arrays, local group delays, their uncertainties and per-frequency
   delay states;
8. (0.2) the catalogue 1.1 identities (model or prescription definition, BS equilibrium, QH prescription,
   entry identity, extended analysis identity proposal), coverage, independent-method status and the
   residual source (QH: the stored correction).  ``audit_export`` re-derives every status in a file from
   the arrays and the catalogue entry.

The levels are also given on the common grid t_k = 0.07 k (the library's ``common_series`` rule); an
analysis resamples native arrays itself and needs these only for cross-checks.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import h5py
import numpy as np

from .adapter_archive import _read_tree, _write_tree
from .schema import Experiment, RunSpec, canonical_json, digest

VERSION = "eco-analysis-export/0.2"
# Request section 6.3: provisional analysis names (decision D9 open on the analysis side).
ANALYSIS_NAMES = {"GS": {"compactness": "compactness_M_over_a"}, "WH": {"r0_over_M": "r0_over_M"},
                  "BS": {"sigma_c": "sigma_c"}, "QH": {"EplM": "EplM_hierarchy"}}
PARAMETER_MEANING = {
    "GS": "compactness = M/a with 0 < M/a < 1/2; the shell areal radius is a = M/compactness",
    "WH": "r0_over_M = throat areal radius r0/M > 2; axial delta-throat jump Lambda_th = 2(r0/M - 1)/(r0/M)^2 / M",
    "BS": "sigma_c = central value of sigma = sqrt(8 pi G) phi = sqrt(2) phi_tilde (eco_library.boson_star "
          "LITERATURE field_conversion); mini boson star, nodeless continuation from the declared anchor",
    "QH": "EplM = E_Pl M of the OWA reflectivity; gamma = OWA reflectivity parameter, 1 at every version-1 point",
}
# Request section 6.2: the only convention strings that map; anything else is refused, never mapped.
TIME_TOKEN = {"exp(-i*omega*t)": "exp(-i omega t); Im(omega) < 0 damped"}
UNITS_TOKEN = {"G=c=M_ADM=1": "G=c=1; M_ADM=1"}
FOURIER = "integral psi(t)*exp(+i*omega*t) dt"
PULSE = ("exp(-(x-xs)^2/(2*sigma^2)); carrier=0", 1.0, "left")
AREAL_RADIUS_TOLERANCE = 1e-9
COMMON_DT = .07
COVERAGE = (0., 500.)
RATIO_TOLERANCE = 1e-9
LIBRARY_NORM = ("library: sqrt(trapezoid(y^2, t)) over the common-grid samples with lo <= t <= hi (closed "
                "windows); not the analysis norm (request section 6.4); never converted")
LIBRARY_WINDOWS_TO_ANALYSIS = {"prompt": "PR", "ringdown": "R", "late": "L", "full": "F"}
ANALYSIS_WINDOWS = {"PR": (100., 190., "half-open"), "R": (150., 190., "half-open"),
                    "L": (190., 500., "closed"), "F": (100., 500., "closed")}
DECLARED_ORDER_BASIS = {
    "BH": "docs/METHOD.md: smooth fourth-order finite differences with RK4; a declared "
          "expectation tested by the observed order, not a measured result",
    "GS": "docs/METHOD.md: shell aligned with a node, averaged shell potential and a "
          "regular-centre stencil; smooth-region fourth-order stencils do not establish fourth-order "
          "convergence across the shell, so the declared comparison order is 2; an expectation to be tested",
    "WH": "docs/METHOD.md: smooth fourth-order finite differences with RK4 and the "
          "delta-throat node treatment; an expectation to be tested",
    "BS": "docs/METHOD.md section 3: order 4 at a fixed equilibrium background (TD grid "
          "refined, background fixed); an expectation to be tested",
    "QH": "docs/METHOD.md section 3: order 4 for the total and for the isolated correction "
          "(each level with its own BH stage, measured source and Jost data); an expectation to be tested",
}
UNUSABLE_RULE = ("a record is unusable for evidence if any named check failed or its solver_status is not OK; "
                 "an entry with an unusable production or reference record is not ACCEPTED_V1")
INITIAL_DATA_KEYS = ("F_in", "F_out", "F_ingoing_left_u_plus", "F_outgoing_right_u_minus", "initial_amplitude",
                     "initial_carrier_omega", "initial_direction", "initial_time", "pulse_shape", "sigma",
                     "characteristic_sign_convention", "initial_data_scope", "initial_regular_center_projection",
                     "initial_characteristic_F_in_before_center_projection", "interface_treatment",
                     "center_treatment", "throat_matching", "throat_Lambda")
# Reflection coefficients (GS, WH): grid, variants and the agreement convention, declared before any
# coefficient was exported.  The grid covers all power of the sigma = 4 pulse (|psi~|^2 at omega = 1.5
# is below 1e-15 of its peak).  VERIFIED_BY_VARIATION means two or more variant arrays, each changing a
# numerical setting of the computation (or an independent method), all within VARIANT_AGREEMENT of the
# production array at every frequency; identical variants are UNQUANTIFIED, larger spreads NOT_CONVERGED.
REFLECTIVITY_OMEGA = .01 + .0025*np.arange(597)
VARIANT_AGREEMENT = 1e-6
JOST_VARIANTS = {"jost_step_half": {"step": .0125}, "outer_X_600": {"outer_X": 600.}}
SPREAD_BANDS = {"below_packet_band": (.01, .15), "packet_band_0.15_0.9": (.15, .9), "above_packet_band": (.9, 1.5)}
GS_INTERFACE_VARIANTS = {"radial_ode_notebook_settings": {"rtol": 2e-10, "atol": 1e-13, "start": 2e-5},
                         "radial_ode_tight": {"rtol": 1e-12, "atol": 1e-20, "start": 1e-5}}
REFLECTION_CONVENTION = (
    "analysis conventions.py: at the plane x_p, interface (left of the cavity) A e^{-i w (x-x_p)} + R_i A "
    "e^{+i w (x-x_p)}; barrier (right of the cavity) C e^{+i w (x-x_p)} + R_b C e^{-i w (x-x_p)}.  Library: "
    "U = A e^{+izX} + B e^{-izX} in local characteristics at the seam X = 0 (notebook cell 26), "
    "R_barrier = B/A of the outgoing Jost solution, R_interface = right-moving/left-moving amplitude of the "
    "inner solution (GS: notebook gs_shell_reflectivity; WH: delta throat with the opposite exterior, "
    "R_eff = r + t^2 R_far/(1 - r R_far)); loop = R_barrier R_interface.  The seam is not a potential-free "
    "plane, so moving R to another plane by e^{+-2 i w dx} is exact only across potential-free regions.")


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def tortoise(r, M=1.):
    return r + 2*M*math.log(r/(2*M) - 1)


def analysis_experiment_inputs(experiment, metadata):
    """ExperimentBlock arguments by the table of request section 6.2; anything outside it is refused."""
    e = dict(experiment)
    if (e.get("M_ADM"), e.get("ell"), e.get("polarisation"), e.get("field")) != (1., 2, "axial", "Q_RW"):
        raise ValueError("milestone-1 mapping requires M_ADM = 1, ell = 2, axial, Q_RW")
    if e.get("time_convention") not in TIME_TOKEN or e.get("units") not in UNITS_TOKEN:
        raise ValueError("time or unit convention outside the interface table")
    if e.get("physical_fourier_convention") != FOURIER:
        raise ValueError("Fourier convention differs from the analysis spectrum convention")
    if (e.get("pulse_shape"), e.get("amplitude"), e.get("initial_direction")) != PULSE:
        raise ValueError("pulse shape, carrier, amplitude or direction outside the interface table")
    radii = {}
    for key, name in (("source", "source_global_rstar"), ("observer", "observer_global_rstar")):
        if metadata.get(f"{key}_global") != e[name]:
            raise ValueError(f"metadata {key}_global differs from the experiment")
        radii[key] = metadata[f"{key}_areal_radius_over_M"]*e["M_ADM"]
        if abs(tortoise(radii[key], e["M_ADM"]) - e[name]) > AREAL_RADIUS_TOLERANCE:
            raise ValueError(f"{key} areal radius does not map to the global coordinate within tolerance")
    return {"M_ADM": e["M_ADM"], "ell": e["ell"], "polarisation": e["polarisation"], "field_name": e["field"],
            "time_convention": TIME_TOKEN[e["time_convention"]], "units": UNITS_TOKEN[e["units"]],
            "source_areal_radius": radii["source"], "observer_areal_radius": radii["observer"],
            "x_source_global": e["source_global_rstar"], "x_observer_global": e["observer_global_rstar"],
            "pulse_shape": "gaussian", "pulse_direction": "left-going", "sigma": e["sigma"],
            "F_in": e["F_in"], "t0": e["t0"]}


def analysis_param_hash(model, params):
    """(param_hash, named params, unmapped params) by request section 6.3: payload {family, params} with
    the analysis names and 12 significant digits, full SHA-256.  Unnamed library parameters are returned,
    not hashed: two points differing only in them would share an analysis hash."""
    names = ANALYSIS_NAMES[model]
    if not set(names) <= set(params):
        raise ValueError("a parameter with an analysis name is missing")
    named = {names[k]: float(v) for k, v in params.items() if k in names}
    canon = {k: format(v, ".12g") for k, v in sorted(named.items())}
    payload = json.dumps({"family": model, "params": canon}, sort_keys=True, separators=(",", ":"))
    unmapped = {k: v for k, v in params.items() if k not in names}
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), named, unmapped


def library_point_id(model, params):
    """The library point identity: schema.RunSpec.point_id's rule, applied by the export to every model."""
    return model.lower()+"-"+digest({"model": model, "params": dict(params)})[:16]


def gs_interface_by_radial_ode(omega, compactness, *, rtol, atol, start, ell=2, M=1.):
    """GS interface coefficient by an independent method: adaptive radial solve of the de Sitter interior
    from the regular centre (the ODE of notebook cell 13, gravastar_regular_data, re-implemented with its
    settings as arguments) and the 2x2 shell junction of the notebook's own check (cell 27)."""
    from scipy.integrate import solve_ivp
    a = M/compactness
    H2, f_shell, eps = 2*M/a**3, 1-2*M/a, start*a
    out = np.empty(len(omega), complex)
    for i, w in enumerate(np.asarray(omega, float)):
        def rhs(r, y, w=w):
            f, fp = 1-H2*r*r, -2*H2*r
            return [y[1], -(fp/f)*y[1]-(w*w-f*ell*(ell+1)/r**2)/(f*f)*y[0]]
        sol = solve_ivp(rhs, (eps, a), np.array([eps**(ell+1), (ell+1)*eps**ell], complex), rtol=rtol,
                        atol=atol, method="DOP853")
        if not sol.success:
            raise RuntimeError(sol.message)
        q, dq = sol.y[:, -1]
        out[i] = np.linalg.solve([[1., -q], [1j*w, -f_shell*dq]], [-1., 1j*w])[0]
    return out


def _variation_state(production, variants):
    differences = {n: float(np.max(np.abs(v-production))) for n, v in variants.items()}
    if len(variants) < 2 or not all(np.all(np.isfinite(v)) for v in variants.values()):
        return "UNQUANTIFIED", differences
    if all(d == 0. for d in differences.values()):
        return "UNQUANTIFIED", differences
    return ("VERIFIED_BY_VARIATION" if max(differences.values()) <= VARIANT_AGREEMENT else "NOT_CONVERGED"), differences


def reflectivities(entry, library):
    """The stored reflection 0.2 arrays of a GS or WH entry (archive checksum from catalogue 1.1)."""
    from . import reflection
    block = entry["reflection"]
    path = Path(library)/block["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != block["sha256"]:
        raise ValueError(f"reflection archive {block['file']} differs from catalogue 1.1")
    meta, arr = reflection.read(path)
    arrays, rows = {"omega": arr["omega"]}, {}
    for key, kind in (("R_interface", "interface"), ("R_barrier", "barrier")):
        a, m = arr[key], meta["arrays"][key]
        if m["evidence"] != block["arrays"][key]["evidence"]:
            raise ValueError(f"reflection {key} evidence differs from catalogue 1.1")
        arrays[key] = {"R": a["R"], "variants": a["variants"], "tau": a["tau"], "sigma_tau": a["sigma_tau"],
                       "tau_state_code": a["tau_state_code"]}
        rows[key] = {"kind": kind, "plane": meta["plane_rstar"], "sector": meta["sector"], "evidence": m["evidence"],
                     "provenance": f"library:eco_library.reflection 0.2: {m['method']}",
                     "variant_order": m["variant_order"], "variant_settings": m["variant_settings"],
                     "variant_max_abs_difference": m["variant_max_abs_difference"],
                     "independent_frequencies": m["independent_frequencies"],
                     "independent_max_abs_difference": m["independent_max_abs_difference"],
                     "tau_state_codes": m["tau_state_codes"], "tau_states": m["tau_states"],
                     "tau_states_packet_band": m["tau_states_packet_band"],
                     "group_delay_claimable": m["group_delay_claimable"], "min_abs_R": m["min_abs_R"]}
    return arrays, {"version": meta["version"], "plan": meta["plan"], "archive": block["file"],
                    "archive_sha256": block["sha256"], "arrays": rows, "planes": {"seam_rstar": meta["plane_rstar"]},
                    "convention": REFLECTION_CONVENTION, "grid": meta["grid"], "rules": meta["rules"],
                    "variant_agreement": meta["rules"]["agreement"], "delta": meta["delta"],
                    "group_delay": "tau = Im[(R(w+d) - R(w-d))/(2 d R(w))], d = %g (local derivative); sigma_tau = "
                                   "|tau(d) - tau(2d)| + variant spread; quote tau only where tau_state_code is "
                                   "RESOLVED (code index into tau_state_codes)" % meta["delta"]}


def _level(role, level, label, identity, meta, t, y, checks):
    passed = {name: bool(c["passed"]) for name, c in checks.items()}
    return {"role": role, "level": level, "label": label, "record_id": identity, "actual_dx": meta["actual_dx"],
            "dt": meta["dt"], "sample_every": meta["sample_every"], "sample_spacing": float(np.mean(np.diff(t))),
            "n_samples": int(t.size), "coverage_M": [float(t[0]), float(t[-1])],
            "covers_0_500_from_t0": bool(t[0] == 0. and t[-1] >= COVERAGE[1]),
            "t_sha256": _sha(t), "psi_sha256": _sha(y), "solver_id": meta.get("solver_id"),
            "solver_status": meta.get("solver_status"), "checks": passed,
            "usable": bool(all(passed.values()) and meta.get("solver_status") == "OK")}


def _ratio(levels):
    ratios = [a["actual_dx"]/b["actual_dx"] for a, b in zip(levels, levels[1:])]
    if max(ratios)-min(ratios) > RATIO_TOLERANCE*max(ratios):
        raise ValueError(f"refinement ratio is not constant: {ratios}")
    return ratios


class _Reports:
    def __init__(self):
        self.cache = {}

    def get(self, root, plan):
        from . import offfiducial as off
        if root not in self.cache:
            self.cache[root] = off.make_report(plan, root, None, [], mode="historical")
        return self.cache[root]


def export_entry(entry, library, out_dir, reports=None):
    """Write one export file for a catalogue entry; refuses to overwrite."""
    from . import offfiducial as off
    from .catalogue_base import BLOCK_ROOTS
    from .convergence import common_series
    reports = reports or _Reports()
    library = Path(library)
    root = library/BLOCK_ROOTS[entry["block"]]
    plan = json.loads((root/"plan.json").read_text(encoding="utf8"))
    model = entry["model"]
    point = plan["points"][model]
    if point["params"] != entry["params"]:
        raise ValueError("catalogue entry and block plan disagree on the point")
    seq = point["sequence"]
    eco_labels = plan["sequences"][seq]["labels"]
    load = lambda label: off.load_job(root, plan["jobs"][label], None, mode="historical")
    payloads = {label: load(label) for label in eco_labels}
    eco = [(label, plan["jobs"][label]["identity"], payloads[label]["metadata"], payloads[label]["checks"],
            np.asarray(payloads[label]["t"]), np.asarray(payloads[label]["psi"])) for label in eco_labels]
    arrays = {"native": {}, "common": {}}
    if model == "QH":
        bh, corrections, identity, bitwise = [], [], [], []
        bh_sequence = plan["sequences"]["BH"]["labels"]
        for label, bh_label in zip(eco_labels, bh_sequence):
            stage = payloads[label]["result"]["bh_reference"]
            ts, ys = np.asarray(stage["t"]), np.asarray(stage["psi"])
            c = np.asarray(payloads[label]["result"]["response"]["correction"])
            if not np.array_equal(ts, np.asarray(payloads[label]["t"])):
                raise ValueError("QH stage and record time grids differ")
            identity.append(float(np.max(np.abs(np.asarray(payloads[label]["psi"])-(ys+c)))))
            reference = load(bh_label)
            bitwise.append(bool(np.array_equal(ys, np.asarray(reference["psi"]))))
            bh.append((f"{label}:bh_stage", f"{plan['jobs'][label]['identity']}:bh_stage", stage["metadata"],
                       stage["checks"], ts, ys))
            corrections.append(c)
        cat = entry["bh_reference"]
        if (cat is None or cat.get("identity_max_abs_psi_minus_stage_minus_correction") != identity
                or cat.get("stage_bitwise_equal_to_block_BH") != dict(zip(bh_sequence, bitwise))):
            raise ValueError("QH stage identity differs from the catalogue 1.1 reference block")
        reference_block = {
            "library_reference": plan["sequences"][seq]["bh_reference"], "library_mode": cat["mode"],
            "analysis_mode": "CO_COMPUTED_SAME_SOLVER" if all(cat["match"].values()) else "MATCHED_RESOLUTION",
            "analysis_mode_status": "PROPOSED: QH was outside milestone 1; the analysis owner decides; the five "
                                    "items are derived from the stored stages (catalogue 1.1), not set",
            "match": cat["match"], "match_rules": cat["match_rules"], "match_levels": cat["levels"],
            "match_basis": cat["basis"],
            "identity_max_abs_psi_minus_stage_minus_correction": identity,
            "stage_bitwise_equal_to_block_BH_sequence": dict(zip(bh_sequence, bitwise))}
    else:
        bh_labels = plan["sequences"][point["bh_sequence"]]["labels"]
        refs = {label: load(label) for label in bh_labels}
        bh = [(label, plan["jobs"][label]["identity"], refs[label]["metadata"], refs[label]["checks"],
               np.asarray(refs[label]["t"]), np.asarray(refs[label]["psi"])) for label in bh_labels]
        corrections = None
        cat = entry["bh_reference"]
        if cat is None or cat["mode"] != "MATCHED_RESOLUTION_NOT_IDENTICAL_GRID":
            raise ValueError("catalogue entry lacks the library BH reference block")
        reference_block = {"library_reference": cat["sequence"], "library_mode": cat["mode"],
                           "analysis_mode": "MATCHED_RESOLUTION",
                           "analysis_mode_status": "request section 6.5 rule: MATCHED_RESOLUTION_NOT_IDENTICAL_GRID "
                                                   "maps to MATCHED_RESOLUTION (no cancellation credited)",
                           "match": cat["match"], "match_rules": cat["match_rules"], "match_levels": cat["levels"]}
    series = [{"t": r[4], "psi": r[5]} for r in eco+bh]
    if corrections is not None:
        series += [{"t": r[4], "psi": c} for r, c in zip(eco, corrections)]
    t, ys = common_series(series, dt=COMMON_DT)
    if t[0] != 0. or not np.array_equal(t, COMMON_DT*np.arange(t.size)):
        raise ValueError("common grid is not t_k = 0.07 k")
    arrays["common"] = {"t": t, "eco": np.vstack(ys[:3]), "bh": np.vstack(ys[3:6])}
    if corrections is not None:
        arrays["common"]["correction"] = np.vstack(ys[6:9])
    levels = []
    for role, rows in (("eco", eco), ("bh", bh)):
        for k, (label, identity, meta, checks, tt, yy) in enumerate(rows):
            name = off.LEVEL_NAMES[k]
            arrays["native"][f"{role}_{name}"] = {"t": tt, "psi": yy}
            levels.append(_level(role, name, label, identity, meta, tt, yy, checks))
    if corrections is not None:
        for k, c in enumerate(corrections):
            arrays["native"][f"correction_{off.LEVEL_NAMES[k]}"] = {"t": eco[k][4], "psi": c}
    eco_ratio, bh_ratio = _ratio([l for l in levels if l["role"] == "eco"]), _ratio([l for l in levels if l["role"] == "bh"])
    fine = payloads[eco_labels[-1]]
    meta = fine["metadata"]
    experiment = Experiment(**plan["experiment"])
    if not experiment.fingerprint == plan["experiment_fingerprint"] == entry["experiment_fingerprint"]:
        raise ValueError("library experiment fingerprint does not recompute")
    phash, named, unmapped = analysis_param_hash(model, point["params"])
    point_id = library_point_id(model, point["params"])
    ident = entry["identity"]
    if (phash, point_id) != (ident["analysis_param_hash"], ident["library_point_id"]):
        raise ValueError("identities differ from catalogue 1.1")
    if model in ("GS", "WH"):
        spec_ids = {RunSpec.from_dict(plan["jobs"][label]["spec"]).point_id for label in eco_labels}
        if spec_ids != {point_id}:
            raise ValueError("library point_id does not match RunSpec.point_id")
    report = reports.get(root, plan)
    seqs = [seq] + [s for s in (point.get("correction_sequence"), point.get("bh_sequence")) if s]
    bh_order = off.DECLARED_ORDER["BH"]
    header = {
        "version": VERSION, "entry_id": entry["entry_id"], "block": entry["block"], "model": model,
        "params": point["params"], "parameter_meaning": PARAMETER_MEANING[model],
        "catalogue_state": entry["state"], "catalogue_reasons": entry["reasons"],
        "catalogue_entry_identity_sha256": ident["entry_identity_sha256"],
        "identities": {
            "model_definition": ident["model_definition"], "entry_identity_sha256": ident["entry_identity_sha256"],
            "entry_identity_payload": ident["entry_identity_payload"],
            "bs_equilibrium": ident.get("bs_equilibrium"), "qh_prescription": ident.get("qh_prescription"),
            "extended_analysis_identity_proposal": ident["extended_analysis_identity_proposal"],
            "analysis_param_hash_unique_in_catalogue": ident["analysis_param_hash_unique_in_catalogue"],
            "library_point_id": point_id, "library_point_id_rule": "schema.RunSpec.point_id",
            "library_experiment_fingerprint": plan["experiment_fingerprint"],
            "library_experiment_fingerprint_recomputed": experiment.fingerprint,
            "analysis_param_hash": phash, "analysis_params": named, "unmapped_library_params": unmapped,
            "analysis_experiment_inputs": analysis_experiment_inputs(plan["experiment"], meta),
            "rule": "library and analysis identities hash different field sets; both are given and neither is "
                    "assumed equal to the other (request sections 6.2-6.3)"},
        "experiment": plan["experiment"],
        "coordinates": {k: meta.get(k) for k in ("source_global", "observer_global", "source_native",
                                                 "observer_native", "source_areal_radius_over_M",
                                                 "observer_areal_radius_over_M",
                                                 "native_to_global_offset_right_exterior", "x_interface_global")},
        "interface_kind": "NO_SHARP_INTERFACE" if model == "BS" else "SHARP_INTERFACE",
        "x_interface_global": meta.get("x_interface_global"),
        "levels": {"records": levels, "eco_sequence": seq, "ratio": eco_ratio[0], "ratio_bh": bh_ratio[0],
                   "ratios_measured": {
                       "eco": eco_ratio, "bh": bh_ratio}, "h_rule": "actual_dx",
                   "declared_order": {"eco": off.DECLARED_ORDER[model], "bh": bh_order,
                                      "residual": min(off.DECLARED_ORDER[model], bh_order)},
                   "declared_order_basis": {"eco": DECLARED_ORDER_BASIS[model], "bh": DECLARED_ORDER_BASIS["BH"]},
                   "common_grid": "t_k = 0.07 k from the latest first sample (0), cubic spline on native samples, "
                                  "to the earliest last sample (library convergence.common_series)",
                   "unusable_record_rule": UNUSABLE_RULE},
        "bh_reference": reference_block,
        "library_window_entries": {
            "norm_convention": LIBRARY_NORM, "window_names": LIBRARY_WINDOWS_TO_ANALYSIS,
            "windows_M": plan["common_experiment"]["windows_M"],
            "sequences": {s: report["sequences"][s]["refinement"] for s in seqs},
            "residual": report["residuals"].get(seq), "decision": report["points"][model],
            "observable_resolution": entry["observable_resolution"]},
        "initial_data": {
            "rule": "request section 6.1: Gaussian in psi, Pi = D1_4(psi) for a left-going pulse, F_in the energy "
                    "fraction; an interface projection, where present, is applied once before the first step",
            "records": {r[0]: {"audit": r[3].get("initial_data_audit"),
                               "source_monitor_audit": r[3].get("initial_data_audit_source_monitor"),
                               "metadata": {k: r[2][k] for k in INITIAL_DATA_KEYS if k in r[2]}} for r in eco+bh}},
        "modes": None, "modes_status": entry["modes_status"], "modes_withdrawn": entry.get("modes_withdrawn"),
        "reflectivity": None,
        "reflectivity_status": ("NOT_APPLICABLE: no sharp interface" if model == "BS" else
                                "NOT_EXPORTED: the QH coefficient is a prescription with its own planes in the QH "
                                "archives; its conventions are not reconciled with the analysis in this version"
                                if model == "QH" else entry["reflection"]["status"]+" (reflection 0.2); "
                                + entry["reflection"]["group_delay_rule"]),
        "residual": {"source": "STORED_CORRECTION" if model == "QH" else "ECO_MINUS_MATCHED_BH",
                     "arrays": ("common.correction and native.correction_<level>: the stored correction, i.e. the "
                                "record minus its BH stage without the roundoff of subtracting two O(1) arrays; use "
                                "it for every window, including the early ones where it is near 1e-10"
                                if model == "QH" else "common.eco - common.bh (different grids; no cancellation)")},
        "declared_order_by_sequence": entry["declared_order"], "order_band": entry["order_band"],
        "independent_method": entry["independent_method"], "coverage": entry["coverage"],
        "not_covered": entry["not_covered"], "prescription_domain": entry.get("prescription_domain"),
        "producer_provenance": {k: fine["provenance"].get(k) for k in (
            "git_commit", "git_dirty", "python", "numpy", "scipy", "h5py", "platform")},
    }
    if entry["modes"]:
        path = library/entry["modes"]["file"]
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["modes"]["sha256"]:
            raise ValueError("modes file checksum differs from the catalogue")
        rows = []
        for row in json.loads(raw)["modes"]:
            evidence = row.get("export_evidence", row["evidence"])
            rows.append({"library_row": row, "analysis_mode_arguments": {
                "omega": row["omega"], "family": row["family"], "selected_index": row["selected_index"],
                "method": row["method"], "scope": row["scope"],
                "sigma_num": row["sigma_num"] if evidence == "VERIFIED_BY_VARIATION" else None,
                "evidence": evidence, "branch_id": row["branch_id"], "bh_counterpart": None,
                "exhaustive": bool(row["exhaustive"]), "ell": row["ell"], "polarisation": row["polarisation"]}})
        header["modes"] = {"file": entry["modes"]["file"], "sha256": entry["modes"]["sha256"], "rows": rows,
                           "rule": "sigma_num is passed only with VERIFIED_BY_VARIATION; no BH counterpart is "
                                   "stated; lists are not exhaustive; BS poles are withdrawn in catalogue 1.1"}
    if model in ("GS", "WH"):
        arrays["reflectivity"], header["reflectivity"] = reflectivities(entry, library)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir/f"{entry['entry_id']}.h5"
    write_export(path, header, arrays)
    return path, header


def write_export(path, header, arrays):
    """Write header (canonical JSON) and the array tree with a checksummed manifest; never overwrites."""
    path = Path(path)
    if path.exists():
        raise ValueError(f"export exists, not overwritten: {path.name}")
    text = canonical_json(header)
    with h5py.File(path, "x") as handle:
        manifest = {}
        _write_tree(handle, "arrays", arrays, manifest)
        encoded = canonical_json(manifest)
        handle.attrs.update(schema_version=VERSION, header=text,
                            header_sha256=hashlib.sha256(text.encode()).hexdigest(), manifest=encoded,
                            manifest_sha256=hashlib.sha256(encoded.encode()).hexdigest())
    return path


def read_export(path):
    """(header, arrays) after checking the header, the manifest and every array checksum."""
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("schema_version") != VERSION:
            raise ValueError(f"not an export of version {VERSION}")
        for key in ("header", "manifest"):
            if hashlib.sha256(handle.attrs[key].encode()).hexdigest() != handle.attrs[key+"_sha256"]:
                raise ValueError(f"export {key} checksum mismatch")
        manifest = json.loads(handle.attrs["manifest"])
        arrays = _read_tree(handle["arrays"], manifest)
        if manifest:
            raise ValueError("export manifest lists nodes that are not in the file")
        return json.loads(handle.attrs["header"]), arrays


def index_row(path, header):
    """Small, tracked summary of one export (no arrays)."""
    with h5py.File(path, "r") as handle:
        sums = {k: handle.attrs[k+"_sha256"] for k in ("header", "manifest")}
    refl = header["reflectivity"]
    return {"file": Path(path).name, "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "header_sha256": sums["header"], "manifest_sha256": sums["manifest"], "block": header["block"],
            "model": header["model"], "params": header["params"], "catalogue_state": header["catalogue_state"],
            "identities": {k: header["identities"][k] for k in (
                "library_point_id", "library_experiment_fingerprint", "analysis_param_hash", "analysis_params",
                "unmapped_library_params")},
            "bh_reference": {k: header["bh_reference"].get(k) for k in ("analysis_mode", "analysis_mode_status", "match")},
            "levels": [{k: r[k] for k in ("role", "level", "record_id", "actual_dx", "dt", "psi_sha256", "usable",
                                          "covers_0_500_from_t0")} for r in header["levels"]["records"]],
            "ratio": header["levels"]["ratio"], "declared_order": header["levels"]["declared_order"],
            "modes": None if not header["modes"] else [
                {k: r["analysis_mode_arguments"][k] for k in ("branch_id", "omega", "evidence", "sigma_num")}
                for r in header["modes"]["rows"]],
            "reflectivity": None if not refl else {k: {f: v[f] for f in ("evidence", "plane", "variant_max_abs_difference")}
                                                    for k, v in refl["arrays"].items()},
            **{"reflectivity_archive_sha256": None if not refl else refl["archive_sha256"],
                "reflectivity_tau_states": None if not refl else {k: v["tau_states"] for k, v in refl["arrays"].items()},
                "entry_identity_sha256": header["identities"]["entry_identity_sha256"],
                "residual_source": header["residual"]["source"], "modes_status": header["modes_status"]}}


def audit_export(header, arrays, entry):
    """Every status in a 0.2 export re-derived from its own arrays and the catalogue 1.1 entry; returns the
    list of contradictions (empty is the only pass)."""
    from . import reflection
    p = []
    model = header["model"]
    if header.get("version") != VERSION:
        return [f"version {header.get('version')} is not {VERSION}"]
    if header["catalogue_state"] != entry["state"] or header["catalogue_reasons"] != entry["reasons"]:
        p.append("catalogue state or reasons differ from catalogue 1.1")
    ids, ident = header["identities"], entry["identity"]
    if ids["library_point_id"] != library_point_id(model, header["params"]) or header["params"] != entry["params"]:
        p.append("library point id or parameters do not recompute")
    if ids["analysis_param_hash"] != analysis_param_hash(model, header["params"])[0]:
        p.append("analysis param_hash does not recompute")
    for key in ("entry_identity_sha256", "model_definition", "bs_equilibrium", "qh_prescription",
                "extended_analysis_identity_proposal"):
        if ids.get(key) != ident.get(key):
            p.append(f"identity {key} differs from catalogue 1.1")
    for key in ("independent_method", "coverage", "not_covered"):
        if header[key] != entry[key]:
            p.append(f"{key} differs from catalogue 1.1")
    recs = header["levels"]["records"]
    for r in recs:
        native = arrays["native"][f"{r['role']}_{r['level']}"]
        usable = all(r["checks"].values()) and r["solver_status"] == "OK"
        if r["usable"] != usable:
            p.append(f"{r['label']}: usable flag contradicts its checks")
        if r["psi_sha256"] != _sha(native["psi"]) or r["t_sha256"] != _sha(native["t"]):
            p.append(f"{r['label']}: array checksum in the header differs from the array")
        if entry["state"] == "ACCEPTED_V1" and not usable:
            p.append(f"{r['label']}: accepted entry with an unusable record")
    common = arrays["common"]
    if not np.array_equal(common["t"], COMMON_DT*np.arange(common["t"].size)):
        p.append("common grid is not t_k = 0.07 k")
    seq = header["levels"]["eco_sequence"]
    order = header["levels"]["declared_order"]
    if order["eco"] != entry["declared_order"][seq] or order["residual"] != min(order["eco"], order["bh"]):
        p.append("declared orders differ from catalogue 1.1 or the residual order is not the smaller one")
    ref = header["bh_reference"]
    if model == "QH":
        if header["residual"]["source"] != "STORED_CORRECTION" or "correction" not in common:
            p.append("QH residual source is not the stored correction")
        else:
            scale = float(np.max(np.abs(common["eco"])))
            if float(np.max(np.abs(common["eco"]-common["bh"]-common["correction"]))) > 1e-13*max(scale, 1.):
                p.append("QH record, stage and correction are not consistent on the common grid")
        if ref["match"] != entry["bh_reference"]["match"]:
            p.append("QH reference flags differ from the catalogue's derived flags")
    else:
        if header["residual"]["source"] != "ECO_MINUS_MATCHED_BH" or "correction" in common:
            p.append("residual source mislabelled")
        if ref["match"] != entry["bh_reference"]["match"] or order["bh"] != entry["declared_order"][entry["bh_reference"]["sequence"]]:
            p.append("BH reference flags or BH order differ from catalogue 1.1")
    if ref["analysis_mode"] == "CO_COMPUTED_SAME_SOLVER" and not all(ref["match"].values()):
        p.append("co-computed reference mode claimed without all five match items")
    rows = (header["modes"] or {}).get("rows", [])
    if model in ("BS", "QH") and (header["modes"] is not None or header["modes_status"] != entry["modes_status"]):
        p.append("BS/QH export carries modes or a different modes status")
    if model in ("GS", "WH"):
        stored = entry["modes"]["rows"]
        if len(rows) != len(stored):
            p.append("mode rows differ from catalogue 1.1")
        for row, cat in zip(rows, stored):
            a = row["analysis_mode_arguments"]
            if a["evidence"] != cat["export_evidence"] or (a["sigma_num"] is not None) != (a["evidence"] == "VERIFIED_BY_VARIATION") \
                    or a["bh_counterpart"] is not None or a["exhaustive"]:
                p.append(f"mode {a['branch_id']}: evidence, sigma_num, counterpart or exhaustiveness overstated")
        refl, data = header["reflectivity"], arrays["reflectivity"]
        if refl["archive_sha256"] != entry["reflection"]["sha256"]:
            p.append("reflection archive differs from catalogue 1.1")
        for key, meta in refl["arrays"].items():
            a = data[key]
            diffs = [float(np.max(np.abs(a["variants"][n]-a["R"]))) for n in meta["variant_order"]]
            indep = meta["independent_max_abs_difference"]
            ok = (len(diffs) >= 2 and any(d != 0. for d in diffs) and max(diffs) <= refl["variant_agreement"]
                  and (indep is None or indep <= refl["variant_agreement"]))
            if (meta["evidence"] == "VERIFIED_BY_VARIATION") != ok or meta["evidence"] != entry["reflection"]["arrays"][key]["evidence"]:
                p.append(f"reflectivity {key}: evidence contradicts the exported variant arrays")
            states = reflection.delay_states(a["R"], a["tau"], a["sigma_tau"])
            codes = np.array([meta["tau_state_codes"].index(s) for s in states], dtype=np.int8)
            if not np.array_equal(codes, a["tau_state_code"]):
                p.append(f"reflectivity {key}: delay states do not recompute from R, tau and sigma_tau")
            if {s: int(np.sum(states == s)) for s in meta["tau_state_codes"]} != meta["tau_states"]:
                p.append(f"reflectivity {key}: delay-state counts differ")
            if meta["group_delay_claimable"] != (meta["evidence"] == "VERIFIED_BY_VARIATION"):
                p.append(f"reflectivity {key}: group-delay claim contradicts the evidence")
    return p


# Values printed in the request (sections 6.2-6.3) for the baseline experiment and two points.
REQUEST_VECTORS = {"library_experiment_fingerprint_E0": "57c7498aa5ead7ce1e40c4cc2193dff20d66bb05c3f34eae6c23b0b04805cb22",
                   "analysis_fingerprint_E0_prefix": "21e66fcaa9838e2b",
                   "GS": {"params": {"compactness": .47}, "point_id": "gs-fd9d4fbb720b3b40", "param_hash_prefix": "472ff4ed1a335f13"},
                   "WH": {"params": {"r0_over_M": 2.0005}, "point_id": "wh-2438af6900deea37", "param_hash_prefix": "8440259ed72a2d08"}}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("export", "index"))
    parser.add_argument("--catalogue", type=Path, default=Path("results/catalogue.json"))
    parser.add_argument("--library", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("runs/entries"))
    parser.add_argument("--entries", nargs="*")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        catalogue = json.loads(args.catalogue.read_text(encoding="utf8"))
        reports = _Reports()
        for entry in catalogue["entries"]:
            if args.entries is None or entry["entry_id"] in args.entries:
                path, header = export_entry(entry, args.library, args.output, reports)
                print(path.name, header["identities"]["analysis_param_hash"][:16], flush=True)
        return 0
    if args.command == "index":
        catalogue = json.loads(args.catalogue.read_text(encoding="utf8"))
        rows = {}
        for entry in catalogue["entries"]:
            path = args.output/f"{entry['entry_id']}.h5"
            header, _ = read_export(path)
            rows[entry["entry_id"]] = index_row(path, header)
        index = {"version": VERSION, "catalogue_sha256": hashlib.sha256(args.catalogue.read_bytes()).hexdigest(),
                 "exports": rows, "regenerate": "python -m eco_library reproduce (entries are written by its last step); array "
                 "checksums are in each file's manifest"}
        text = json.dumps(index, indent=1, sort_keys=True, allow_nan=False)+"\n"
        if args.report:
            args.report.write_text(text, encoding="utf8")
        else:
            print(text)
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
