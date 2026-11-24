"""The catalogue: the base assembly (eco_library.catalogue_base), unchanged, plus the audit outputs.

Added per entry (docs/VALIDATION.md): the model or prescription identity read from the stored
records (never relabelled), the BS equilibrium identity, the QH prescription identity and its validated
domain, an entry identity hash, the request-compatible analysis param_hash with a collision guard and a
proposed extended analysis identity (decision D9 belongs to the analysis owner), BH-reference flags for
QH derived from the stored stages, the declared order of every sequence, the reflection 0.2 block and
the transfer-gate review (GS, WH), the withdrawn BS modes, the independent-method coverage with its gap,
and the observer, source-width and record-length coverage.

No entry state changes.  Every added number is copied from a tracked summary or a stored record (or is
an exact comparison), so the derived floats of version 1.0 remain the only derived floats and the
verification policy of eco_library.verification applies unchanged.  ``status_audit`` re-derives every
exported status from the evidence stored beside it with code separate from the assembly.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import catalogue_base as base_cat

CATALOGUE_VERSION = "eco-response-library/1.1"
BASE_FILE = "base-catalogue.json"
FILE = "catalogue.json"
DEFINITION_VERSION = "eco-model-definitions/1"
REFLECTION_DIR = "reflection"
INPUTS = {"reflection": "reflection-summary.json", "transfer_review": "transfer-gate-review.json",
          "bs_mode_study": "bs-mode-matching-study.json", "qh_study": "qh-1e6-study.json",
          "waveform_audit": "waveform-audit.json", "edge_robustness": "edge-robustness-o120.json",
          "record_length": "record-length-requirements.json"}
# The declared definition of each model; the stored record metadata must equal scope and solver exactly.
MODEL_DEFINITIONS = {
    "GS": {"model_id": "GS/zero-surface-energy-thin-shell-gravastar/axial", "kind": "SPACETIME",
           "scope": "Zero-surface-energy thin-shell gravastar, axial subclass",
           "solver_id": "frozen_axial_fd4_rk4_td_v1",
           "excludes": "polar perturbations and shell dynamics; other gravastar classes"},
    "WH": {"model_id": "WH/yang-xia-miao-thin-shell-schwarzschild-wormhole/axial-delta-throat", "kind": "SPACETIME",
           "scope": "Yang-Xia-Miao thin-shell Schwarzschild wormhole with axial delta throat; not Damour-Solodukhin",
           "solver_id": "frozen_axial_fd4_rk4_td_v1",
           "excludes": "the Damour-Solodukhin geometry (a different spacetime)"},
    "BS": {"model_id": "BS/mini-boson-star/axial-metric-sector", "kind": "SPACETIME",
           "scope": "mini BS, decoupled linear axial metric sector; no polar or scalar perturbations",
           "solver_id": "frozen_mini_bs_ekg_axial_fd4_rk4_v1",
           "excludes": "polar and scalar perturbations, self-interacting potentials, excited (nodal) stars"},
    "QH": {"model_id": "QH/owa-effective-horizon-boundary/axial", "kind": "PRESCRIPTION",
           "scope": "Nonrotating axial Schwarzschild with OWA effective horizon boundary; not a quantum spacetime",
           "solver_id": "owa_measured_characteristic_jost_v1",
           "excludes": "a quantum-object spacetime; any complex-frequency continuation (no QH modes)"},
}
EQUATIONS = "docs/METHOD.md"
BS_REQUEST_KEYS = ("sigma_c", "phi_tilde_c", "equilibrium_n", "equilibrium_rmax", "equilibrium_tol",
                   "equilibrium_solve", "branch_identity")
BS_PROPERTY_KEYS = ("omega_s_over_mu", "omega_s_M", "raw_mass_mu_units", "radius_99_mu_units", "R99_over_M",
                    "EKG_residual")
QH_DEFINING_KEYS = ("K_action", "fft_action", "reflection_planes", "surface_role", "dc_treatment",
                    "low_frequency_treatment", "high_frequency_treatment")
QH_PRESCRIPTION = {
    "name": "Oshita-Wang-Afshordi Boltzmann reflectivity (arXiv:1905.00464v3 PDF Eqs. 2.13-2.16)",
    "formula": "R(omega) = exp(-pi|omega|/kappa) exp(-2 i Im log A), log A = (i q/kappa) ln[gamma q/(4 kappa "
               "E_Pl M e^{-1/2})] + ln Gamma(-2iq/kappa) - ln Gamma(-iq/kappa) - ln Gamma(1 - iq/kappa), "
               "q = |omega|, kappa = 1/(4M), R(-omega) = conj R(omega), global plane x = 0",
    "kind": "PRESCRIPTION: an effective boundary condition on Schwarzschild, not a spacetime"}
QH_VALIDATED = {"EplM": [1e8, 1e12], "gamma": [1., 1.],
                "basis": "docs/VALIDATION.md: at E_Pl M = 1e6 the prescription keeps a negative-lag fraction "
                         ">= 7.2e-4 at every near-horizon plane and the causal implementation differs from the "
                         "exact frequency-domain one by >= 1.49% in the late window (gate 1%); the accepted "
                         "points 1e8, 1e10, 1e12 pass the predeclared independent causal-PDE gate"}
BS_MODES_STATUS = "NOT_VALIDATED_MATCHING_ILL_CONDITIONED"
BS_MODES_REASON = ("docs/VALIDATION.md: the axial poles drift by 19-29% between Schwarzschild matching radii "
                   "1.4 and 2.6 R99 while the matching condition loses its conditioning (|d res/d omega| falls "
                   "from 3e-3 to 2e-8); no matching radius gives a converged approximation, so sigma_num (the "
                   "numerical spread at a fixed radius) does not describe the error; no BS pole is exported and "
                   "no BH-overtone correspondence is stated")
INDEPENDENT_GAP_BS = ("no independent-method comparison of the BS waveform; the evidence is TD refinement, "
                      "the background-reconstruction controls, the fresh-equilibrium checks and the Macedo et al. "
                      "Table I benchmark of the background (docs/VALIDATION.md)")
TRANSFER_SHARED = ("shared by the FD and TD paths: potential, initial data, the frozen TD solver used for the "
                   "BH monitor source, extraction; independent: seam scattering with damped Laplace inversion "
                   "versus direct evolution (docs/VALIDATION.md)")
COVERAGE_OBSERVER_B0 = "MEASURED: r*_o 95 -> 120 at the four B0 points (docs/VALIDATION.md)"
COVERAGE_OBSERVER_EDGE = "MEASURED: r*_o 95 -> 120 at this edge point (docs/VALIDATION.md)"
CHANGES = [
    "entry identity: model or prescription definition from stored metadata, BS equilibrium, QH prescription, "
    "entry_identity_sha256, analysis_param_hash collision guard, proposed extended analysis identity (D9)",
    "QH BH-reference flags derived from the stored stages (no constant flags)",
    "declared order per sequence (GS 2 and BHGS 4 kept separate)",
    "GS/WH reflection 0.2 arrays (VERIFIED_BY_VARIATION) with per-frequency group-delay states",
    "GS/WH transfer-gate review: raw discrepancies and ratio to the combined budget",
    "BS modes withdrawn (" + BS_MODES_STATUS + ")",
    "QH validated prescription domain E_Pl M in [1e8, 1e12] at gamma = 1",
    "independent-method coverage per entry, with the BS gap stated",
    "observer, source-width and record-length coverage per entry",
]


def _canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha_obj(value):
    return hashlib.sha256(_canon(value).encode("utf8")).hexdigest()


def _sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exact(params):
    """Parameter values as exact decimal strings (repr round-trip), for library identities."""
    return {k: repr(float(v)) for k, v in sorted(params.items())}


def _single(metas, key):
    values = {_canon(m.get(key)) for m in metas}
    if len(values) != 1:
        raise ValueError(f"stored levels disagree on {key}")
    return metas[0].get(key)


def model_identity(model, metas):
    """The declared definition, checked against the stored metadata of every production level."""
    d = MODEL_DEFINITIONS[model]
    stored = {k: _single(metas, k) for k in ("model_family", "model_scope", "solver_id")}
    if stored != {"model_family": model, "model_scope": d["scope"], "solver_id": d["solver_id"]}:
        raise ValueError(f"{model}: stored model definition differs from the declared one: {stored}")
    return {"model_id": d["model_id"], "kind": d["kind"], "definition_version": DEFINITION_VERSION,
            "scope": d["scope"], "solver_id": d["solver_id"], "excludes": d["excludes"], "equations": EQUATIONS,
            "frozen_declarations_sha256": _sha_obj(_single(metas, "baseline_provenance"))}


def bs_equilibrium(metas):
    """What equilibrium was requested (identity) and what it is (stored properties, all levels)."""
    request = {k: _single(metas, k) for k in BS_REQUEST_KEYS}
    request["family"] = "mini boson star: massive complex scalar, no self-interaction, nodeless continuation"
    request["field_conversion"] = _single(metas, "literature")["field_conversion"]
    try:
        properties, identical = {k: _single(metas, k) for k in BS_PROPERTY_KEYS}, True
    except ValueError:
        properties, identical = {k: [m.get(k) for m in metas] for k in BS_PROPERTY_KEYS}, False
    sha = _sha_obj(request)
    return {"equilibrium_id": "bs-eq-"+sha[:16], "request_sha256": sha, "request": request,
            "properties": properties, "properties_identical_at_all_levels": identical,
            "rule": "identity = the stored request (central value, solver settings, solve mode, branch rule); "
                    "the properties are stored results and are not part of the identity"}


def qh_prescription(metas, params):
    settings = {k: _single(metas, k) for k in QH_DEFINING_KEYS}
    version = {**QH_PRESCRIPTION, "stored_settings": settings}
    instance = {"prescription_version_sha256": _sha_obj(version), "params": _exact(params)}
    return {"prescription_version_id": "qh-owa-"+_sha_obj(version)[:16],
            "prescription_version_sha256": _sha_obj(version),
            "prescription_instance_sha256": _sha_obj(instance), **version,
            "in_validated_domain": bool(QH_VALIDATED["EplM"][0] <= params["EplM"] <= QH_VALIDATED["EplM"][1]
                                        and QH_VALIDATED["gamma"][0] <= params["gamma"] <= QH_VALIDATED["gamma"][1])}


def analysis_identities(model, params, identity):
    from .interface_export import ANALYSIS_NAMES, analysis_param_hash
    phash, named, unmapped = analysis_param_hash(model, params)
    names = ANALYSIS_NAMES[model]
    payload = {"family": model, "model_id": identity["model_definition"]["model_id"], "definition_version": DEFINITION_VERSION,
               "params": {names.get(k, k): format(float(v), ".12g") for k, v in sorted(params.items())}}
    if model == "BS":
        payload["equilibrium_request_sha256"] = identity["bs_equilibrium"]["request_sha256"]
    if model == "QH":
        payload["prescription_version_sha256"] = identity["qh_prescription"]["prescription_version_sha256"]
    return {"analysis_param_hash": phash, "analysis_params": named, "unmapped_library_params": unmapped,
            "analysis_param_hash_rule": "request section 6.3 (analysis names, 12 significant digits); "
                                        "parameters without an analysis name are not hashed",
            "extended_analysis_identity_proposal": {
                "status": "PROPOSED for analysis decision D9; not used by the pinned analysis code",
                "payload": payload, "sha256": _sha_obj(payload)}}


def qh_bh_reference(root, plan, point):
    """QH reference flags from stored data: the internal stage against the block BH sequence (five items by
    catalogue.MATCH_RULES), each flag also requiring bitwise-equal stage and BH arrays, the record sampled
    on the stage's time grid and record = stage + stored correction exactly."""
    from . import offfiducial as off
    load = lambda label: off.load_job(root, plan["jobs"][label], None, mode="historical")
    qh = [load(label) for label in plan["sequences"][point["sequence"]]["labels"]]
    bh_labels = plan["sequences"]["BH"]["labels"]
    bh = [load(label) for label in bh_labels]
    stages = [p["result"]["bh_reference"] for p in qh]
    match, levels = base_cat.sequence_match([s["metadata"] for s in stages], [b["metadata"] for b in bh])
    bitwise = [bool(np.array_equal(s["t"], b["t"]) and np.array_equal(s["psi"], b["psi"])) for s, b in zip(stages, bh)]
    grid = [bool(np.array_equal(s["t"], p["t"])) for s, p in zip(stages, qh)]
    identity = [float(np.max(np.abs(np.asarray(p["psi"]) - (np.asarray(s["psi"]) +
                                                           np.asarray(p["result"]["response"]["correction"])))))
                for s, p in zip(stages, qh)]
    exact = all(bitwise) and all(grid) and all(i == 0. for i in identity)
    return {"sequence": "BH", "mode": "INTERNAL_STAGE_EQUAL_TO_BLOCK_BH" if exact else "INTERNAL_STAGE_NOT_EQUAL",
            "match": {k: bool(match[k] and exact) for k in base_cat.MATCH_RULES},
            "match_rules": base_cat.MATCH_RULES, "levels": levels,
            "stage_bitwise_equal_to_block_BH": dict(zip(bh_labels, bitwise)),
            "record_time_grid_equal_to_stage": grid,
            "identity_max_abs_psi_minus_stage_minus_correction": identity,
            "basis": "the QH record is its internal BH stage plus the stored correction on the stage's samples; "
                     "each flag holds only if the stage metadata match the block BH records by the five rules "
                     "and the arrays are bitwise equal, so the residual is exactly the stored correction"}


def _record_metas(root, plan, sequence):
    from . import offfiducial as off
    return [off.load_job(root, plan["jobs"][label], None, mode="historical")["metadata"]
            for label in plan["sequences"][sequence]["labels"]]


def entry_model(entry_id):
    return entry_id.split("-", 1)[0].upper()


def _reflection(library, summary, entry_id, params):
    from . import reflection
    point = summary["points"][entry_id]
    rel = f"{REFLECTION_DIR}/{entry_id}.h5"
    sha = _sha_file(library/rel)
    if sha != point["file_sha256"]:
        raise ValueError(f"reflection archive {rel} differs from its summary checksum")
    header, _ = reflection.read(library/rel)
    if header["version"] != summary["version"]:
        raise ValueError("reflection archive version differs from the summary")
    if (header["model"], header["params"], header["plane_rstar"]) != (entry_model(entry_id), params, point["plane_rstar"]):
        raise ValueError(f"reflection archive {rel} describes another point or plane")
    arrays = {}
    for key, row in point["arrays"].items():
        if header["arrays"][key]["evidence"] != row["evidence"]:
            raise ValueError(f"reflection evidence of {entry_id}/{key} differs between archive and summary")
        arrays[key] = {k: row[k] for k in (
            "evidence", "variant_max_abs_difference", "independent_max_abs_difference", "min_abs_R",
            "tau_states", "tau_states_packet_band", "unresolved_omega", "phase_ill_conditioned_omega_range",
            "max_sigma_tau_over_scale", "group_delay_claimable")}
    return {"version": summary["version"], "plan": summary["plan"], "file": rel, "sha256": sha,
            "plane_rstar": point["plane_rstar"], "rules": header["rules"], "arrays": arrays,
            "status": ("VERIFIED_BY_VARIATION" if all(a["evidence"] == "VERIFIED_BY_VARIATION" for a in arrays.values())
                       else "NOT_VERIFIED"),
            "group_delay_rule": "a group delay may be quoted only at frequencies whose tau state is RESOLVED "
                                "(|R| >= R_min and sigma_tau <= 1e-6 max(|tau|, 1 M)); other frequencies are "
                                "excluded, never interpolated"}


def guard_collisions(entries):
    """Refuse a catalogue in which two entries share the request-compatible analysis param_hash (it omits
    parameters without an analysis name, e.g. QH gamma); the entry identities would still differ."""
    hashes = {}
    for entry in entries:
        hashes.setdefault(entry["identity"]["analysis_param_hash"], []).append(entry["entry_id"])
    collisions = {h: ids for h, ids in hashes.items() if len(ids) > 1}
    if collisions:
        raise ValueError(f"analysis param_hash collision (an unnamed parameter varies): {collisions}")


def assemble_pair(library, results):
    """(version-1.0 rebuild, version-1.1 catalogue) from the same archives in one pass."""
    from . import offfiducial as off
    library, results = Path(library), Path(results)
    base = base_cat.assemble(library, results)
    out = copy.deepcopy(base)
    inputs = {k: json.loads((results/v).read_text(encoding="utf8")) for k, v in INPUTS.items()}
    input_sha = {v: _sha_file(results/v) for v in INPUTS.values()}
    block_summaries = {b: json.loads((results/f).read_text(encoding="utf8")) for b, f in base_cat.BLOCK_SUMMARIES.items()}
    edge = {(p["params"] and next(iter(p["params"].items()))): name for name, p in inputs["edge_robustness"]["points"].items()}
    lengths = {(r["model"], _canon(r["params"])): r for r in inputs["record_length"]["rows"]}
    plans = {b: json.loads((library/r/"plan.json").read_text(encoding="utf8")) for b, r in base_cat.BLOCK_ROOTS.items()}
    for entry in out["entries"]:
        model, block, params = entry["model"], entry["block"], entry["params"]
        root, plan = library/base_cat.BLOCK_ROOTS[block], plans[block]
        point = plan["points"][model]
        metas = _record_metas(root, plan, point["sequence"])
        identity = {"library_point_id": None, "model_definition": model_identity(model, metas),
                    "params_exact": _exact(params), "experiment_fingerprint": entry["experiment_fingerprint"]}
        from .interface_export import library_point_id
        identity["library_point_id"] = library_point_id(model, params)
        if model == "BS":
            identity["bs_equilibrium"] = bs_equilibrium(metas)
        if model == "QH":
            identity["qh_prescription"] = qh_prescription(metas, params)
        core = {"model_id": identity["model_definition"]["model_id"], "definition_version": DEFINITION_VERSION,
                "frozen_declarations_sha256": identity["model_definition"]["frozen_declarations_sha256"],
                "params": identity["params_exact"], "experiment_fingerprint": entry["experiment_fingerprint"],
                "bs_equilibrium_request_sha256": (identity.get("bs_equilibrium") or {}).get("request_sha256"),
                "qh_prescription_version_sha256": (identity.get("qh_prescription") or {}).get("prescription_version_sha256")}
        identity["entry_identity_payload"] = core
        identity["entry_identity_sha256"] = _sha_obj(core)
        identity.update(analysis_identities(model, params, identity))
        entry["identity"] = identity
        entry["declared_order"] = {s: plan["sequences"][s]["expected_order"] for s in entry["refinement"]}
        entry["order_band"] = off.POLICY["order_band"]
        if model == "QH":
            entry["bh_reference"] = qh_bh_reference(root, plan, point)
            gate = block_summaries[block]["qh_independent_gate"]
            entry["independent_method"] = {
                "method": "independent causal time-domain PDE (wall in the near-horizon free region)",
                "state": "PASS" if gate["passed"] else "FAIL", "gate_relative_late": gate["gate"],
                "late_relative_L2_finest": gate["late_relative_L2_finest"], "pde_late_state": gate["pde_late_state"],
                "failed_checks": gate["failed_checks"]}
            entry["prescription_domain"] = {**QH_VALIDATED, "in_validated_domain": identity["qh_prescription"]["in_validated_domain"],
                                            "study_sha256": input_sha[INPUTS["qh_study"]]}
            if not identity["qh_prescription"]["in_validated_domain"]:
                entry["reasons"] = entry["reasons"] + ["outside the validated QH prescription domain "
                                                       "(E_Pl M in [1e8, 1e12]; docs/VALIDATION.md)"]
        if model in ("GS", "WH"):
            entry["reflection"] = _reflection(library, inputs["reflection"], entry["entry_id"], params)
            review = inputs["transfer_review"]["points"][entry["entry_id"]]
            entry["transfer"]["gate_review"] = {"windows": review["gate_late_full"], "all_windows": review["windows"],
                                                "original_rule": inputs["transfer_review"]["original_rule"],
                                                "independence": TRANSFER_SHARED,
                                                "file": INPUTS["transfer_review"],
                                                "file_sha256": input_sha[INPUTS["transfer_review"]]}
            ratio = max(review["gate_late_full"][w]["ratio_to_combined_budget"] for w in ("late", "full"))
            entry["independent_method"] = {
                "method": "frequency-domain transfer (seam scattering, damped Laplace inversion) versus direct TD",
                "state": "PASS" if all(entry["transfer"]["gate"][w]["state"] == "AGREES" for w in ("late", "full")) else "FAIL",
                "max_ratio_to_combined_budget_late_full": ratio, "shared_ingredients": TRANSFER_SHARED}
        if model == "BS":
            entry["modes_withdrawn"] = {"rows_in_1_0": len(entry["modes"]["rows"]) if entry["modes"] else 0,
                                        "file_in_1_0": None if not entry["modes"] else entry["modes"]["file"],
                                        "reason": BS_MODES_REASON, "study": INPUTS["bs_mode_study"],
                                        "study_sha256": input_sha[INPUTS["bs_mode_study"]]}
            entry["modes"], entry["modes_status"] = None, BS_MODES_STATUS
            entry["independent_method"] = {"method": None, "state": "GAP", "detail": INDEPENDENT_GAP_BS}
        if model == "QH":
            entry["modes_status"] = ("PRESCRIPTION_DEPENDENT_NOT_COMPUTED: the prescription is defined on the real "
                                     "axis (|omega|); no complex-frequency continuation is established")
        key = next(iter(params.items()))
        observer = (COVERAGE_OBSERVER_B0 if block == "B0" else COVERAGE_OBSERVER_EDGE
                    if key in edge else "NOT_MEASURED (E0 only)")
        length = lengths[(model, _canon(params))]
        entry["coverage"] = {
            "observer": observer, "source_width": ("MEASURED: sigma 3, 4, 5 M (docs/VALIDATION.md)"
                                                   if block == "B0" else "NOT_MEASURED (E0 only)"),
            "source_position": "NOT_STUDIED",
            "record_length": {k: length[k] for k in ("slowest_decay_time_M", "round_trip_M",
                                                     "record_needed_M_declared_rule", "decay_times_in_520M",
                                                     "echo_arrivals_M", "basis")},
            "record_length_file": INPUTS["record_length"]}
        if model == "GS" and length["decay_times_in_520M"] is not None and length["decay_times_in_520M"] < 5:
            entry["not_covered"] = entry["not_covered"] + [
                "ringing beyond the record end (the record holds %.2g slowest-pole decay times)" % length["decay_times_in_520M"]]
    guard_collisions(out["entries"])
    for entry in out["entries"]:
        entry["identity"]["analysis_param_hash_unique_in_catalogue"] = True
    out["catalogue_version"] = CATALOGUE_VERSION
    out["supersedes"] = {"catalogue_version": base["catalogue_version"], "file": f"runs/summaries/{BASE_FILE}",
                         "sha256": _sha_file(results/BASE_FILE),
                         "rule": "1.0 stays unchanged and must still rebuild; 1.1 adds fields and changes no state"}
    out["changes_from_1_0"] = CHANGES
    out["inputs"] = {v: {"sha256": s} for v, s in input_sha.items()}
    out["domains"]["QH"]["validated_interval"] = QH_VALIDATED["EplM"]
    out["domains"]["QH"]["validated_basis"] = QH_VALIDATED["basis"]
    out["statements"] = {**out["statements"],
                         "modes": "GS, WH: verified poles, not exhaustive; BS: withdrawn (" + BS_MODES_STATUS +
                                  "); QH: not computed",
                         "reflection": "GS, WH: reflection 0.2 arrays; group delays only at RESOLVED frequencies",
                         "qh": "a prescription, not a spacetime; validated for E_Pl M in [1e8, 1e12] at gamma = 1",
                         "independent_method_gap": "BS waveforms (no independent-method comparison)",
                         "coverage": "observer: B0 points and GS 0.49, WH 2.0005; source width: B0 points; "
                                     "source position: not studied; records 520 M (docs/VALIDATION.md)"}
    out["counts"] = {s: sum(e["state"] == s for e in out["entries"]) for s in ("ACCEPTED_V1", "INCONCLUSIVE", "REJECTED")}
    return base, out


def assemble(library, results):
    return assemble_pair(library, results)[1]


# --------------------------------------------------------------------------- status audit

AUDIT_AGREEMENT = 1e-6        # reflection 0.2 agreement (docs/METHOD.md)
AUDIT_RESOLUTION = .1         # resolution fraction (docs/METHOD.md)
AUDIT_TRANSFER_FACTOR = 10.   # original transfer gate (eco_library.transfer.AGREEMENT_FACTOR)
AUDIT_RTOL = 1e-12
STATUS_RULES = [
    "entry state = REJECTED if the plan decision is REJECTED, ACCEPTED_V1 if it is ACCEPTED_PILOT_SCOPE and the "
    "observable is RESOLVED in late and full, otherwise INCONCLUSIVE (with at least one reason)",
    "observable RESOLVED iff (refinement budget + non-subdominant controls)/norm <= 0.1, recomputed",
    "OBSERVED_ORDER_CONSISTENT iff |p_observed - declared order| <= band; accepted entries need it for every "
    "sequence in late and full",
    "BH-reference flags recomputed from the stored per-level settings; the not-identical-grid mode only when "
    "grid, dt and domain do not all match; the QH internal-stage mode only with bitwise-equal stages and an "
    "exact record = stage + correction identity",
    "QH: in_validated_domain recomputed from the parameters; outside it never ACCEPTED_V1; accepted QH entries "
    "pass the independent PDE gate",
    "modes: BS withdrawn (no rows), QH none; GS/WH rows VERIFIED_BY_VARIATION only with a finite positive "
    "sigma_num, no BH counterpart, not exhaustive",
    "reflection: VERIFIED_BY_VARIATION iff at least two variants, not all identical, every variant and the "
    "independent check within 1e-6; group delay claimable iff verified; tau-state counts cover the grid",
    "transfer: AGREES iff FD-TD <= allowance = 10 x (TD budget + FD spread + BH monitor); the stated ratio is "
    "FD-TD over that combined budget; every review number equals the entry's own gate (same experiment)",
    "identities: library point id and entry identity hash recompute; analysis param_hash unique; QH unmapped "
    "parameters are exactly gamma",
    "counts equal the entry tallies",
]


def _flags_from_levels(levels):
    """The five MATCH_RULES items recomputed from the stored per-level [eco, bh] settings."""
    def same(detail, *keys):
        return all(detail[k][0] is not None and detail[k][0] == detail[k][1] for k in keys)
    out = {"solver": [], "grid": [], "dt": [], "domain": [], "extraction": []}
    for d in levels:
        grid = same(d, "x_left", "actual_dx", "Nx")
        out["solver"].append(same(d, "solver_id"))
        out["grid"].append(grid)
        out["dt"].append(same(d, "dt"))
        out["domain"].append(same(d, "x_left", "x_right"))
        out["extraction"].append(grid and same(d, "observer_global", "observer_native"))
    return {k: bool(v) and all(v) for k, v in out.items()}


def status_audit(catalogue):
    """Every exported status re-derived from the evidence stored beside it (separate code from the
    assembly).  Returns the list of contradictions; an empty list is the only pass."""
    from .interface_export import library_point_id
    problems = []

    def bad(entry, message):
        problems.append(f"{entry['entry_id']}: {message}")

    for e in catalogue["entries"]:
        model = e["model"]
        res = e["observable_resolution"]
        for w, r in res.items():
            if r["state"] == "UNQUANTIFIED":
                continue
            frac = (r["refinement_budget_L2"]+r["non_subdominant_controls_L2"])/r["norm_L2"]
            if not math.isclose(frac, r["budget_fraction_of_norm"], rel_tol=AUDIT_RTOL):
                bad(e, f"{w}: budget fraction does not recompute")
            if (r["state"] == "RESOLVED") != (r["budget_fraction_of_norm"] <= AUDIT_RESOLUTION):
                bad(e, f"{w}: resolution state {r['state']} contradicts fraction {r['budget_fraction_of_norm']:.3g}")
        resolved = all(res[w]["state"] == "RESOLVED" for w in ("late", "full"))
        expected = ("REJECTED" if e["plan_decision"] == "REJECTED" else
                    "ACCEPTED_V1" if e["plan_decision"] == "ACCEPTED_PILOT_SCOPE" and resolved else "INCONCLUSIVE")
        if e["state"] != expected:
            bad(e, f"state {e['state']} but the stored evidence gives {expected}")
        if e["state"] != "ACCEPTED_V1" and not e["reasons"]:
            bad(e, "entry not accepted but no reason given")
        band = e["order_band"]
        for s, rows in e["refinement"].items():
            order = e["declared_order"][s]
            for w, r in rows.items():
                inside = r["p_observed"] is not None and abs(r["p_observed"]-order) <= band
                if (r["state"] == "OBSERVED_ORDER_CONSISTENT") != inside and r["state"] in (
                        "OBSERVED_ORDER_CONSISTENT", "ORDER_NOT_CONFIRMED"):
                    if not (r["state"] == "ORDER_NOT_CONFIRMED" and r["p_observed"] is None):
                        bad(e, f"{s}/{w}: state {r['state']} contradicts p {r['p_observed']} (order {order})")
                if e["state"] == "ACCEPTED_V1" and r["state"] != "OBSERVED_ORDER_CONSISTENT":
                    bad(e, f"accepted with {s}/{w} {r['state']}")
        ref = e["bh_reference"]
        if model == "QH":
            exact = (all(ref["stage_bitwise_equal_to_block_BH"].values()) and all(ref["record_time_grid_equal_to_stage"])
                     and all(i == 0. for i in ref["identity_max_abs_psi_minus_stage_minus_correction"]))
            flags = {k: v and exact for k, v in _flags_from_levels(ref["levels"]).items()}
            if ref["match"] != flags:
                bad(e, "QH reference flags do not recompute from the stored levels")
            if (ref["mode"] == "INTERNAL_STAGE_EQUAL_TO_BLOCK_BH") != exact:
                bad(e, "QH reference mode contradicts the stored stage identity")
            inside = (QH_VALIDATED["EplM"][0] <= e["params"]["EplM"] <= QH_VALIDATED["EplM"][1]
                      and QH_VALIDATED["gamma"][0] <= e["params"]["gamma"] <= QH_VALIDATED["gamma"][1])
            if e["prescription_domain"]["in_validated_domain"] != inside or \
                    e["identity"]["qh_prescription"]["in_validated_domain"] != inside:
                bad(e, "validated-domain flag does not recompute")
            if e["state"] == "ACCEPTED_V1" and not inside:
                bad(e, "accepted outside the validated QH domain")
            im = e["independent_method"]
            gate_ok = im["late_relative_L2_finest"] <= im["gate_relative_late"] and not any(im["failed_checks"].values())
            if (im["state"] == "PASS") != gate_ok:
                bad(e, "independent PDE state contradicts its gate numbers")
            if e["state"] == "ACCEPTED_V1" and im["state"] != "PASS":
                bad(e, "accepted QH entry without a passed independent PDE gate")
            if e["modes"] is not None:
                bad(e, "QH entry carries modes")
            if set(e["identity"]["unmapped_library_params"]) != {"gamma"}:
                bad(e, "QH unmapped parameters are not exactly gamma")
        elif ref is not None:
            if ref["match"] != _flags_from_levels(ref["levels"]):
                bad(e, "BH-reference flags do not recompute from the stored levels")
            if ref["mode"] == base_cat.LIBRARY_REFERENCE_MODE and all(ref["match"][k] for k in ("grid", "dt", "domain")):
                bad(e, "not-identical-grid mode with identical grid, dt and domain")
        if model == "BS":
            if e["modes"] is not None or e["modes_status"] != BS_MODES_STATUS:
                bad(e, "BS modes not withdrawn")
            if e["independent_method"]["state"] != "GAP" or e["independent_method"]["method"] is not None:
                bad(e, "BS independent-method gap not stated")
        if model in ("GS", "WH"):
            for row in (e["modes"] or {}).get("rows", []):
                if row["export_evidence"] == "VERIFIED_BY_VARIATION" and not (
                        isinstance(row["sigma_num"], float) and math.isfinite(row["sigma_num"]) and row["sigma_num"] > 0):
                    bad(e, f"mode {row['branch_id']}: verified without a finite positive sigma_num")
                if row["export_evidence"] != row["numerical_evidence"] or row["bh_counterpart"] is not None or row["exhaustive"]:
                    bad(e, f"mode {row['branch_id']}: evidence, counterpart or exhaustiveness overstated")
            refl = e["reflection"]
            for key, a in refl["arrays"].items():
                diffs = list(a["variant_max_abs_difference"].values())
                indep = a["independent_max_abs_difference"]
                ok = (len(diffs) >= 2 and any(d != 0. for d in diffs) and max(diffs) <= AUDIT_AGREEMENT
                      and (indep is None or indep <= AUDIT_AGREEMENT))
                if (a["evidence"] == "VERIFIED_BY_VARIATION") != ok:
                    bad(e, f"reflection {key}: evidence {a['evidence']} contradicts its variant numbers")
                if a["group_delay_claimable"] != (a["evidence"] == "VERIFIED_BY_VARIATION"):
                    bad(e, f"reflection {key}: group-delay claim contradicts the evidence")
                if sum(a["tau_states"].values()) != 597 or any(a["tau_states_packet_band"][s] > a["tau_states"][s]
                                                               for s in a["tau_states"]):
                    bad(e, f"reflection {key}: tau-state counts inconsistent")
            if (refl["status"] == "VERIFIED_BY_VARIATION") != all(a["evidence"] == "VERIFIED_BY_VARIATION"
                                                                  for a in refl["arrays"].values()):
                bad(e, "reflection status contradicts its arrays")
            states = []
            for w, g in e["transfer"]["gate"].items():
                combined = g["td_budget_relative"]+g["fd_variant_spread_relative"]+g["bh_monitor_discrepancy_relative"]
                if not math.isclose(g["allowance_relative"], AUDIT_TRANSFER_FACTOR*combined, rel_tol=1e-9):
                    bad(e, f"transfer {w}: allowance is not 10 x the combined budget")
                if (g["state"] == "AGREES") != (g["fd_td_relative"] <= g["allowance_relative"]):
                    bad(e, f"transfer {w}: state contradicts the discrepancy")
                review = e["transfer"]["gate_review"]["all_windows"].get(w)
                if review is None or review["fd_td_relative"] != g["fd_td_relative"] or \
                        review["residual_budget_relative"] != g["td_budget_relative"] or \
                        not math.isclose(review["ratio"], g["fd_td_relative"]/g["td_budget_relative"], rel_tol=1e-12):
                    bad(e, f"transfer review {w}: numbers are not those of the entry's own gate")
                if w in ("late", "full"):
                    states.append(g["state"])
                    r = e["transfer"]["gate_review"]["windows"][w]
                    if any(r[k] != g[k] for k in g):
                        bad(e, f"transfer review {w}: late/full block differs from the entry's gate")
                    if not math.isclose(r["ratio_to_combined_budget"], r["fd_td_relative"]/(
                            r["td_budget_relative"]+r["fd_variant_spread_relative"]+r["bh_monitor_discrepancy_relative"]),
                            rel_tol=1e-9):
                        bad(e, f"transfer review {w}: ratio does not recompute")
            if (e["independent_method"]["state"] == "PASS") != all(s == "AGREES" for s in states):
                bad(e, "independent-method state contradicts the transfer gate")
        ident = e["identity"]
        if ident["library_point_id"] != library_point_id(model, e["params"]):
            bad(e, "library point id does not recompute")
        if ident["entry_identity_sha256"] != _sha_obj(ident["entry_identity_payload"]) or \
                ident["entry_identity_payload"]["params"] != _exact(e["params"]):
            bad(e, "entry identity does not recompute")
    hashes = [e["identity"]["analysis_param_hash"] for e in catalogue["entries"]]
    if len(set(hashes)) != len(hashes):
        problems.append("catalogue: analysis param_hash collision")
    tally = {s: sum(e["state"] == s for e in catalogue["entries"]) for s in ("ACCEPTED_V1", "INCONCLUSIVE", "REJECTED")}
    if tally != catalogue["counts"]:
        problems.append(f"catalogue: counts {catalogue['counts']} differ from the entries {tally}")
    return problems


def check(library, results, catalogue_path, runtime_reference=None):
    """catalogue.check with the 1.1 assembler, plus: the 1.0 file unchanged and still rebuilt from the same
    archives (its own runtime record and policy), and the status audit of the stored 1.1 file."""
    from . import verification as ver
    library, results = Path(library), Path(results)
    holder = {}

    def rebuild(lib, res):
        base, cat = assemble_pair(lib, res)
        holder["base"] = base
        return cat

    out = base_cat.check(library, results, catalogue_path, runtime_reference, rebuild=rebuild)
    stored = json.loads(Path(catalogue_path).read_text(encoding="utf8"))
    base_path = results/BASE_FILE
    base_sha = ver.file_sha256(base_path)
    if "base" in holder:
        policy = ver.rebuild_policy(json.loads(base_path.read_text(encoding="utf8")), holder["base"],
                                    ver.reference_runtime(base_cat.runtime_reference_path(base_path)), base_sha,
                                    ver.runtime_fingerprint())
        out["base_catalogue"] = {"file": BASE_FILE, "sha256": base_sha, "state": policy["state"], "tier": policy["tier"],
                                 "byte_identical": policy["byte_identical"],
                                 "recorded_in_1_1": stored.get("supersedes", {}).get("sha256") == base_sha}
    else:
        out["base_catalogue"] = {"file": BASE_FILE, "sha256": base_sha, "state": "FAIL", "error": "rebuild failed"}
    if not out["base_catalogue"].get("recorded_in_1_1"):
        out["base_catalogue"]["state"] = "FAIL"
    try:
        problems = status_audit(stored)
    except Exception as exc:  # a malformed catalogue cannot be audited: never a pass
        problems = [f"audit error: {type(exc).__name__}: {exc}"]
    out["status_audit"] = {"state": "PASS" if not problems else "FAIL", "problems": problems, "rules": STATUS_RULES}
    out["state"] = ("PASS" if out["state"] == "PASS" and out["base_catalogue"]["state"] == "PASS" and not problems
                    else "FAIL")
    return out


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("assemble", "check", "audit"))
    parser.add_argument("--library", type=Path, default=Path("runs"))
    parser.add_argument("--results", type=Path, default=Path("runs/summaries"))
    parser.add_argument("--catalogue", type=Path, default=Path("results")/FILE)
    parser.add_argument("--check-output", type=Path)
    parser.add_argument("--basis", default="the runtime that assembled this catalogue; a byte-identical rebuild "
                                            "in this runtime is the SAME_RUNTIME tier of eco_library.verification")
    args = parser.parse_args(argv)
    if args.command == "assemble":
        if args.catalogue.exists():
            print(f"exists, not overwritten: {args.catalogue}", file=sys.stderr)
            return 2
        from . import verification as ver
        runtime = base_cat.runtime_reference_path(args.catalogue)
        if runtime.exists():
            print(f"exists, not overwritten: {runtime}", file=sys.stderr)
            return 2
        catalogue = assemble(args.library, args.results)
        args.catalogue.write_text(json.dumps(catalogue, indent=1, sort_keys=True, allow_nan=False)+"\n", encoding="utf8")
        record = {"basis": args.basis, "catalogue": args.catalogue.as_posix(),
                  "catalogue_sha256": ver.file_sha256(args.catalogue), "policy": ver.POLICY_VERSION,
                  "reference_runtime": ver.runtime_fingerprint()}
        runtime.write_text(json.dumps(record, indent=1, sort_keys=True)+"\n", encoding="utf8")
        print(json.dumps(catalogue["counts"]), record["catalogue_sha256"])
        return 0
    if args.command == "audit":
        problems = status_audit(json.loads(args.catalogue.read_text(encoding="utf8")))
        print(json.dumps(problems, indent=1))
        return 0 if not problems else 2
    result = check(args.library, args.results, args.catalogue)
    text = json.dumps(result, indent=1, sort_keys=True, allow_nan=False)+"\n"
    if args.check_output:
        args.check_output.write_text(text, encoding="utf8")
    print(json.dumps({k: result[k] for k in ("state", "rebuild_identical", "integrity")}),
          json.dumps(result["base_catalogue"]), json.dumps(result["status_audit"]["state"]),
          json.dumps({k: result["rebuild_policy"][k] for k in ("tier", "state", "derived_differing_count")}),
          sorted(set(result["reproduction"].values())), file=sys.stderr)
    return 0 if result["state"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
