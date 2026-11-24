"""Measured-source OWA horizon response, with explicit Fourier/reference planes.

No saved waveform is an input.  The production path reruns the BH experiment,
measures its horizon-side field AND spatial derivative, and solves the outgoing
Jost problem.  An optional second-order characteristic PDE provides an
independent reconstruction without using the barrier coefficients or K.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.special import loggamma

from .baseline_runtime import default_notebook_path, load_frozen_runtime
from .solvers import run_td

SOLVER_ID = "owa_measured_characteristic_jost_v1"
_DECLARATIONS = ("quantum_reflection", "quantum_local_reflection",
                 "barrier_coefficients_vectorized", "interp_scattering", "surface_K")


def _runtime(notebook_path=None):
    """Extend the existing hash-pinned runtime; execute no notebook experiments."""
    rt = load_frozen_runtime(notebook_path)
    path = default_notebook_path() if notebook_path is None else Path(notebook_path)
    # Verify the second read too: do not parse bytes changed after the TD loader.
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != rt.provenance["notebook_sha256"]:
        raise ValueError("Frozen notebook SHA-256 changed during QH loading")
    source = "".join(json.loads(raw)["cells"][20]["source"])
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name in _DECLARATIONS]
    if len(nodes) != len(_DECLARATIONS) or {n.name for n in nodes} != set(_DECLARATIONS):
        raise ValueError("Reviewed quantum declaration set changed")
    rt.namespace.update(PchipInterpolator=PchipInterpolator, loggamma=loggamma)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), f"{path.name}:cell20:qh", "exec"), rt.namespace)
    for node in nodes:
        rt.provenance["declarations"].append({"name": node.name, "cell_index": 20,
            "first_line": node.lineno, "last_line": node.end_lineno,
            "source_sha256": hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()})
    rt.provenance["qh_loader"] = "reviewed_ast_qh_declarations_v1"
    return rt


def _number(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, not a boolean")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _params(params):
    if not isinstance(params, Mapping):
        raise ValueError("QH parameters must be a mapping")
    params = dict(params)
    if set(params) - {"EplM", "gamma"}:
        raise ValueError("QH physical parameters are EplM and gamma")
    return {"EplM": _number(params.get("EplM", 1e8), "EplM", positive=True),
            "gamma": _number(params.get("gamma", 1.), "gamma", positive=True)}


def reflection(omega, *, EplM=1e8, gamma=1., surface=None, notebook_path=None):
    """Physical amplitude A_out/A_in; global x=0 basis unless surface is given."""
    p = _params({"EplM": EplM, "gamma": gamma})
    if not np.isrealobj(omega):
        raise ValueError("Real-axis reflectivity requires real frequencies")
    w = np.asarray(omega, float)
    if not np.all(np.isfinite(w)):
        raise ValueError("omega must be finite and real")
    surface = None if surface is None else _number(surface, "surface")
    rt = _runtime(notebook_path)
    result = rt["quantum_reflection"](w, kappa=.25, **p)
    return result if surface is None else result * np.exp(2j*w*surface)


def default_omega_grid():
    return np.unique(np.r_[np.geomspace(1e-4, 1e-2, 36, endpoint=False), np.linspace(.01, 1.5, 420)])


def measured_ingoing_spectrum(t, field, derivative, nfft, *, fft_dt=None):
    """NumPy characteristic I=(FFT(Q)+FFT(Q_x)/(i omega))/2, excluding DC.

    FFT(Q) is not by itself an incoming wave.  The real-axis static split is
    undefined: I[0]=0 is an explicit prescription, not a measured DC amplitude.
    """
    t, field, derivative = map(np.asarray, (t, field, derivative))
    if (t.ndim != 1 or len(t) < 3 or field.shape != t.shape or derivative.shape != t.shape
            or not np.isrealobj(t) or not np.isrealobj(field) or not np.isrealobj(derivative)
            or not np.all(np.isfinite([t, field, derivative]))):
        raise ValueError("Finite real field and derivative on one time grid required")
    dt = float(np.mean(np.diff(t))) if fft_dt is None else _number(fft_dt, "fft_dt", positive=True)
    if abs(t[0]) > 1e-12 or dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("Uniform t=0 source grid required")
    if isinstance(nfft, bool) or not isinstance(nfft, (int, np.integer)) or nfft < len(t) or nfft % 2:
        raise ValueError("An even nfft covering the whole source record is required")
    if nfft > 2**23:
        raise ValueError("FFT allocation exceeds the bounded adapter limit")
    omega = 2*np.pi*np.fft.rfftfreq(nfft, dt)
    field_fft, derivative_fft = np.fft.rfft(field, nfft), np.fft.rfft(derivative, nfft)
    inc = np.zeros_like(field_fft)
    inc[1:] = .5*(field_fft[1:] + derivative_fft[1:]/(1j*omega[1:]))
    outgoing = field_fft - inc
    outgoing[0] = 0.
    band = (omega >= .15) & (omega <= .9)
    power = abs(inc)**2
    weight = float(np.sum(power[band]))
    omega_c = float(np.sum(omega[band]*power[band])/weight) if weight > 0 else None
    return {"omega": omega, "field_fft": field_fft, "derivative_fft": derivative_fft,
            "ingoing_fft": inc, "outgoing_fft": outgoing,
            "omega_c": omega_c, "omega_c_band": [.15, .9],
            "omega_c_definition": "sum(omega*abs(measured_I_numpy)^2)/sum(abs(measured_I_numpy)^2) in stated band",
            "dc_status": "DIRECTION_UNDEFINED_SET_TO_ZERO"}


def reconstruct(source, time, scattering, *, EplM=1e8, gamma=1., surface=-40.,
                observer=95., nfft=None, interpolation="pchip", notebook_path=None):
    """Build a new response from measured source and freshly computed Jost data.

    Suitable for hierarchy/padding/interpolation controls without rerunning an
    identical BH/source experiment.  No output waveform is accepted as source.
    """
    p = _params({"EplM": EplM, "gamma": gamma})
    surface, observer = _number(surface, "surface"), _number(observer, "observer")
    if not np.isrealobj(time):
        raise ValueError("Output time must be real")
    time = np.asarray(time, float)
    if time.ndim != 1 or len(time) < 3 or not np.isfinite(time).all():
        raise ValueError("Finite output time grid required")
    dt = float(np.mean(np.diff(time)))
    if abs(time[0]) > 1e-12 or dt <= 0 or not np.allclose(np.diff(time), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("Uniform t=0 output time grid required")
    if surface >= 0 or observer <= surface:
        raise ValueError("QH surface must be negative and precede observer")
    if float(source["surface"]) != surface or float(scattering["surface"]) != surface:
        raise ValueError("Source/Jost monitor plane differs from transfer plane")
    if float(scattering["observer"]) != observer:
        raise ValueError("Jost extraction plane differs from requested observer")
    if source["time"][-1] < time[-1]:
        raise ValueError("Source record must cover the requested waveform")
    if nfft is None:
        nfft = 1 << (max(4*len(time), len(source["time"]))-1).bit_length()
    parts = measured_ingoing_spectrum(source["time"], source["field"], source["derivative"], nfft, fft_dt=dt)
    w, raw_grid = parts["omega"], np.asarray(scattering["omega"])
    if raw_grid.dtype.kind not in "iuf":
        raise ValueError("Jost frequencies must be real numeric values")
    grid = np.asarray(raw_grid, float)
    if (grid.ndim != 1 or len(grid) < 8 or not np.isfinite(grid).all()
            or grid[0] <= 0 or np.any(np.diff(grid) <= 0) or grid[-1] >= np.pi/dt):
        raise ValueError("Strictly increasing positive Jost grid below Nyquist required")
    if interpolation not in ("pchip", "linear"):
        raise ValueError("interpolation must be pchip or linear")
    for name in ("R_barrier", "T_detector"):
        values = np.asarray(scattering[name])
        if values.shape != grid.shape or not np.isfinite(values).all():
            raise ValueError(f"Finite {name} values on the Jost grid required")
    rt = _runtime(notebook_path)
    rb = rt["interp_scattering"](w, grid, scattering["R_barrier"], "R", 2, interpolation)
    te = rt["interp_scattering"](w, grid, scattering["T_detector"], "T_E", 2, interpolation)
    global_R = rt["quantum_reflection"](w, kappa=.25, **p)
    local_R = global_R*np.exp(2j*w*surface)
    loop = rb*local_R*np.exp(-2j*w*surface)
    K = rt["surface_K"](w, rb, te, local_R, surface, observer)
    global_transfer = te*global_R*np.exp(1j*w*observer)/(1-rb*global_R)
    # This algebraic reference-plane identity is distinct from the spectral cutoff.
    plane_error = float(np.max(abs(K*np.exp(-1j*w*surface)-global_transfer)/(1+abs(global_transfer))))
    K[w > grid[-1]] = 0.
    echo = np.fft.irfft(np.conj(K)*parts["ingoing_fft"], nfft)[:len(time)]
    return {"correction": echo, "omega": w, "R_barrier": rb, "T_detector": te,
            "R_horizon_global": global_R, "R_horizon_local": local_R,
            "loop_function": loop, "K": K, "source_spectrum": parts,
            "nfft": int(nfft), "global_local_identity_error": plane_error}


def independent_td(*, dx=.1, surface=-40., observer=95., t_end=520.,
                   EplM=1e8, gamma=1., source_global=55., sigma=4., x_right=400.,
                   constant_reflection=None, notebook_path=None):
    """Paired absorbing/reflecting diamond PDE; real causal-history convolution.

    This adapts the frozen independent_rw_td for arbitrary declared hierarchy,
    Gaussian and outer boundary.  It never uses K or Jost coefficients.  The
    OWA impulse response still comes from its analytic frequency prescription;
    its negative-lag norm is reported rather than assumed to vanish.
    """
    h, duration, width = (_number(dx, "dx", positive=True), _number(t_end, "t_end", positive=True),
                          _number(sigma, "sigma", positive=True))
    surface, observer, right, center = (_number(surface, "surface"), _number(observer, "observer"),
                                         _number(x_right, "x_right"), _number(source_global, "source_global"))
    p = _params({"EplM": EplM, "gamma": gamma})
    if not surface < min(center-8*width, observer) < max(center+8*width, observer) < right:
        raise ValueError("Independent source and observer must be inside the PDE domain")
    nx = int(round((right-surface)/h))+1
    nt = int(np.ceil(duration/h))+1
    if nx < 12 or nx > 2_000_000 or nt > 100_000:
        raise ValueError("Independent PDE exceeds bounded grid/time allocation")
    if constant_reflection is not None:
        constant = _number(constant_reflection, "constant_reflection")
        if abs(constant) > 1:
            raise ValueError("Constant-reflection test requires abs(R)<=1")
    rt = _runtime(notebook_path)
    x = surface+h*np.arange(nx)
    index = int(round((observer-surface)/h))
    if abs(x[index]-observer) > 1e-9:
        raise ValueError("Independent detector must coincide with a spatial grid node")
    v = rt["rw_potential"](rt["r_from_schwarzschild_rstar"](x), 2, 1.)
    if v[0] >= 1e-8:
        raise ValueError("Independent wall must be in the V<1e-8 near-horizon overlap region")
    time = h*np.arange(nt)
    nker = 1 << (max(32*nt, 32768)-1).bit_length()
    if constant_reflection is None:
        w = 2*np.pi*np.fft.rfftfreq(nker, h)
        coeff = rt["quantum_local_reflection"](w, surface, kappa=.25, **p)
        kernel = np.fft.irfft(np.conj(coeff), nker)
        negative = float(np.linalg.norm(kernel[nker//2+1:])/np.linalg.norm(kernel))
        impulse = kernel[:nt].copy()
    else:
        negative, impulse = 0., np.array([constant])
    u = x-center
    q0 = np.exp(-.5*(u/width)**2)
    velocity = -u*q0/width**2
    accel = ((u/width)**2-1)*q0/width**2-v*q0
    q = np.stack([q0, q0])
    prev = q-h*velocity+.5*h*h*accel
    nxt = np.empty_like(q)
    fac = 1/(1+.5*h*h*v[1:-1])
    incoming, outgoing, delta = np.zeros(nt), np.zeros(nt), np.zeros(nt)
    for j in range(1, nt):
        nxt[:, 1:-1] = (q[:, :-2]+q[:, 2:])*fac-prev[:, 1:-1]
        nxt[:, -1] = q[:, -2]
        nxt[0, 0] = q[0, 1]
        incoming[j] = q[1, 1]-(outgoing[j-2] if j >= 2 else 0.)
        count = min(j+1, len(impulse))
        outgoing[j] = np.dot(impulse[:count], incoming[j-count+1:j+1][::-1])
        nxt[1, 0] = incoming[j]+outgoing[j]
        delta[j] = nxt[1, index]-nxt[0, index]
        prev, q, nxt = q, nxt, prev
    return {"time": time, "correction": delta, "dx": h, "dt": h,
            "negative_lag_kernel_L2_fraction": negative,
            "right_boundary_return_M": float(2*x[-1]-center-observer-8*width),
            "kernel_nfft": nker, "surface": surface, "observer": observer,
            "params": p, "numerics_evidence": "UNQUANTIFIED",
            "method": "paired diamond PDE with causal reflected-characteristic history"}


def run_quantum_horizon(params: Mapping[str, float] | None = None, *, dx=.1,
        t_end=520., cfl=.35, source_global=55., observer_global=95., sigma=4.,
        surface=-40., x_left=-250., x_right=400., sample_every=2,
        source_t_end=650., source_x_left=-400., scattering_step=.02,
        scattering_x_right=200., omega_grid=None, nfft=None, independent_dx=None,
        notebook_path=None):
    """Recompute QH and its measured source; return waveform plus scientific arrays.

    x_left=-140 gives the explicitly labelled frozen-domain replay.  Defaults
    use the safe new BH domain.  Raw waveform numerical evidence stays
    UNQUANTIFIED until linked resolution/systematic controls are evaluated.
    """
    p = _params({} if params is None else params)
    values = {k: _number(v, k, positive=k in ("dx", "t_end", "sigma", "source_t_end", "scattering_step"))
              for k, v in dict(dx=dx, t_end=t_end, cfl=cfl, source_global=source_global,
                observer_global=observer_global, sigma=sigma, surface=surface, x_left=x_left,
                x_right=x_right, source_t_end=source_t_end, source_x_left=source_x_left,
                scattering_step=scattering_step, scattering_x_right=scattering_x_right).items()}
    dx, t_end, sigma = values["dx"], values["t_end"], values["sigma"]
    surface, observer_global = values["surface"], values["observer_global"]
    if not 0 < cfl < .55 or not surface < 0 or not surface < source_global-8*sigma:
        raise ValueError("Invalid CFL or source/interface geometry")
    if not source_x_left < surface < observer_global < scattering_x_right or scattering_x_right > x_right:
        raise ValueError("Require source left edge < surface < observer < Jost right <= TD right")
    if source_t_end < t_end:
        raise ValueError("Source duration must cover requested waveform duration")
    if isinstance(sample_every, bool) or not isinstance(sample_every, int) or sample_every < 1:
        raise ValueError("sample_every must be a positive integer")
    if not x_left < source_global-8*sigma or not x_right > max(observer_global, source_global+8*sigma):
        raise ValueError("BH source/observer must lie inside the output domain")
    if (max((x_right-x_left)/dx, (x_right-source_x_left)/dx) > 2_000_000
            or source_t_end/(cfl*dx) > 5_000_000
            or (scattering_x_right-surface)/scattering_step > 2_000_000):
        raise ValueError("Requested solve exceeds bounded allocation/work limits")
    if t_end < 8*cfl*dx or sample_every > math.ceil(t_end/(cfl*dx)):
        raise ValueError("At least eight evolution steps and two output samples are required")
    if nfft is not None and (isinstance(nfft, bool) or not isinstance(nfft, (int, np.integer))
                             or nfft < math.ceil(source_t_end/(cfl*dx*sample_every))
                             or nfft % 2 or nfft > 2**23):
        raise ValueError("nfft must be even, cover the source and respect the allocation limit")
    if independent_dx is not None:
        _number(independent_dx, "independent_dx", positive=True)
    if omega_grid is not None and not np.isrealobj(omega_grid):
        raise ValueError("Jost frequencies must be real")
    grid = default_omega_grid() if omega_grid is None else np.asarray(omega_grid, float)
    if grid.ndim != 1 or len(grid) < 8 or not np.isfinite(grid).all() or grid[0] <= 0 or np.any(np.diff(grid) <= 0):
        raise ValueError("Jost grid must have at least eight finite increasing positive frequencies")
    if len(grid) > 100_000 or grid[-1] >= np.pi/(cfl*dx*sample_every):
        raise ValueError("Jost grid must fit the bounded allocation and output Nyquist frequency")
    rt = _runtime(notebook_path)
    if float(rt["rw_potential"](rt["r_from_schwarzschild_rstar"](surface))) >= 1e-8:
        raise ValueError("QH interface must lie in the V<1e-8 near-horizon overlap region")
    bh = run_td("BH", {}, dx=dx, t_end=t_end, cfl=cfl, source_global=source_global,
                observer_global=observer_global, sigma=sigma, x_left=x_left,
                x_right=x_right, sample_every=sample_every, notebook_path=notebook_path)
    time = bh["t"]
    source_dt = float(time[1]-time[0])
    sampling = source_dt*np.arange(int(np.floor(source_t_end/source_dt))+1)
    if len(sampling) < 3:
        raise ValueError("Source duration does not cover three output samples")
    xs = np.arange(source_x_left, x_right+.5*dx, dx)
    if not xs[4] < surface < xs[-5] or not xs[4] < source_global-8*sigma or not source_global+8*sigma < xs[-5]:
        raise ValueError("Source monitor/Gaussian must have interior stencil support")
    vs = rt["rw_potential"](rt["r_from_schwarzschild_rstar"](xs))
    ts, sr, sm = rt["evolve_wave"](xs, vs, t_max=max(source_t_end, float(sampling[-1])+.2),
        cfl=cfl, pulse_center=source_global, pulse_width=sigma,
        extraction_points={"observer": observer_global, "monitor": surface},
        left_bc="outgoing", sample_every=sample_every, derivative_points=("monitor",))
    source = {"time": sampling, "field": CubicSpline(ts, sr["monitor"], extrapolate=False)(sampling),
              "derivative": CubicSpline(ts, sr["monitor_dx"], extrapolate=False)(sampling),
              "native_time": ts, "native_field": sr["monitor"], "native_derivative": sr["monitor_dx"],
              "surface": surface, "metadata": sm}
    rb, tb, te = rt["barrier_coefficients_vectorized"](grid, 2, 1., xL=surface,
                 xR=scattering_x_right, step=scattering_step, extract=observer_global)
    scattering = {"omega": grid, "R_barrier": rb, "T_infinity": tb, "T_detector": te,
                  "surface": surface, "observer": observer_global}
    response = reconstruct(source, time, scattering, **p, surface=surface,
                           observer=observer_global, nfft=nfft, notebook_path=notebook_path)
    psi = bh["psi"]+response["correction"]
    w = response["omega"]
    boltzmann = float(np.max(abs(abs(response["R_horizon_global"])**2-np.exp(-8*np.pi*w))))
    source_return = min(source_global+surface-2*xs[0]-8*sigma,
                        2*xs[-1]-source_global-surface-8*sigma)
    flux_residual = float(np.max(abs(abs(rb)**2+abs(tb)**2-1)))
    checks = {**bh["checks"],
        "finite": {"passed": bool(np.isfinite(psi).all() and np.isfinite(response["K"]).all())},
        "source_stability": {"passed": bool(sm["stability_pass"])},
        "source_boundary": {"passed": bool(source_return > sampling[-1]),
                            "earliest_return_M": float(source_return), "record_end_M": float(sampling[-1])},
        "barrier_flux": {"passed": bool(flux_residual < 2e-6), "max_abs": flux_residual,
                         "tolerance": 2e-6, "scope": "Inherited real-axis flux check, not a phase-error bound"},
        "boltzmann_flux_identity": {"passed": boltzmann < 1e-12, "max_abs": boltzmann, "tolerance": 1e-12},
        "global_local_identity": {"passed": response["global_local_identity_error"] < 1e-12,
            "max_scaled_abs": response["global_local_identity_error"], "tolerance": 1e-12}}
    metadata = {**bh["metadata"], "model_family": "QH", "params": p, "solver_id": SOLVER_ID,
        "model_scope": "Nonrotating axial Schwarzschild with OWA effective horizon boundary; not a quantum spacetime",
        "x_interface_global": surface, "surface_role": "computational reference plane, not physical reflection depth",
        "baseline_provenance": rt.provenance, "numerics_evidence": "UNQUANTIFIED",
        "solver_status": "OK" if checks["finite"]["passed"] and checks["stability"]["passed"] and checks["source_stability"]["passed"] else "SOLVER_FAILED",
        "source_monitor_global": surface, "source_duration": float(sampling[-1]),
        "source_omega_c": response["source_spectrum"]["omega_c"],
        "source_omega_c_band": response["source_spectrum"]["omega_c_band"],
        "source_omega_c_definition": response["source_spectrum"]["omega_c_definition"],
        "scattering_step": scattering_step, "scattering_x_right": scattering_x_right,
        "scattering_flux_residual": flux_residual, "omega_min_integrated": float(grid[0]),
        "omega_max_integrated": float(grid[-1]), "nfft": response["nfft"],
        "low_frequency_treatment": "RW R magnitude limit and linear finite-detector T to zero at DC; unwrapped phase continuation",
        "dc_treatment": "real-axis incoming DC undefined, prescribed zero; no static response inferred",
        "high_frequency_treatment": "K=0 above integrated omega_max; requires cutoff audit",
        "reflection_planes": {"R_barrier": 0., "R_horizon_global": 0., "R_horizon_local": surface,
                              "T_detector_phase_removed_at": observer_global},
        "K_action": "physical outgoing correction at observer = K * physical ingoing field at monitor surface",
        "fft_action": "correction=irfft(conj(K_physical)*ingoing_fft_numpy); time convention exp(-i omega t)",
        "uncertainty_status": "UNQUANTIFIED; algebraic checks and replay do not bound waveform error",
        "modes_status": "NOT_COMPUTED", "classification_status": "NOT_COMPUTED"}
    result = {"t": time, "psi": psi, "x": bh["x"], "r": bh["r"], "V": bh["V"],
              "metadata": metadata, "checks": checks, "bh_reference": bh,
              "source": source, "scattering": scattering, "response": response,
              "independent": {"status": "NOT_COMPUTED"}}
    if independent_dx is not None:
        direct = independent_td(dx=independent_dx, surface=surface, observer=observer_global,
            t_end=float(time[-1])+.1, source_global=source_global, sigma=sigma,
            x_right=x_right, notebook_path=notebook_path, **p)
        comparison = CubicSpline(direct["time"], direct["correction"], extrapolate=False)(time)
        late = (time >= 190) & (time <= 500)
        norm = float(np.linalg.norm(comparison[late]))
        direct.update(status="COMPUTED_UNQUANTIFIED", correction_on_output_grid=comparison,
            comparison_window_M=[190., 500.],
            raw_late_relative_L2=(float(np.linalg.norm((response["correction"]-comparison)[late])/norm)
                                  if time[-1] >= 500 and norm > 0 else None))
        result["independent"] = direct
        if direct["raw_late_relative_L2"] is not None:
            checks["independent_reconstruction"] = {
                "passed": bool(direct["raw_late_relative_L2"] < .01
                               and direct["negative_lag_kernel_L2_fraction"] < .001),
                "raw_late_relative_L2": direct["raw_late_relative_L2"],
                "tolerance": .01, "negative_lag_tolerance": .001,
                "scope": "Inherited unaligned TD/FD overlap-region agreement, not a numerical error bound"}
    return result
