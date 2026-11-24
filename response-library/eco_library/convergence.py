"""Observed refinement evidence, not certified continuum-error bounds."""
from numbers import Real
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import CubicSpline

WINDOWS = {"prompt": (100., 190.), "ringdown": (150., 190.),
           "late": (190., 500.), "full": (100., 500.)}


def _real_vector(value, name, minimum=2):
    """Validate before casting: complex/string/bool data are not real fields."""
    raw = np.asarray(value)
    if (raw.ndim != 1 or len(raw) < minimum or raw.dtype.kind not in "iuf"
            or not np.isfinite(raw).all()):
        raise ValueError(f"{name} must be a finite real numeric vector")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} exceeds float64 range")
    return result


def common_series(records, dt=.07):
    if (not records or isinstance(dt, (bool, np.bool_)) or not isinstance(dt, Real)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("Nonempty records and finite positive common dt required")
    validated = []
    for record in records:
        native = _real_vector(record["t"], "time")
        field = _real_vector(record["psi"], "waveform")
        if native.shape != field.shape or not np.all(np.diff(native) > 0):
            raise ValueError("Matching waveform shape and strictly increasing time required")
        validated.append((native, field))
    start = max(t[0] for t, _ in validated)
    end = min(t[-1] for t, _ in validated)
    if end <= start:
        raise ValueError("Records have no common time support")
    t = start + np.arange(int(np.floor((end-start)/dt))+1)*dt
    if len(t) < 3:
        raise ValueError("Too few common samples for numerical evidence")
    ys = [CubicSpline(native, field, extrapolate=False)(t) for native, field in validated]
    if not all(np.isfinite(y).all() for y in ys):
        raise ValueError("Interpolation outside common time support")
    return t, ys


def assess(t, ys, hs, *, expected_order, windows=WINDOWS, order_tolerance=.5):
    """Coarse-to-fine, exactly three nested levels; estimates are conditional."""
    if len(ys) != 3 or len(hs) != 3:
        raise ValueError("First milestone requires exactly three levels")
    t = _real_vector(t, "time", minimum=3)
    ys = [_real_vector(y, "waveform", minimum=3) for y in ys]
    if (not np.all(np.diff(t) > 0) or any(y.shape != t.shape for y in ys)):
        raise ValueError("Finite aligned waveforms and increasing time required")
    if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
            or not np.isfinite(v) for v in (expected_order, order_tolerance))
            or expected_order <= 0 or not 0 <= order_tolerance < expected_order):
        raise ValueError("Invalid declared order or order band")
    hs = _real_vector(hs, "mesh spacings", minimum=3)
    ratios = hs[:-1]/hs[1:]
    if not np.isfinite(hs).all() or not np.all(hs > 0) or not np.all(ratios > 1) or not np.allclose(ratios, ratios[0], rtol=1e-8, atol=0):
        raise ValueError("Constant actual refinement ratio required; requested dx is insufficient")
    out = {}
    for name, (lo, hi) in windows.items():
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise ValueError("Invalid convergence window")
        if t[0] > lo or t[-1] < hi:
            out[name] = {"state": "UNQUANTIFIED", "reason": "RECORD_TOO_SHORT"}
            continue
        mask = (t >= lo) & (t <= hi)
        tw = t[mask]
        if len(tw) < 3:
            raise ValueError("Too few samples in convergence window")
        norm = lambda y: float(np.sqrt(trapezoid(np.asarray(y)[mask]**2, tw)))
        dcm, dmf = norm(ys[0]-ys[1]), norm(ys[1]-ys[2])
        scale = norm(ys[2])
        entry = {"d_coarse_medium_L2": dcm, "d_medium_fine_L2": dmf,
                 "fine_L2": scale, "actual_h": hs.tolist(), "ratio": float(ratios[0]),
                 "expected_order": expected_order, "order_tolerance": order_tolerance,
                 "p_observed": None, "fine_error_estimate_L2": None,
                 "fine_relative_error_estimate": None, "is_error_bound": False}
        if max(dcm, dmf) <= max(1e-300, 1e-12*scale):
            entry.update(state="AGREEMENT_AT_FLOOR", reason="Order/error not identifiable; no zero uncertainty inferred")
        elif dmf == 0 or dcm <= dmf:
            entry.update(state="ORDER_NOT_CONFIRMED", reason="No decreasing nonzero refinement differences")
        else:
            p = float(np.log(dcm/dmf)/np.log(ratios[0]))
            entry["p_observed"] = p
            if abs(p-expected_order) <= order_tolerance:
                est = float(1.25*dmf/(ratios[0]**p-1))
                entry.update(state="OBSERVED_ORDER_CONSISTENT", fine_error_estimate_L2=est,
                             fine_relative_error_estimate=est/scale if scale > 0 else None,
                             reason="Conditional Richardson estimate; independent systematics not bounded")
            else:
                entry.update(state="ORDER_NOT_CONFIRMED", reason="Observed order outside declared convention band")
        out[name] = entry
    return out
