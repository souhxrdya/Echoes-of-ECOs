"""Canonical experiment and run identities. No inference from filenames."""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from numbers import Real


class FrozenParams(Mapping):
    """Small immutable, pickle-safe mapping of canonical scalar parameters.

    A frozen dataclass containing an ordinary dict is not immutable.  Copying
    into a tuple also prevents a caller-owned dict changing a run's identity
    after it has been scheduled, hashed, or written.
    """
    __slots__ = ("_items",)

    def __init__(self, value=()):
        items = dict(value)
        if any(not isinstance(k, str) or isinstance(v, bool)
               or not isinstance(v, Real) or not math.isfinite(v)
               for k, v in items.items()):
            raise ValueError("Model parameters must be finite real scalars with string keys")
        object.__setattr__(self, "_items", tuple(sorted((k, float(v)) for k, v in items.items())))

    def __setattr__(self, name, value):
        raise TypeError("Model parameters are immutable")

    def __getitem__(self, key):
        for name, value in self._items:
            if name == key:
                return value
        raise KeyError(key)

    def __iter__(self):
        return (name for name, _ in self._items)

    def __len__(self):
        return len(self._items)

    def __hash__(self):
        return hash(self._items)

    def __reduce__(self):
        return (type(self), (dict(self._items),))

    def __deepcopy__(self, memo):
        return self


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Experiment:
    M_ADM: float = 1.0
    ell: int = 2
    polarisation: str = "axial"
    field: str = "Q_RW"
    source_global_rstar: float = 55.0
    observer_global_rstar: float = 95.0
    sigma: float = 4.0
    amplitude: float = 1.0
    initial_direction: str = "left"
    F_in: float = 1.0
    t0: float = 0.0
    time_convention: str = "exp(-i*omega*t)"
    physical_fourier_convention: str = "integral psi(t)*exp(+i*omega*t) dt"
    pulse_shape: str = "exp(-(x-xs)^2/(2*sigma^2)); carrier=0"
    units: str = "G=c=M_ADM=1"

    def __post_init__(self):
        numeric = (self.M_ADM, self.source_global_rstar, self.observer_global_rstar,
                   self.sigma, self.amplitude, self.F_in, self.t0)
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
               for x in numeric):
            raise ValueError("Experiment scalars must be finite real numbers, not booleans")
        if isinstance(self.ell, bool) or not isinstance(self.ell, int):
            raise ValueError("ell must be an integer, not a boolean")
        if self.M_ADM != 1.0 or self.ell != 2 or self.polarisation != "axial":
            raise ValueError("First library milestone supports M=1, ell=2 axial only")
        if self.sigma <= 0 or not all(math.isfinite(x) for x in
                (self.source_global_rstar, self.observer_global_rstar, self.sigma)):
            raise ValueError("Nonfinite positions or invalid source width")
        if (self.amplitude, self.initial_direction, self.F_in, self.t0, self.field,
                self.time_convention, self.physical_fourier_convention, self.pulse_shape, self.units) != (
                1.0, "left", 1.0, 0.0, "Q_RW", "exp(-i*omega*t)",
                "integral psi(t)*exp(+i*omega*t) dt",
                "exp(-(x-xs)^2/(2*sigma^2)); carrier=0", "G=c=M_ADM=1"):
            raise ValueError("Unsupported initial data or convention")

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class RunSpec:
    model: str
    params: Mapping = field(default_factory=FrozenParams)
    dx: float = 0.1
    t_end: float = 520.0
    cfl: float = 0.35
    x_left: float | None = None
    x_right: float = 400.0
    sample_every: int = 2
    experiment: Experiment = field(default_factory=Experiment)

    def __post_init__(self):
        object.__setattr__(self, "params", FrozenParams(self.params))
        numeric = (self.dx, self.t_end, self.cfl, self.x_right)
        if self.x_left is not None:
            numeric += (self.x_left,)
        if any(isinstance(x, bool) or not isinstance(x, Real) or not math.isfinite(x)
               for x in numeric):
            raise ValueError("Run scalars must be finite real numbers, not booleans")
        if not isinstance(self.experiment, Experiment):
            raise ValueError("experiment must be an Experiment")
        if self.model not in {"BH", "GS", "WH"}:
            raise ValueError("Milestone 1 implements BH/GS/WH only; BS/QH are NOT_COMPUTED")
        expected = {"BH": set(), "GS": {"compactness"}, "WH": {"r0_over_M"}}[self.model]
        if set(self.params) != expected:
            raise ValueError(f"{self.model} parameters must be {sorted(expected)}")
        canonical_json(self.to_dict())
        if not (self.dx > 0 and self.t_end > 0 and 0 < self.cfl < .55):
            raise ValueError("Invalid resolution, duration or CFL")
        if (isinstance(self.sample_every, bool) or not isinstance(self.sample_every, int)
                or self.sample_every < 1):
            raise ValueError("sample_every must be a positive integer")
        if self.model == "GS" and not 0 < self.params["compactness"] < .5:
            raise ValueError("GS requires 0 < M/a < 1/2")
        if self.model == "WH" and not self.params["r0_over_M"] > 2:
            raise ValueError("WH requires r0/M > 2")

    @property
    def point_id(self):
        return self.model.lower() + "-" + digest({"model": self.model, "params": dict(self.params)})[:16]

    @property
    def record_id(self):
        return self.model.lower() + "-" + digest(self.to_dict())[:24]

    def to_dict(self):
        value = {f.name: getattr(self, f.name) for f in fields(self)}
        value["params"] = dict(self.params)
        value["experiment"] = asdict(self.experiment)
        return value

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["experiment"] = Experiment(**value["experiment"])
        return cls(**value)
