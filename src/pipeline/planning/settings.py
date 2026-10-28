import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class OpenTPSSettings:
    """
    Settings for OpenTPS proton planning, loaded from JSON with :meth:`load`.

    Required JSON values identify the machine, target, prescription, beam angles,
    and generation model; relative machine paths are resolved from the JSON
    file's directory. Each plan is optimized on the planning CT plus its selected
    generated scenarios, then recalculated on real fractions; prescription and
    OAR limits are total-plan Gy.

    Attributes
    ----------
    beam_model_path : Path
        OpenTPS/MCsquare beam data library file (BDL).

    ct_calibration_dir : Path
        OpenTPS CT calibration/scanner directory.

    target_label : int
        Positive integer label of the target in the structure maps.

    prescription_gy : float
        Positive target prescription in Gy for the complete plan.

    gantry_angles : list[float]
        One or more finite gantry angles in degrees, one per beam.

    model_name : str
        Non-empty identifier for the model that produced the selected generated
        CT scenarios. Stored with the resulting dose manifest.

    scenario_sets : dict[str, list[int] | Literal["all"]]
        Named collections of generated variant indices used for optimization.
        A list selects those positive indices; ``"all"`` selects every
        available generated variant for that patient. Defaults to a planning-CT-
        only ``nominal`` set and a ``robust`` set using all generated variants.
        Set names must start with a letter or digit and contain only letters,
        digits, underscores, or hyphens. Indices in a list must be unique.

    couch_angles : list[float] | None
        Couch angle in degrees for each gantry angle. Defaults to zero for every
        beam when omitted; otherwise, the list length must match
        ``gantry_angles``.

    oar_limits_gy : dict[int, float]
        Mapping from positive, non-target structure labels to maximum dose
        limits in Gy. Defaults to no OAR objectives.

    target_weight : float
        Positive weight for the target uniform-dose objective. Defaults to 1.0.

    oar_weight : float
        Positive weight for each OAR maximum-dose objective. Defaults to 1.0.

    spot_spacing_mm : float
        Positive lateral spot spacing in millimetres. Defaults to 5.0.

    layer_spacing_mm : float
        Positive energy-layer spacing in millimetres. Defaults to 5.0.

    target_margin_mm : float
        Non-negative margin in millimetres used when designing the target spot
        geometry. Defaults to 5.0.

    beamlet_primaries : int
        Positive MCsquare particle count for beamlet calculation. More primaries
        reduce Monte Carlo noise at greater computational cost. Defaults to
        50,000.

    dose_primaries : int
        Positive MCsquare particle count for final dose recalculation. Defaults
        to 1,000,000.
        
    max_iterations : int
        Positive upper bound on worst-case optimizer iterations. Defaults to
        200.

    Raises
    ------
    FileNotFoundError
        If the beam model or CT calibration directory does not exist.
    ValueError
        If required labels, angles, scenario selections, or numeric settings are
        invalid. See :meth:`validate` for the checks performed.
    """

    beam_model_path: Path
    ct_calibration_dir: Path
    target_label: int
    prescription_gy: float
    gantry_angles: list[float]
    model_name: str
    scenario_sets: dict[str, list[int] | Literal["all"]] = field(
        default_factory=lambda: {"nominal": [], "robust": "all"}
    )
    couch_angles: list[float] | None = None
    oar_limits_gy: dict[int, float] = field(default_factory=dict)
    target_weight: float = 1.0
    oar_weight: float = 1.0
    spot_spacing_mm: float = 5.0
    layer_spacing_mm: float = 5.0
    target_margin_mm: float = 5.0
    beamlet_primaries: int = 50_000
    dose_primaries: int = 1_000_000
    max_iterations: int = 200

    @classmethod
    def load(cls, path: Path) -> "OpenTPSSettings":
        with path.open(encoding="utf-8") as stream:
            values = json.load(stream)
        for key in ("beam_model_path", "ct_calibration_dir"):
            value = Path(values[key])
            values[key] = (path.parent / value).resolve() if not value.is_absolute() else value
        values["oar_limits_gy"] = {int(label): limit for label, limit in values.get("oar_limits_gy", {}).items()}
        settings = cls(**values)
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.beam_model_path.is_file():
            raise FileNotFoundError(f"OpenTPS beam model not found: {self.beam_model_path}")
        if not self.ct_calibration_dir.is_dir():
            raise FileNotFoundError(f"OpenTPS CT calibration directory not found: {self.ct_calibration_dir}")
        if not isinstance(self.target_label, int) or self.target_label <= 0:
            raise ValueError("OpenTPS target_label must be an explicit positive structure label")
        if not self.model_name.strip():
            raise ValueError("OpenTPS model_name must identify the model that generated the selected CTs")
        positive = (
            "prescription_gy",
            "target_weight",
            "oar_weight",
            "spot_spacing_mm",
            "layer_spacing_mm",
            "beamlet_primaries",
            "dose_primaries",
            "max_iterations",
        )
        for name in positive:
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"OpenTPS {name} must be finite and greater than zero")
        for name in ("beamlet_primaries", "dose_primaries", "max_iterations"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"OpenTPS {name} must be an integer")
        if not math.isfinite(self.target_margin_mm) or self.target_margin_mm < 0:
            raise ValueError("OpenTPS target_margin_mm must be finite and non-negative")
        if not self.gantry_angles or not all(math.isfinite(angle) for angle in self.gantry_angles):
            raise ValueError("OpenTPS gantry_angles must contain finite angles")
        if self.couch_angles is None:
            self.couch_angles = [0.0] * len(self.gantry_angles)
        if len(self.couch_angles) != len(self.gantry_angles) or not all(map(math.isfinite, self.couch_angles)):
            raise ValueError("OpenTPS couch_angles must contain one finite angle per beam")
        if not self.scenario_sets:
            raise ValueError("OpenTPS scenario_sets cannot be empty")
        for name, indices in self.scenario_sets.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
                raise ValueError(f"Invalid scenario-set name: {name!r}")
            if indices == "all":
                continue
            if not isinstance(indices, list) or any(type(index) is not int or index < 1 for index in indices):
                raise ValueError(f"Scenario set {name} must be 'all' or a list of positive variant indices")
            if len(indices) != len(set(indices)):
                raise ValueError(f"Duplicate variant indices in scenario set {name}")
        for label, limit in self.oar_limits_gy.items():
            if label <= 0 or label == self.target_label or not math.isfinite(limit) or limit < 0:
                raise ValueError(f"Invalid OAR label/limit: {label}={limit}")
