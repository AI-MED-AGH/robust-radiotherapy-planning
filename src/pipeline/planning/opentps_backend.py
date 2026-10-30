import copy
from pathlib import Path
from typing import Any

import numpy as np
from opentps.core.data.images import CTImage, ROIMask
from opentps.core.data.plan import ProtonPlanDesign
from opentps.core.io.dicomIO import writeRTPlan
from opentps.core.io.mcsquareIO import readBDL
from opentps.core.io.scannerReader import readScanner
from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import MCsquareDoseCalculator
from opentps.core.processing.planOptimization.objectives.doseFidelity import DoseFidelity
from opentps.core.processing.planOptimization.objectives.dosimetricObjectives._DMax import DMax
from opentps.core.processing.planOptimization.objectives.dosimetricObjectives._DUniform import DUniform
from opentps.core.processing.planOptimization.objectives.robustFunctions.robustWorstCase import RobustWorstCase
from opentps.core.processing.planOptimization.objectives.weightedSum import WeightedSum
from scipy.optimize import minimize

from src.pipeline.planning.geometry import dose_to_nifti, opentps_image
from src.pipeline.planning.settings import OpenTPSSettings


class OpenTPSBackend:
    """
    Build proton plans and optimize common spot MUs across selected anatomies.

    Each anatomy has its own beamlet matrix and target/OAR masks, while all
    scenarios retain the same spot columns. Optimization changes monitor units
    (MUs), which control spot intensity, without redesigning spot geometry.
    The objective is the largest anatomy-specific weighted penalty.

    MCsquare beamlet rows and objective masks use the same flipped-XY,
    Fortran-order voxel convention. The geometry helpers handle the separate
    NIfTI RAS-to-OpenTPS LPS conversion.

    Attributes
    ----------
    settings : OpenTPSSettings
        Planning, objective, particle-count, and optimization settings.

    calibration : Any
        Scanner calibration loaded for plan design and MCsquare calculation.

    calculator : Any
        Shared MCsquare calculator containing the beam model and simulation
        directory. Its scoring grid and particle count change between calls;
        one backend instance must not run concurrent calculations.

    Notes
    -----
    OpenTPS components are imported with this module. Keep this backend in
    the dedicated worker process to avoid mixing its native dependencies with
    PyTorch on Windows. Callers supply validated settings and axis-aligned
    images with masks on their corresponding anatomical grids.
    """

    def __init__(self, settings: OpenTPSSettings, simulation_dir: Path) -> None:
        """
        Configure an MCsquare dose calculator with scanner and beam data.

        Parameters
        ----------
        settings : OpenTPSSettings
            Previously validated settings containing machine calibration and beam
            model paths. This constructor does not call ``settings.validate``.

        simulation_dir : Path
            Working directory assigned to MCsquare as an absolute path. This
            constructor does not explicitly create the directory.

        Returns
        -------
        None
            Stores the settings, image classes, scanner calibration, and
            configured calculator on the new backend instance.
        """

        self.settings = settings
        self.CTImage = CTImage
        self.ROIMask = ROIMask
        self.calibration = readScanner(str(settings.ct_calibration_dir))
        self.calculator = MCsquareDoseCalculator()
        self.calculator.ctCalibration = self.calibration
        self.calculator.beamModel = readBDL(str(settings.beam_model_path))
        self.calculator.simulationDirectory = str(simulation_dir.resolve())

    def build_plan(self, ct: Any, target: Any) -> Any:
        """
        Design a common proton spot geometry covering the supplied target.

        Beam angles, lateral spot spacing, energy-layer spacing, and target margin
        come from the settings. The caller may provide a union of scenario targets
        so the shared spot geometry covers their combined extent.

        Parameters
        ----------
        ct : Any
            OpenTPS CTImage in LPS coordinates used for plan design.

        target : Any
            OpenTPS ROIMask describing the target used to place spots, with
            geometry compatible with ``ct``.

        Returns
        -------
        plan : Any
            OpenTPS proton plan after ``simplify(threshold=None)``. Its spot
            intensities have not yet been optimized by this backend.

        Raises
        ------
        ValueError
            If the resulting plan contains no proton spots.
        """

        design = ProtonPlanDesign()
        design.ct = ct
        design.calibration = self.calibration
        design.targetMask = target
        design.gantryAngles = self.settings.gantry_angles
        design.couchAngles = self.settings.couch_angles
        design.spotSpacing = self.settings.spot_spacing_mm
        design.layerSpacing = self.settings.layer_spacing_mm
        design.targetMargin = self.settings.target_margin_mm
        plan = design.buildPlan()
        plan.simplify(threshold=None)
        if plan.numberOfSpots == 0:
            raise ValueError("OpenTPS could not place any proton spots for the selected target")
        return plan

    def set_grid(self, ct: Any) -> None:
        # Explicitly reset all three: a plan design otherwise retains the
        # planning grid when computeDose is called with a different real CT.
        """
        Reset the calculator's dose-scoring geometry to the current CT.

        Resetting origin, spacing, and grid size prevents a calculation on a new
        anatomy from retaining the previous CT's scoring grid.

        Parameters
        ----------
        ct : Any
            OpenTPS image supplying ``origin``, ``spacing``, and ``gridSize``.

        Returns
        -------
        None
            Updates the shared calculator in place; no dose is calculated.
        """

        self.calculator.scoringOrigin = ct.origin
        self.calculator.scoringVoxelSpacing = ct.spacing
        self.calculator.scoringGridSize = ct.gridSize

    def scenario(self, image: Any, masks: dict[int, Any], plan: Any, name: str) -> Any:
        """
        Calculate beamlets and assemble the objective for one anatomy.

        MCsquare calculates a sparse matrix mapping common spot MUs to voxel doses
        for this CT. The target receives a uniform-prescription penalty; each
        configured OAR receives an overdose penalty above its dose limit. These
        are weighted soft penalties, not guaranteed dose constraints.

        Parameters
        ----------
        image : nibabel.spatialimages.SpatialImage
            Axis-aligned CT in RAS+ millimetres with HU intensities.

        masks : dict[int, nibabel.spatialimages.SpatialImage]
            Anatomy-specific binary masks keyed by structure label. Must include
            the configured target and every configured OAR, aligned with ``image``.

        plan : Any
            OpenTPS proton plan defining the same ordered spots used for every
            scenario passed to ``optimize``.

        name : str
            Identifier assigned to the OpenTPS CT and used in diagnostic errors.

        Returns
        -------
        fidelity : Any
            OpenTPS DoseFidelity with a voxel-by-spot beamlet matrix, weighted
            structure objectives, and direct MU weights (``xSquared=False``).
            Gradient multiplication is restricted to the union of objective masks.
            The calculator retains this CT's scoring grid and beamlet primaries.

        Raises
        ------
        KeyError
            If a required structure label is absent from ``masks``.

        ValueError
            If the beamlet matrix has an unexpected shape, non-finite or negative
            entries, its origin or spacing differs from the CT, or a required
            converted structure mask is empty. Shape checks alone cannot detect
            every possible permutation of beamlet rows or spot columns.
        """

        # Convert the NIfTI anatomy into the image representation expected by OpenTPS
        ct = opentps_image(image, self.CTImage, name=name)
        self.set_grid(ct)
        # Beamlets describe each spot separately, before choosing its final intensity
        self.calculator.nbPrimaries = self.settings.beamlet_primaries
        beamlets = self.calculator.computeBeamlets(ct, plan)
        matrix = beamlets.toSparseMatrix()
        # Rows represent CT voxels; columns must match the common plan's ordered spots
        # Reject invalid dose contributions before they enter the optimization
        if (
            matrix.shape != (int(np.prod(ct.gridSize)), plan.numberOfSpots)
            or not np.isfinite(matrix.data).all()
            or (matrix.data < 0).any()
        ):
            raise ValueError(f"Invalid or reordered beamlet matrix for {name}")
        # A correctly sized matrix must also describe the same physical grid as the CT
        if not np.allclose(beamlets.doseOrigin, ct.origin) or not np.allclose(beamlets.doseSpacing, ct.spacing):
            raise ValueError(f"MCsquare beamlet geometry differs from CT geometry for {name}")
        objectives = WeightedSum()
        # Insert the target first: optimize() later uses the first objective to scale initial MUs
        limits = {self.settings.target_label: self.settings.prescription_gy, **self.settings.oar_limits_gy}
        for label, limit in limits.items():
            # Use this anatomy's structure locations, which may differ between scenarios
            mask = opentps_image(masks[label], self.ROIMask, name=f"label_{label}")
            if not np.any(mask.imageArray):
                raise ValueError(f"Required optimization structure {label} is empty in {name}")
            # Target: penalize deviation from prescription. OAR: penalize exceeding its limit
            objective_class = DUniform if label == self.settings.target_label else DMax
            weight = self.settings.target_weight if label == self.settings.target_label else self.settings.oar_weight
            objective = objective_class(mask, limit, weight=weight)
            # Flatten the structure mask using the voxel ordering expected by the beamlet matrix
            objective._updateMaskVec(ct.spacing, ct.gridSize, ct.origin)
            objectives.functionList.append(objective)
        # For candidate MUs, compute dose = matrix @ MUs and evaluate the structure penalties
        # xSquared=False means optimization variables are MUs directly, rather than their square roots
        fidelity = DoseFidelity(matrix, xSquared=False)
        fidelity.function = objectives
        # This avoids constructing a large sparse derivative matrix per ROI
        fidelity.croppedMultiplication = True
        # Only voxels belonging to at least one objective are needed for the gradient
        fidelity.unionMaskVec = np.logical_or.reduce([item.maskVec for item in objectives.functionList])
        return fidelity

    def optimize(self, plan: Any, scenarios: list[Any]) -> tuple[Any, dict[str, Any]]:
        """
        Optimize non-negative spot MUs against the worst anatomical penalty.

        The largest weighted scenario penalty is minimized with L-BFGS-B. Initial
        MUs are equal and scaled so the mean target dose in the first scenario
        is near prescription. Spot geometry and ordering remain fixed.

        Parameters
        ----------
        plan : Any
            Base OpenTPS proton plan whose spot order matches all scenario
            beamlet matrices. The input plan is not modified.

        scenarios : list[Any]
            Non-empty list of DoseFidelity objects returned by ``scenario``.
            The first should represent the planning anatomy and must have its
            target objective first, as constructed by ``scenario``.

        Returns
        -------
        optimized : Any
            Deep copy of the input plan with optimized ``spotMUs``. No final
            simplification is performed, preserving beamlet column correspondence.

        optimization : dict[str, Any]
            Solver report containing ``objective`` (final worst-case penalty),
            ``iterations`` (iteration count), and ``message`` (termination text).

        Raises
        ------
        ValueError
            If the initial mean target dose is non-finite or non-positive.

        RuntimeError
            If the solver reports failure or returns non-finite MUs or objective.

        Notes
        -----
        Scenario objects are evaluated in place and retain dose/objective state.
        Solver success does not guarantee that every clinical dose limit is met;
        OAR limits and target prescription are represented by soft penalties.
        """

        # Every scenario receives the same MU vector; the largest scenario penalty drives optimization
        objective = RobustWorstCase(nScenarios=len(scenarios))
        objective.robustFunctions = scenarios
        # Begin with equal intensity for every spot in the shared geometry
        initial = np.ones(plan.numberOfSpots, dtype=np.float64)
        # Start near prescription rather than arbitrary machine MU units
        # eval() stores the first scenario's dose, allowing us to measure its mean target dose
        scenarios[0].eval(initial)
        target = scenarios[0].function.functionList[0].maskVec
        mean = float(np.mean(scenarios[0].dose[target]))
        if not np.isfinite(mean) or mean <= 0:
            raise ValueError("OpenTPS beamlets deliver no dose to the target")
        # Dose scales linearly with MUs, so this factor brings the initial mean to prescription
        initial *= self.settings.prescription_gy / mean

        def value_and_gradient(weights: Any) -> tuple[float, Any]:
            """
            Evaluate the worst-case penalty before requesting its gradient.

            Evaluation updates cached scenario doses and selects the current worst
            scenario. The subsequent gradient uses that state, so call order matters.

            Parameters
            ----------
            weights : numpy.ndarray
                Candidate spot MU vector supplied by the optimizer.

            Returns
            -------
            value : float
                Largest weighted penalty across the captured anatomical scenarios.

            gradient : numpy.ndarray
                Float64 gradient of the currently selected worst scenario with respect
                to spot MUs, as supplied by OpenTPS.
            """

            # Recalculate scenario penalties and select the worst anatomy for these candidate MU
            value = float(objective.eval(weights))
            # Evaluate first: grad() uses the worst scenario and dose state selected by eval()
            # Return both together so SciPy receives a matching penalty and gradient
            return value, objective.grad(weights).astype(np.float64)

        # Adjust intensities only; beam directions, spot locations, and energies stay fixed
        # jac=True tells SciPy that the callback returns both the value and its gradient
        result = minimize(
            value_and_gradient,
            initial,
            jac=True,
            method="L-BFGS-B",
            # A spot can be switched off (zero MU), but cannot deliver negative radiation
            bounds=[(0.0, None)] * len(initial),
            options={"maxiter": self.settings.max_iterations, "maxls": 50, "ftol": 1e-8},
        )
        # Do not export a plan if the solver failed or produced invalid numerical results
        if not result.success or not np.isfinite(result.x).all() or not np.isfinite(result.fun):
            raise RuntimeError(f"OpenTPS worst-case optimization failed: {result.message}")
        # Preserve the original plan and put the optimized intensities into an independent copy
        optimized = copy.deepcopy(plan)
        optimized.spotMUs = result.x
        # Do not simplify here: spot columns must stay identical for every set
        return optimized, {
            "objective": float(result.fun),
            "iterations": int(result.nit),
            "message": str(result.message),
        }

    def calculate_dose(self, image: Any, plan: Any, path: Path) -> None:
        """
        Recalculate a fixed plan on one CT and save the dose as NIfTI.

        Parameters
        ----------
        image : nibabel.spatialimages.SpatialImage
            Axis-aligned CT in RAS+ millimetres with HU intensities.

        plan : Any
            OpenTPS proton plan with the spot MUs to evaluate. This method does
            not reoptimize the plan for the supplied CT.

        path : Path
            Output NIfTI dose path. Its parent directory must already exist.

        Returns
        -------
        None
            Runs MCsquare with ``settings.dose_primaries`` and writes dose values
            in Gy with a NIfTI world affine. The shared calculator retains this
            CT's scoring grid and final-dose particle count.

        Raises
        ------
        ValueError
            If the calculated dose fails ``dose_to_nifti`` validation, including
            non-finite, negative, entirely zero, or rotated dose grids.

        OSError
            If the output file cannot be written.
        """

        ct = opentps_image(image, self.CTImage, name=path.stem)
        self.set_grid(ct)
        self.calculator.nbPrimaries = self.settings.dose_primaries
        dose = self.calculator.computeDose(ct, plan)
        dose_to_nifti(dose, path)

    def save_plan(self, plan: Any, path: Path) -> None:
        """
        Export an OpenTPS proton plan as a DICOM RT Plan file.

        Parameters
        ----------
        plan : Any
            OpenTPS proton plan to export, normally returned by ``optimize``.

        path : Path
            Expected output path with a ``.dcm`` suffix. The exporter receives
            its parent directory and filename stem.

        Returns
        -------
        None
            Writes the plan through the OpenTPS DICOM exporter and checks that
            the expected file exists. File existence does not validate contents.

        Raises
        ------
        RuntimeError
            If the expected file is absent after export.
        """

        writeRTPlan(plan, str(path.parent), outputFilename=path.stem)
        if not path.is_file():
            raise RuntimeError(f"OpenTPS did not export the plan to {path}")
