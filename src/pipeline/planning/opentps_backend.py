"""Build and optimize proton plans with OpenTPS and recalculate doses with MCsquare."""

import copy
import logging
import os
import platform
import subprocess
from importlib.metadata import version
from inspect import getfile
from pathlib import Path
from typing import Any

import numpy as np
from opentps.core.data.images import CTImage, ROIMask  # type: ignore[import-untyped]
from opentps.core.data.MCsquare import MCsquareConfig  # type: ignore[import-untyped]
from opentps.core.data.plan import ProtonPlanDesign  # type: ignore[import-untyped]
from opentps.core.io.dicomIO import writeRTPlan  # type: ignore[import-untyped]
from opentps.core.io.mcsquareIO import _read_sparse_header, readBDL  # type: ignore[import-untyped]
from opentps.core.io.scannerReader import readScanner  # type: ignore[import-untyped]
from opentps.core.processing.doseCalculation.protons.mcsquareDoseCalculator import (  # type: ignore[import-untyped]
    MCsquareDoseCalculator,
)
from opentps.core.processing.planOptimization.objectives.doseFidelity import (  # type: ignore[import-untyped]
    DoseFidelity,
)
from opentps.core.processing.planOptimization.objectives.dosimetricObjectives._DMax import (  # type: ignore[import-untyped]
    DMax,
)
from opentps.core.processing.planOptimization.objectives.dosimetricObjectives._DUniform import (  # type: ignore[import-untyped]
    DUniform,
)
from opentps.core.processing.planOptimization.objectives.robustFunctions.robustWorstCase import (  # type: ignore[import-untyped]
    RobustWorstCase,
)
from opentps.core.processing.planOptimization.objectives.weightedSum import WeightedSum  # type: ignore[import-untyped]
from scipy.optimize import minimize
from scipy.sparse import csc_matrix

from src.pipeline.planning.geometry import dose_to_nifti, opentps_image
from src.pipeline.planning.mcsquare_sparse_file import read_objective_matrix
from src.pipeline.planning.settings import OpenTPSSettings

logger = logging.getLogger(__name__)


class _BundledMCsquareConfig(MCsquareConfig):  # type: ignore[misc]
    """
    Serialize OpenTPS settings with the bundled MCsquare template.

    The calculator supplies the settings and can still override them for each
    calculation mode through the usual ``MCsquareConfig`` interface.
    """

    def __init__(self, config: dict[str, Any], template: str) -> None:
        """
        Wrap an existing OpenTPS configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Settings dictionary from OpenTPS, shared with this wrapper so that
            subsequent beamlet and dose settings are included in the output.
        template : str
            Release template with uppercase placeholders for MCsquare settings.
        """

        self.config = config
        self._template = template

    def mcsquareFormatted(self) -> str:
        """
        Render the current settings as MCsquare's configuration file text.

        Scalar values are written as strings. Vector settings such as scoring
        origin are written as space-separated components.

        Returns
        -------
        str
            MCsquare configuration text with all template placeholders filled.

        Raises
        ------
        KeyError
            If the configuration has no value for a template placeholder.
        """

        values = {
            key.upper(): " ".join(map(str, value)) if np.ndim(value) > 0 else str(value)
            for key, value in self.config.items()
        }
        return self._template.format_map(values)


class _OpenTPS301Calculator(MCsquareDoseCalculator):  # type: ignore[misc]
    """
    Adapt the OpenTPS 3.0.1 calculator for the planning worker.

    The adapter uses the release template when the installed wheel omits it,
    aligns the cached dose path with the worker's simulation directory, and
    imports only the beamlet rows used by the optimization objectives.
    """

    def __init__(self, simulation_dir: Path, mcsquare_threads: int) -> None:
        """
        Bind the calculator to a patient-specific simulation directory.

        Parameters
        ----------
        simulation_dir : Path
            Absolute path to the worker's simulation directory. The calculator
            creates its MCsquare working directory inside this path.
        mcsquare_threads : int
            Number of MCsquare threads, or 0 to use all available threads.

        Raises
        ------
        OSError
            If a working directory cannot be created or the fallback template
            is needed and cannot be read.
        """

        super().__init__()
        self._mcsquare_threads = mcsquare_threads
        self.simulationDirectory = str(simulation_dir)
        # OpenTPS caches this output path before the simulation directory is assigned
        self._doseFilePath = str(Path(self._workDir) / "Dose.mhd")
        self._bundled_template: str | None = None
        installed_template = Path(getfile(MCsquareConfig)).resolve().with_name("ConfigTemplate.txt")
        if not installed_template.is_file():
            # The 3.0.1 wheel omits this file. Load the release template before plan design
            bundled_template = Path(__file__).resolve().parents[3] / "opentps_3_0_1" / installed_template.name
            self._bundled_template = bundled_template.read_text(encoding="utf-8")

    @property
    def _generalMCsquareConfig(self) -> Any:
        """
        Return the base config used by OpenTPS's calculation modes.

        Set the thread count for all calculation modes. If the installed
        template is missing, wrap the config to use the bundled template.

        Returns
        -------
        MCsquareConfig
            Configuration with the worker's thread count and a template
            suitable for writing MCsquare input files.
        """

        config = super()._generalMCsquareConfig
        config["Num_Threads"] = self._mcsquare_threads
        if self._bundled_template is None:
            return config
        return _BundledMCsquareConfig(config.config, self._bundled_template)

    def compute_objective_beamlets(self, ct: Any, plan: Any, rows: np.ndarray[Any, Any], name: str) -> csc_matrix:
        """
        Run the MCsquare executable and import the rows used by the objectives.

        The calculator writes a unit-MU copy of the plan, clears its simulation
        output directory, and runs MCsquare on the configured scoring grid.
        The binary importer keeps only the requested voxel rows.

        Parameters
        ----------
        ct : opentps.core.data.images.CTImage
            CT whose origin, spacing, and dimensions match the calculator's
            scoring grid.
        plan : opentps.core.data.plan.ProtonPlan
            Shared spot geometry used across the anatomical scenarios.
        rows : numpy.ndarray
            Unique flattened voxel indices in ascending MCsquare order.
        name : str
            Anatomy identifier used in log messages and errors.

        Returns
        -------
        scipy.sparse.csc_matrix
            Float32 dose per MU with one row per selected voxel and one column
            per plan spot, in spot order.

        Raises
        ------
        subprocess.CalledProcessError
            If the MCsquare executable exits with a nonzero status.
        OSError
            If an input or output file cannot be accessed or the executable
            cannot be started.
        KeyError
            If required metadata is missing from the sparse dose header.
        ValueError
            If plan preparation changes the spot count, the sparse dose header
            disagrees with the plan or scoring grid, or the binary is invalid.
        """

        n_voxels = int(np.prod(ct.gridSize))
        n_spots = plan.numberOfSpots

        # Prepare the executable's inputs as in OpenTPS computeBeamlets
        self.ct = ct
        self._plan = copy.deepcopy(plan)
        self._plan.spotMUs = np.ones(self._plan.spotMUs.shape)
        self._plan.simplify(threshold=None)
        if self._plan.numberOfSpots != n_spots:
            raise ValueError(f"MCsquare plan preparation changed the number of spots for {name}")
        self._roi = None
        self._config = self._beamletComputationConfig
        self._writeFilesToSimuDir()
        self._cleanDir(self._workDir)
        simulation_dir = Path(self._mcsquareSimuDir)
        command = [str(simulation_dir / "MCsquare_win.exe")] if platform.system() == "Windows" else ["bash", "MCsquare"]
        subprocess.run(
            command,
            cwd=simulation_dir,
            env={**os.environ, "MCsquare_Materials_Dir": self._materialFolder},
            check=True,
        )
        logger.info("MCsquare scoring finished for %s. Reading %d objective voxels", name, len(rows))
        header = _read_sparse_header(self._sparseDoseFilePath)
        if "Beamlet" not in header["SimulationMode"] or header["NbrSpots"] != n_spots:
            raise ValueError(f"Invalid MCsquare beamlet file header for {name}")
        if not np.array_equal(header["ImageSize"], ct.gridSize) or not np.allclose(header["VoxelSpacing"], ct.spacing):
            raise ValueError(f"MCsquare beamlet geometry differs from CT geometry for {name}")
        return read_objective_matrix(
            Path(header["Binary_file"]), n_voxels, rows, self._beamletRescaling(), scratch_dir=simulation_dir
        )


class _ObjectiveDoseFidelity(DoseFidelity):  # type: ignore[misc]
    """
    Evaluate dose fidelity using only the voxels covered by the objectives.

    The beamlet matrix and structure masks share the same compact voxel
    indices. The gradient uses the transpose of the dose derivative to map
    the weighted voxel penalties back to the spot weights.

    Attributes
    ----------
    beamlets : scipy.sparse.csc_matrix
        Dose per MU for each selected voxel and plan spot.
    function : WeightedSum
        Weighted structure objectives with masks indexed to the matrix rows.
        The planning backend assigns this after construction.
    """

    def computeFidelityGradient(self, x: Any, dose: Any, dDdx: Any) -> Any:
        """
        Compute the weighted penalty gradient with respect to spot weights.

        Parameters
        ----------
        x : numpy.ndarray
            Current optimization variables, one per plan spot.
        dose : numpy.ndarray
            Dose at the selected voxels for the current weights.
        dDdx : scipy.sparse.spmatrix
            Transposed dose derivative supplied by OpenTPS, with one row per
            spot and one column per selected voxel.

        Returns
        -------
        numpy.ndarray
            Gradient of the weighted objective, in plan spot order.
        """
        return dDdx @ self.function.grad(x, dose=dose, return_dfdD=True)


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
    images with masks on their corresponding anatomical grids. The locked
    OpenTPS 3.0.1 release uses a compatibility subclass of the MCsquare calculator.
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
            Working directory created for MCsquare and assigned as an absolute path.

        Returns
        -------
        None
            Stores the settings, image classes, scanner calibration, and
            configured calculator on the new backend instance.

        Raises
        ------
        RuntimeError
            If the installed OpenTPS Core version differs from 3.0.1.
        """

        installed_version = version("opentps-core")
        if installed_version != "3.0.1":
            raise RuntimeError(f"OpenTPS dose generation requires opentps-core==3.0.1. Found {installed_version}.")

        self.settings = settings
        self.CTImage = CTImage
        self.ROIMask = ROIMask
        self.calibration = readScanner(str(settings.ct_calibration_dir))
        simulation_dir = simulation_dir.resolve()
        simulation_dir.mkdir(parents=True, exist_ok=True)
        self.calculator = _OpenTPS301Calculator(simulation_dir, settings.mcsquare_threads)
        self.calculator.ctCalibration = self.calibration
        self.calculator.beamModel = readBDL(str(settings.beam_model_path))

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
            OpenTPS DoseFidelity with weighted structure objectives and direct
            MU weights. On Windows the matrix and masks use only objective rows.
            The calculator retains this CT's scoring grid and beamlet primaries.

        Raises
        ------
        KeyError
            If a required structure label is absent from ``masks``.

        ValueError
            If the beamlet file has invalid dose coefficients or disagrees with
            the scoring dimensions, voxel spacing, or spot count, or a required
            converted structure mask is empty.
        """

        ct = opentps_image(image, self.CTImage, name=name)
        self.set_grid(ct)
        self.calculator.nbPrimaries = self.settings.beamlet_primaries
        objectives = self._scenario_objectives(ct, masks, name)
        n_voxels = int(np.prod(ct.gridSize))
        union = np.zeros(n_voxels, dtype=bool)
        for item in objectives.functionList:
            union |= item.maskVec
        logger.info(
            "Computing beamlets for %s: %d spots, CT grid %s, MCsquare threads %s",
            name,
            plan.numberOfSpots,
            ct.gridSize,
            self.settings.mcsquare_threads or "auto",
        )
        rows = np.flatnonzero(union)
        del union
        for item in objectives.functionList:
            item.maskVec = item.maskVec[rows]
            # Optimization uses the compact mask vector after geometry is set
            item.roi = None
        matrix = self.calculator.compute_objective_beamlets(ct, plan, rows, name)
        logger.info("Imported %s: %d matrix rows, %d nonzero dose coefficients", name, len(rows), matrix.nnz)
        fidelity = _ObjectiveDoseFidelity(matrix, xSquared=False)
        fidelity.function = objectives
        return fidelity

    def _scenario_objectives(self, ct: Any, masks: dict[int, Any], name: str) -> Any:
        """
        Build weighted target and organ-at-risk objectives for one scenario.

        Use a uniform-dose penalty at the prescription for the target and
        an overdose penalty above each organ-at-risk limit. Convert each
        mask to OpenTPS LPS coordinates and build its voxel vector in
        beamlet row order.

        Parameters
        ----------
        ct : opentps.core.data.images.CTImage
            Scenario CT whose LPS origin, spacing and grid size define the
            beamlet scoring grid.
        masks : dict[int, nibabel.spatialimages.SpatialImage]
            Binary NIfTI masks keyed by structure label. Include the target
            and every configured organ at risk, aligned with the scenario CT.
        name : str
            Scenario identifier used in errors for empty masks.

        Returns
        -------
        WeightedSum
            OpenTPS objective containing the configured weighted penalties.
            The target comes first so the optimizer can use its mask to
            scale the initial monitor units.

        Raises
        ------
        KeyError
            If a required structure label is missing.
        ValueError
            If a required structure mask is empty after conversion.
        """

        objectives = WeightedSum()
        # The optimizer uses the first objective's target mask to scale initial MUs
        limits = {self.settings.target_label: self.settings.prescription_gy, **self.settings.oar_limits_gy}
        for label, limit in limits.items():
            mask = opentps_image(masks[label], self.ROIMask, name=f"label_{label}")
            if not np.any(mask.imageArray):
                raise ValueError(f"Required optimization structure {label} is empty in {name}")

            objective_class = DUniform if label == self.settings.target_label else DMax
            weight = self.settings.target_weight if label == self.settings.target_label else self.settings.oar_weight
            objective = objective_class(mask, limit, weight=weight)
            # Match the mask's voxel ordering to the beamlet matrix
            objective._updateMaskVec(ct.spacing, ct.gridSize, ct.origin)
            objectives.functionList.append(objective)
        return objectives

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
