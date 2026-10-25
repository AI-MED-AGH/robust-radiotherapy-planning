"""Regression checks for run-local dose caching and bounded DVF prefetch."""

import threading
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

import torch
from monai.transforms import Compose  # type: ignore[attr-defined]

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.evaluation import dose_metrics, metrics
from src.pipeline.helpers import helpers_doses, helpers_structures
from src.pipeline.helpers.helpers_structures import WarpedStructureMaskCache, _StructureRegistrationResult


@contextmanager
def temporary_config() -> Iterator[MaisiTestingConfig]:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        yield MaisiTestingConfig(
            data_root=root / "data",
            output_root=root / "output",
            validate_paths=False,
            device="cpu",
            target_image_size=(4, 4, 4),
            structure_labels=[1, 2],
        )


class DosePrefetchTests(unittest.TestCase):
    def test_next_patient_prepares_while_current_metrics_run(self) -> None:
        started = threading.Event()
        release = threading.Event()
        calls: list[tuple[str, int]] = []
        main_thread = threading.get_ident()

        def load(patient: str) -> str:
            calls.append((patient, threading.get_ident()))
            if patient == "Patient_02":
                started.set()
                if not release.wait(5):
                    raise AssertionError("Metrics did not release the next registration")
            return patient

        iterator = helpers_doses._prefetch_dose_data(["Patient_01", "Patient_02", "Patient_03"], load, "cuda")
        try:
            self.assertEqual(next(iterator), ("Patient_01", "Patient_01"))
            self.assertTrue(started.wait(5), "Next patient's registration must overlap current metrics")
            self.assertEqual([patient for patient, _ in calls], ["Patient_01", "Patient_02"])
            release.set()
            self.assertEqual(list(iterator), [("Patient_02", "Patient_02"), ("Patient_03", "Patient_03")])
            self.assertEqual(len({thread for _, thread in calls}), 1)
            self.assertNotEqual(calls[0][1], main_thread)
        finally:
            release.set()
            list(iterator)

    def test_cpu_single_and_empty_inputs_remain_sequential(self) -> None:
        for tasks, device in [([1, 2], "cpu"), ([1], "cuda"), ([], "cuda")]:
            with self.subTest(tasks=tasks, device=device):
                threads: list[int] = []

                def load(task: int, threads: list[int] = threads) -> int:
                    threads.append(threading.get_ident())
                    return task * 2

                self.assertEqual(
                    list(helpers_doses._prefetch_dose_data(tasks, load, device)),
                    [(task, task * 2) for task in tasks],
                )
                self.assertTrue(all(thread == threading.get_ident() for thread in threads))

    def test_worker_error_reaches_caller(self) -> None:
        def load(task: int) -> int:
            if task == 2:
                raise ValueError("registration failed")
            return task

        iterator = helpers_doses._prefetch_dose_data([1, 2], load, "cuda")
        self.assertEqual(next(iterator), (1, 1))
        with self.assertRaisesRegex(ValueError, "registration failed"):
            next(iterator)


class DoseCacheTests(unittest.TestCase):
    def test_workflows_share_cache_but_separate_runs_do_not(self) -> None:
        with temporary_config() as config:
            config.use_scenario_robustness = True
            observed: list[WarpedStructureMaskCache] = []
            key = Path("generated.pt")

            def smoke(config: MaisiTestingConfig, *, warped_mask_cache: WarpedStructureMaskCache) -> None:
                self.assertNotIn(key, warped_mask_cache)
                observed.append(warped_mask_cache)
                warped_mask_cache[key] = {1: torch.ones(1, dtype=torch.bool)}

            def consume(config: MaisiTestingConfig, *, warped_mask_cache: WarpedStructureMaskCache) -> None:
                self.assertIs(warped_mask_cache, observed[-1])
                self.assertIn(key, warped_mask_cache)

            with (
                patch.object(metrics, "_contains_dose_files", return_value=True),
                patch.object(metrics, "evaluate_base_dose_smoke_test", side_effect=smoke),
                patch.object(metrics, "evaluate_predicted_doses", side_effect=consume),
                patch.object(metrics, "evaluate_scenario_robustness", side_effect=consume),
            ):
                metrics.evaluate_doses(config)
                metrics.evaluate_doses(config)
                supplied: WarpedStructureMaskCache = {}
                metrics.evaluate_doses(config, supplied)
            self.assertIsNot(observed[0], observed[1])
            self.assertIs(observed[2], supplied)

    def test_disk_masks_are_ignored_and_only_complete_run_cache_is_reused(self) -> None:
        with temporary_config() as config:
            generated = config.generated_ct_dir / "Patient_01_gen_0.pt"
            planning = config.processed_ct_dir / "Patient_01_fraction_1_.pt"
            structure = config.structures_root / "Patient_01" / "Patient_01_fraction_1_.nii.gz"
            old_cache = config.output_root / "warped_structure_masks" / "test" / "Patient_01" / generated.name
            old_cache.parent.mkdir(parents=True)
            old_cache.write_bytes(b"stale disk masks must never be inspected")
            dose = torch.ones(4, 4, 4)
            labels = torch.ones(4, 4, 4, dtype=torch.int64)
            cache: WarpedStructureMaskCache = {}

            def exists(path: Path) -> bool:
                self.assertEqual(path, structure, "Only the structure input may be checked")
                return True

            def warp() -> dict[int, torch.Tensor] | None:
                return helpers_doses._warp_planning_masks_for_predicted_dose(
                    generated, dose, "Patient_01", {generated.stem: generated}, Compose([]), {}, {}, cache, config
                )[0]

            with (
                patch.object(helpers_doses, "_planning_ct_path_for_patient", return_value=planning) as find_planning,
                patch.object(helpers_doses, "_load_hu_tensor", return_value=dose),
                patch.object(helpers_doses, "_load_structure_label_map", return_value=labels),
                patch.object(
                    helpers_doses,
                    "_calculate_structure_registration_result",
                    return_value=_StructureRegistrationResult(generated, dose, object()),
                ) as register,
                patch.object(helpers_doses, "_warp_planning_mask_to_generated_ct", return_value=labels.bool()),
                patch.object(Path, "exists", exists),
                patch.object(Path, "glob", side_effect=AssertionError("Unexpected disk search")),
                patch.object(Path, "rglob", side_effect=AssertionError("Unexpected disk search")),
                patch.object(torch, "load", side_effect=AssertionError("Unexpected disk cache read")),
                patch.object(torch, "save", side_effect=AssertionError("Unexpected disk cache write")),
            ):
                first = warp()
                self.assertIs(warp(), first)
                self.assertEqual(register.call_count, 1)
                self.assertEqual(find_planning.call_count, 1)
                cache[generated] = {1: labels.bool()}
                self.assertEqual(set(warp() or {}), {1, 2})
                self.assertEqual(register.call_count, 2)
                cache.clear()
                warp()
                self.assertEqual(register.call_count, 3)
            self.assertEqual(old_cache.read_bytes(), b"stale disk masks must never be inspected")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_real_dvf_and_cuda_metrics_match_cpu(self) -> None:
        with temporary_config() as config:
            config.structure_registration_iterations = 1
            config.structure_registration_shrink_factors = []
            config.structure_registration_smoothing_sigmas = []
            config.target_image_size = (8, 8, 8)
            paths = [config.generated_ct_dir / f"Patient_{i:02d}_gen_0.pt" for i in (1, 2)]
            doses = {f"Patient_{i:02d}": Path(f"Patient_{i:02d}_dose.pt") for i in (1, 2)}
            ct = torch.arange(512, dtype=torch.float32).reshape(8, 8, 8)
            labels = torch.ones_like(ct, dtype=torch.int64)
            labels[4:] = 2
            with (
                patch.object(dose_metrics, "_generated_ct_lookup", return_value={p.stem: p for p in paths}),
                patch.object(dose_metrics, "_load_dose_distribution", return_value=ct / 10),
                patch.object(
                    helpers_doses, "_planning_ct_path_for_patient", return_value=Path("Patient_01_fraction_1_.pt")
                ),
                patch.object(helpers_doses, "_load_hu_tensor", return_value=ct),
                patch.object(helpers_doses, "_load_structure_label_map", return_value=labels),
                patch.object(helpers_structures, "_load_hu_tensor", return_value=ct),
                patch.object(Path, "exists", return_value=True),
            ):
                cpu_rows = dose_metrics._evaluate_dose_on_generated_structures(doses, doses, config, "candidate", {})
                config.device = "cuda"
                gpu_rows = dose_metrics._evaluate_dose_on_generated_structures(doses, doses, config, "candidate", {})
            self.assertTrue(cpu_rows)
            self.assertEqual(len(cpu_rows), len(gpu_rows))
            for cpu_row, gpu_row in zip(cpu_rows, gpu_rows, strict=True):
                self.assertAlmostEqual(cpu_row["value"], gpu_row["value"], places=4)
                self.assertEqual(
                    {key: value for key, value in cpu_row.items() if key != "value"},
                    {key: value for key, value in gpu_row.items() if key != "value"},
                )

    def test_generated_scenarios_overlap_and_load_each_patient_dose_once(self) -> None:
        with temporary_config() as config:
            config.device = "cuda"
            paths = [
                config.generated_ct_dir / f"Patient_{patient:02d}_gen_{scenario}.pt"
                for patient, scenario in [(1, 0), (1, 1), (2, 0)]
            ]
            doses = {f"Patient_{i:02d}": Path(f"Patient_{i:02d}_dose.pt") for i in (1, 2)}
            dose = torch.ones(4, 4, 4)
            second_patient_started = threading.Event()
            release = threading.Event()
            seen: list[str] = []

            def warp(**kwargs: Any) -> tuple[dict[int, torch.Tensor], str, None, None]:
                if kwargs["patient_id"] == "Patient_02":
                    second_patient_started.set()
                    if not release.wait(5):
                        raise AssertionError("Metrics did not overlap next patient DVF")
                return {1: dose.bool()}, "warped", None, None

            def append(
                rows: list[dict[str, Any]],
                dose: torch.Tensor,
                masks: dict[int, torch.Tensor],
                config: MaisiTestingConfig,
                metadata: dict[str, Any],
            ) -> None:
                seen.append(metadata["scenario_id"])
                if len(seen) == 2:
                    self.assertTrue(second_patient_started.wait(5))
                    release.set()

            with (
                patch.object(dose_metrics, "_generated_ct_lookup", return_value={p.stem: p for p in paths}),
                patch.object(dose_metrics, "_load_dose_distribution", return_value=dose) as load,
                patch.object(dose_metrics, "_warp_planning_masks_for_predicted_dose", side_effect=warp),
                patch.object(dose_metrics, "_append_single_dose_rows", side_effect=append),
            ):
                try:
                    dose_metrics._evaluate_dose_on_generated_structures(doses, doses, config, "candidate", {})
                finally:
                    release.set()
            self.assertEqual(load.call_count, 2)
            self.assertEqual(seen, [path.stem for path in paths])


if __name__ == "__main__":
    unittest.main()
