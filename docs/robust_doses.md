# OpenTPS dose workflow

This workflow optimizes proton plans on generated CT scenarios, then recalculates each plan on real CTs. Real follow-up scans are used for evaluation, not optimization.

## Inputs

Run `uv sync` in the project's Python 3.12 environment. You need raw CT NIfTIs in HU, prepared CT tensors, and generated HU tensors. The planning CT requires a matching structure label map for optimization. Follow-up CTs without a matching map are skipped during dose recalculation and evaluation. CTs and structures must share a patient coordinate frame.

Use the saved planning settings in `configs/opentps.json` as the shared setup for comparing generation models. Keep the prescription, beam angles, target and OAR definitions, optimization settings, and scenario sets fixed across model comparisons. This setup is intended for realistic dose-based evaluation of generated CTs, rather than patient-specific treatment planning.

The configured machine beam model and CT calibration files must be available at the paths in the JSON; relative paths are resolved from the JSON file. Target and OAR labels must match the dataset's structure maps. Each scenario set includes the planning CT; variant indices select generated CTs, and `"all"` selects every generated variant. These settings are needed only to generate new OpenTPS doses; evaluation of saved doses does not require them.

The default comparison uses `nominal` for non-robust optimization on the planning CT alone and `all_generated` for robust optimization on the planning CT and every generated CT. Doses for both plans are recalculated on the same real follow-up CTs.

For the repository's prostate dataset, label `4` is the prostate and label `1` is the external body contour.

Set `model_name` to an identifier for the method that produced the selected CT scenarios, such as a PCA model or a rectified flow model. It is recorded in dose manifests and evaluation results.

## Run

With prepared and generated CTs already available:

```
uv run python -m src.pipeline.cli dose-generate
uv run python -m src.pipeline.cli dose-evaluate
```

Use the same `--robust-dose-dir` for both commands if you override the default. `dose-evaluate` reuses saved doses and needs no OpenTPS settings file.

To run the complete pipeline, including dose generation and evaluation:

```
uv run python -m src.pipeline.cli all --cts-per-patient 5
```

Replace `5` with the number of CT variants to generate per patient for evaluation. The command uses the default model checkpoint and OpenTPS configuration paths; use `--rflow-weight-path` or `--opentps-config` only to override them.

Dose evaluation is enabled by default; use `--no-dose-metrics` to skip it. The `all` stage always generates robust doses, even with `--no-generate-robust-doses` or `--no-dose-metrics`. For standalone evaluation stages, generation is disabled by default; enable it with `--generate-robust-doses` or run the explicit `dose-generate` stage.

To evaluate the dataset's existing fraction-1 doses without OpenTPS settings, run `dose-evaluate`. If a robust-dose manifest exists, evaluation uses those saved doses instead.

## MCsquare memory use

The default config sets `mcsquare_threads` to `0`, so MCsquare uses all available threads for beamlet and final dose calculations. Set a positive integer to limit the thread count if MCsquare runs out of memory while scoring a CT. Fewer threads can reduce scoring memory and increase run time. Monte Carlo results can vary with the thread count.

During optimization, the worker uses disk-backed sparse beamlet matrices that retain only the voxels used by the objectives. The OS loads matrix pages into RAM as needed. Larger scenario sets can still increase memory pressure during optimization.

On Linux, check the parent error for the worker exit code and the patient’s `opentps.log` for the last completed stage. Exit `-9` means the worker received SIGKILL. Check the kernel log with `journalctl -k -b --grep='Out of memory|Killed process|oom-kill'`, or check the scheduler or container logs, to confirm whether the kill was due to memory.

## Outputs

The worker uses each optimized plan in memory to recalculate doses. Calculated doses, logs, and `manifest.json` are saved under `RESULTS/MAISI_TESTING/robust_dose/<split>/`. Dose evaluation writes `opentps_dose_metrics.csv`, `opentps_dose_summary.csv`, `opentps_dose_dvh.csv`, and `opentps_dose_comparison.csv` to the metrics directory. The manifest includes optimization results and is published only after every patient succeeds.
