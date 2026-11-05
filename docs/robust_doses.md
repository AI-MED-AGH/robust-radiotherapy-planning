# OpenTPS dose workflow

This workflow optimizes proton plans on generated CT scenarios, then recalculates each plan on real CTs. Real follow-up scans are used for evaluation, not optimization.

## Inputs

Run `uv sync` in the project's Python 3.12 environment. You need raw CT NIfTIs in HU, matching structure label maps for each real fraction, prepared CT tensors, and generated HU tensors. CTs and structures must share a patient coordinate frame.

Use the saved planning settings in `configs/opentps.json` as the shared setup for comparing generation models. Keep the prescription, beam angles, target and OAR definitions, optimization settings, and scenario sets fixed across model comparisons. This setup is intended for realistic dose-based evaluation of generated CTs, rather than patient-specific treatment planning.

The configured machine beam model and CT calibration files must be available at the paths in the JSON; relative paths are resolved from the JSON file. Target and OAR labels must match the dataset's structure maps. Each scenario set includes the planning CT; variant indices select generated CTs, and `"all"` selects every generated variant. These settings are needed only to generate new OpenTPS doses; evaluation of saved doses does not require them.

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

## Outputs

Plans, calculated doses, logs, and `manifest.json` are saved under `RESULTS/MAISI_TESTING/robust_dose/<split>/`. Dose evaluation writes `opentps_dose_metrics.csv`, `opentps_dose_summary.csv`, `opentps_dose_dvh.csv`, and `opentps_dose_comparison.csv` to the metrics directory. The manifest is published only after every patient succeeds.
