# OpenTPS dose workflow

This workflow optimizes proton plans on generated CT scenarios, then recalculates each plan on real CTs. Real follow-up scans are used for evaluation, not optimization.

## Inputs

Run `uv sync` in the project's Python 3.12 environment. You need raw CT NIfTIs in HU, matching structure label maps for each real fraction, prepared CT tensors, and generated HU tensors. CTs and structures must share a patient coordinate frame.

Copy `configs/opentps.example.json` to `configs/opentps.json`. Set the machine beam model, CT calibration directory, target label, prescription, beam angles, OAR limits, `model_name`, and `scenario_sets` for your data. The example values are illustrative. JSON paths are relative to the JSON file. Each scenario set includes the planning CT; variant indices select generated CTs, and `"all"` selects every generated variant.

This config is required only to **generate new OpenTPS doses**. CTs and dose maps alone do not supply the machine and planning settings it needs.

## Run

With prepared and generated CTs already available:

```
uv run python -m src.pipeline.cli dose-generate --opentps-config configs/opentps.json
uv run python -m src.pipeline.cli dose-evaluate
```

Use the same `--robust-dose-dir` for both commands if you override the default. `dose-evaluate` reuses saved doses and needs no OpenTPS settings file.

To run the complete pipeline, including dose generation and evaluation:

```
uv run python -m src.pipeline.cli all --cts-per-patient 5 --rflow-weight-path path/to/model.pt --opentps-config configs/opentps.json
```

`all` enables both dose steps by default. Use `--no-generate-robust-doses` or `--no-dose-metrics` to skip either step. 

To evaluate the dataset's existing fraction-1 doses without OpenTPS settings, run `dose-evaluate`, or run `all --no-generate-robust-doses` for the full CT pipeline plus dose evaluation.

## Outputs

Plans, calculated doses, logs, and `manifest.json` are saved under `RESULTS/MAISI_TESTING/robust_dose/<split>/`. Dose evaluation writes `opentps_dose_metrics.csv`, `opentps_dose_summary.csv`, `opentps_dose_dvh.csv`, and `opentps_dose_comparison.csv` to the metrics directory. The manifest is published only after every patient succeeds.
