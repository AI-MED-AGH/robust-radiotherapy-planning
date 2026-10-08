"""Export one real generated/reference pair into standalone benchmark tensors."""

import argparse
import json
from pathlib import Path

import torch

from src.pipeline.config import MaisiTestingConfig
from src.pipeline.helpers.helpers import _load_hu_tensor
from src.pipeline.helpers.helpers_structures import (
    _get_structure_label_transform,
    _label_to_binary_mask,
    _load_structure_label_map,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient", default="Patient_01")
    parser.add_argument("--generated-index", type=int, default=1)
    parser.add_argument("--fraction", type=int, default=2)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=Path("src/data_full"))
    parser.add_argument("--pipeline-root", type=Path, default=Path("RESULTS/MAISI_TESTING"))
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--output", type=Path, default=Path("RESULTS/benchmarks/Patient_01_pair"))
    args = parser.parse_args()
    if min(args.generated_index, args.fraction, args.label) <= 0:
        parser.error("Generated index, fraction and structure label must be positive")

    config = MaisiTestingConfig(
        stage="evaluate",
        device="cpu",
        data_root=args.data_root,
        output_root=args.pipeline_root,
        evaluation_split=args.split,
        use_dose_metrics=False,
    )
    generated_name = f"{args.patient}_gen_{args.generated_index}.pt"
    reference_name = f"{args.patient}_fraction_{args.fraction}_"
    generated_path = config.generated_ct_dir / args.patient / generated_name
    reference_path = config.processed_ct_dir / f"{reference_name}.pt"
    cache_path = config.warped_structure_cache_dir / args.patient / generated_name
    structure_path = config.structures_root / args.patient / f"{reference_name}.nii.gz"
    for path in (generated_path, reference_path, cache_path, structure_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required pair input is missing: {path}")

    generated = _load_hu_tensor(generated_path, config, torch.device("cpu"))
    reference = _load_hu_tensor(reference_path, config, torch.device("cpu"))
    cached_masks = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(cached_masks, dict) or args.label not in cached_masks:
        raise ValueError(f"Warped mask cache has no structure label {args.label}")
    generated_mask = cached_masks[args.label]
    if not isinstance(generated_mask, torch.Tensor):
        raise ValueError("Cached structure mask must be a tensor")
    generated_mask = generated_mask.detach().cpu().bool()
    label_map = _load_structure_label_map(structure_path, _get_structure_label_transform(config))
    reference_mask = _label_to_binary_mask(label_map, args.label)
    tensors = {
        "ssim_pred": generated,
        "ssim_ref": reference,
        "hausdorff_pred": generated_mask,
        "hausdorff_ref": reference_mask,
    }
    for name, tensor in tensors.items():
        if tensor.ndim != 3 or tensor.shape != generated.shape:
            raise ValueError(f"{name}: incompatible shape {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: contains nonfinite values")
    if not generated_mask.any() or not reference_mask.any():
        raise ValueError("Selected structure is empty; choose a nonempty label for a representative benchmark")

    args.output.mkdir(parents=True, exist_ok=True)
    for name, tensor in tensors.items():
        torch.save(tensor.contiguous(), args.output / f"{name}.pt")
    manifest = {
        "patient": args.patient,
        "generated_ct": str(generated_path.resolve()),
        "reference_ct": str(reference_path.resolve()),
        "generated_mask_cache": str(cache_path.resolve()),
        "reference_structure": str(structure_path.resolve()),
        "label": args.label,
        "spacing": list(config.spacing),
        "shape": list(generated.shape),
        "data_min": config.data_min,
        "data_max": config.data_max,
        "preparation": "Pipeline HU conversion and structure preprocessing; existing warped mask cache reused",
    }
    (args.output / "pair.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
