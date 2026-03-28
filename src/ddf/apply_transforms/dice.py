import glob
import json
import os

import nibabel as nib
import numpy as np


def iou(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate IoU for two binary arrays of the same size.

    Args:
        x (np.ndarray): The first array.
        y (np.ndarray): The second array.

    Returns:
        float: The IoU float value.
    """
    dum = x + y
    dum[dum > 0] = 1
    bar = np.copy(x)
    bar[bar > 0] = 1
    foo = np.copy(y)
    foo[foo > 0] = 1
    return float(np.sum(foo * bar) / np.sum(dum))


# Config
DATA_FILE = "DATA/data_dict.json"
DATA_DIR = "DATA/"

SAVE_DIR = "RESULTS/DEFORMATIONS/APPLY_TRANSFORMS/STRUCTURES_Ts"
THS = [0.1 + i * 0.1 for i in range(10)]

LABELS = [2, 3, 4, 5]

with open(DATA_FILE, "r") as f:
    data_dict = json.load(f)["test"]

ids = sorted(
    set([os.path.basename(item["moving_image"]).split("_")[1] for item in data_dict])
)

dices: dict[int, list[float]] = {2: [], 3: [], 4: [], 5: []}
adices: dict[int, list[float]] = {2: [], 3: [], 4: [], 5: []}
geds: dict[int, list[float]] = {2: [], 3: [], 4: [], 5: []}

for sid in LABELS:
    print(f"{sid=}")
    for pid in ids:
        # Read probability maps
        sname = f"{SAVE_DIR}/Patient_{pid}/PROBABILIY_MAPS/GT_Patient_{pid}_STRUCTURE_{sid}.nii.gz"
        gt = nib.load(sname).get_fdata()  # type: ignore
        sname = f"{SAVE_DIR}/Patient_{pid}/PROBABILIY_MAPS/PRED_Patient_{pid}_STRUCTURE_{sid}.nii.gz"
        pred = nib.load(sname).get_fdata()  # type: ignore

        # Calculate gray-level dice
        dice = 2 * np.sum(np.sqrt(gt * pred)) / (np.sum(gt) + np.sum(pred))
        dices[sid].append(float(dice))

        # Calculate adice
        adice = []
        for TH in THS:
            th_gt = np.zeros(gt.shape, dtype=np.float32)
            th_gt[gt >= TH] = 1
            th_pred = np.zeros(pred.shape, dtype=np.float32)
            th_pred[pred >= TH] = 1
            a = 2 * np.sum(th_gt * th_pred) / (np.sum(th_gt) + np.sum(th_pred))
            adice.append(a)
        adice_mean = np.mean(adice)
        adices[sid].append(float(adice_mean))

        # Calculate ged
        fnames = glob.glob(
            f"{SAVE_DIR}/Patient_{pid}/GT_Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz"
        )
        gt_imgs = [nib.load(fname).get_fdata() for fname in fnames]  # type: ignore

        fnames = glob.glob(
            f"{SAVE_DIR}/Patient_{pid}/Variants/Variant_*_STRUCTURE_{sid}_Patient_*.nii.gz"
        )
        pred_imgs = [nib.load(fname).get_fdata() for fname in fnames]  # type: ignore

        sum1 = .0
        for i in range(len(gt_imgs)):
            for j in range(len(pred_imgs)):
                sum1 += 1 - iou(gt_imgs[i], pred_imgs[j])
        sum1 /= len(gt_imgs) * len(pred_imgs)

        sum2 = .0
        for i in range(len(gt_imgs) - 1):
            for j in range(i, len(gt_imgs)):
                sum2 += 1 - iou(gt_imgs[i], gt_imgs[j])
        sum2 /= len(gt_imgs) * (len(gt_imgs) - 1) / 2

        sum3 = .0
        for i in range(len(pred_imgs) - 1):
            for j in range(i, len(pred_imgs)):
                sum3 += 1 - iou(pred_imgs[i], pred_imgs[j])
        sum3 /= len(pred_imgs) * (len(pred_imgs) - 1) / 2

        ged = 2 * sum1 - sum2 - sum3
        geds[sid].append(float(ged))

        # Save results
        with open("results.json", "w") as f:
            json.dump(
                {"ids": ids, "dices": dices, "adices": adices, "geds": geds},
                f,
                indent=4,
            )

        print(f"\t{pid=}, {dice=}, {adice=}, {ged=}")

    print(
        sid,
        np.mean(dices[sid]),
        np.std(dices[sid]),
        np.mean(adices[sid]),
        np.std(adices[sid]),
        np.mean(geds[sid]),
        np.std(geds[sid]),
    )

for key in dices.keys():
    print(key, np.mean(dices[key]), np.std(dices[key]))

for key in adices.keys():
    print(key, np.mean(adices[key]), np.std(adices[key]))

for key in geds.keys():
    print(key, np.mean(geds[key]), np.std(geds[key]))

with open("results.json", "w") as f:
    json.dump({"ids": ids, "dices": dices, "adices": adices, "geds": geds}, f, indent=4)
