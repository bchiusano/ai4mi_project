import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def patient_id(filename: str) -> str:
    parts = Path(filename).stem.split("_")
    return "_".join(parts[:2])


parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--folds", required=True)
parser.add_argument("--destination", required=True)
args = parser.parse_args()

source = Path(args.source).resolve()
folds_path = Path(args.folds).resolve()
destination = Path(args.destination).resolve()

if destination.exists():
    raise FileExistsError(
        f"{destination} already exists. Use a new destination name."
    )

images = {p.name: p.resolve() for p in (source / "img").glob("*.png")}
labels = {p.name: p.resolve() for p in (source / "gt").glob("*.png")}

if not images or not labels:
    raise RuntimeError("No PNG images or labels found in the source dataset.")

if images.keys() != labels.keys():
    raise RuntimeError("Image and label filenames do not match.")

empty = [
    str(path)
    for path in list(images.values()) + list(labels.values())
    if path.stat().st_size == 0
]
if empty:
    raise RuntimeError(f"Found {len(empty)} empty source files, for example {empty[0]}")

files_by_patient = defaultdict(list)
for filename in images:
    files_by_patient[patient_id(filename)].append(filename)

available_patients = set(files_by_patient)

with folds_path.open() as file:
    definition = json.load(file)

folds = definition["folds"]

validation_counts = Counter(
    patient
    for fold in folds
    for patient in fold["val"]
)
cv_patients = set(validation_counts)

if any(count != 1 for count in validation_counts.values()):
    raise RuntimeError("Every CV patient must occur in validation exactly once.")

if not cv_patients <= available_patients:
    missing = sorted(cv_patients - available_patients)
    raise RuntimeError(f"Patients missing from source dataset: {missing}")

destination.mkdir(parents=True)

for fold_number, fold in enumerate(folds):
    train_patients = set(fold["train"])
    val_patients = set(fold["val"])

    if train_patients & val_patients:
        raise RuntimeError(f"Fold {fold_number} has train/validation overlap.")

    if train_patients | val_patients != cv_patients:
        raise RuntimeError(f"Fold {fold_number} does not cover all CV patients.")

    for split_name, patients in (
        ("train", sorted(train_patients)),
        ("val", sorted(val_patients)),
    ):
        img_dir = destination / f"fold_{fold_number}" / split_name / "img"
        gt_dir = destination / f"fold_{fold_number}" / split_name / "gt"
        img_dir.mkdir(parents=True)
        gt_dir.mkdir(parents=True)

        for patient in patients:
            for filename in files_by_patient[patient]:
                (img_dir / filename).symlink_to(images[filename])
                (gt_dir / filename).symlink_to(labels[filename])

    print(
        f"fold_{fold_number}: "
        f"{len(train_patients)} train patients, "
        f"{len(val_patients)} validation patients"
    )

holdout_patients = sorted(available_patients - cv_patients)

for kind, source_files in (("img", images), ("gt", labels)):
    target_dir = destination / "holdout" / kind
    target_dir.mkdir(parents=True)

    for patient in holdout_patients:
        for filename in files_by_patient[patient]:
            (target_dir / filename).symlink_to(source_files[filename])

shutil.copy2(folds_path, destination / "folds.json")

with (destination / "holdout_patients.json").open("w") as file:
    json.dump(holdout_patients, file, indent=2)

print(f"CV patients: {len(cv_patients)}")
print(f"Held-out patients: {len(holdout_patients)}: {holdout_patients}")
print(f"Created dataset at {destination}")