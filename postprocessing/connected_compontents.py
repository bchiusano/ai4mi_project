import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label

from utils import patient_id_from_stem


def decode_mask(path: Path, classes: int, label_step: float) -> np.ndarray:
	image = np.asarray(Image.open(path))
	if image.ndim != 2:
		raise ValueError(f"Expected a 2D label image at {path}, got shape {image.shape}")
	labels = np.rint(image.astype(np.float64) / label_step).astype(np.int64)
	if labels.min() < 0 or labels.max() >= classes:
		raise ValueError(f"Unexpected labels in {path}: {np.unique(image)}")
	return labels.astype(np.uint8)


def load_volumes(folder: Path, classes: int, label_step: float) -> tuple[
	dict[str, np.ndarray], dict[str, dict[int, Path]]
]:
	slices: dict[str, dict[int, tuple[Path, np.ndarray]]] = defaultdict(dict)
	for path in sorted(folder.glob("*.png")):
		patient = patient_id_from_stem(path.stem)
		patient_part, separator, slice_part = path.stem.rpartition("_")
		if not separator or not slice_part.isdigit():
			raise ValueError(f"Cannot read slice index from filename: {path.name}")
		if int(slice_part) in slices[patient]:
			raise ValueError(f"Duplicate slice {slice_part} for {patient}")
		slices[patient][int(slice_part)] = (path, decode_mask(path, classes, label_step))

	if not slices:
		raise RuntimeError(f"No prediction PNG files in {folder}")

	volumes: dict[str, np.ndarray] = {}
	paths_by_patient: dict[str, dict[int, Path]] = {}
	for patient, patient_slices in slices.items():
		indices = sorted(patient_slices)
		if indices != list(range(len(indices))):
			raise ValueError(f"Slice indices for {patient} are not contiguous from zero: {indices}")
		first_slice = patient_slices[indices[0]][1]
		volume = np.empty((len(indices), *first_slice.shape), dtype=np.uint8)
		paths_by_patient[patient] = {}
		for index in indices:
			path, slice_labels = patient_slices[index]
			if slice_labels.shape != first_slice.shape:
				raise ValueError(f"Inconsistent slice shape for {patient}: {path}")
			volume[index] = slice_labels
			paths_by_patient[patient][index] = path
		volumes[patient] = volume
	return volumes, paths_by_patient


def keep_largest_component(labels: np.ndarray, class_index: int) -> np.ndarray:
	"""Keep the largest 6-connected 3D component of one class; send others to background."""
	components, component_count = label(labels == class_index)
	if component_count <= 1:
		return labels.copy()
	component_sizes = np.bincount(components.ravel())[1:]
	largest_component = int(np.argmax(component_sizes)) + 1
	result = labels.copy()
	result[(labels == class_index) & (components != largest_component)] = 0
	return result


def class_mean_dice(
	predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray], class_index: int
) -> float:
	patient_scores: list[float] = []
	for patient in sorted(predictions):
		prediction = predictions[patient] == class_index
		target = targets[patient] == class_index
		denominator = int(prediction.sum() + target.sum())
		if denominator:
			patient_scores.append(2.0 * np.logical_and(prediction, target).sum() / denominator)
	return float(np.mean(patient_scores)) if patient_scores else float("nan")


def select_improving_classes(
	predictions: dict[str, np.ndarray], targets: dict[str, np.ndarray], classes: int
) -> tuple[list[int], dict[str, dict[str, float | bool]]]:
	if predictions.keys() != targets.keys():
		raise ValueError("Prediction and target patient IDs do not match")
	for patient in predictions:
		if predictions[patient].shape != targets[patient].shape:
			raise ValueError(f"Prediction/target shape mismatch for {patient}")

	selected: list[int] = []
	metrics: dict[str, dict[str, float | bool]] = {}
	for class_index in range(1, classes):
		filtered = {
			patient: keep_largest_component(volume, class_index)
			for patient, volume in predictions.items()
		}
		baseline_dice = class_mean_dice(predictions, targets, class_index)
		filtered_dice = class_mean_dice(filtered, targets, class_index)
		improves = bool(np.isfinite(filtered_dice) and filtered_dice > baseline_dice)
		if improves:
			selected.append(class_index)
		metrics[str(class_index)] = {
			"baseline_mean_patient_dice": baseline_dice,
			"largest_component_mean_patient_dice": filtered_dice,
			"selected": improves,
		}
	return selected, metrics


def save_volumes(
	volumes: dict[str, np.ndarray], paths_by_patient: dict[str, dict[int, Path]],
	output_dir: Path, label_step: float,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	for patient, volume in volumes.items():
		for index, path in paths_by_patient[patient].items():
			encoded = np.rint(volume[index].astype(np.float64) * label_step).astype(np.uint8)
			Image.fromarray(encoded).save(output_dir / path.name)


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Select and apply per-class largest-component postprocessing on 3D predictions."
	)
	parser.add_argument("--pred-dir", type=Path, required=True)
	parser.add_argument("--output-dir", type=Path, required=True)
	mode = parser.add_mutually_exclusive_group(required=True)
	mode.add_argument("--gt-dir", type=Path, help="Validation labels; selects improving classes.")
	mode.add_argument("--config", type=Path, help="Saved JSON selection; applies it without labels.")
	parser.add_argument("--classes", type=int, default=5)
	parser.add_argument("--label-step", type=float, default=63.0)
	args = parser.parse_args()

	if args.pred_dir.resolve() == args.output_dir.resolve():
		parser.error("--output-dir must differ from --pred-dir")

	if args.config:
		config: dict[str, Any] = json.loads(args.config.read_text())
		classes = int(config["classes"])
		label_step = float(config["label_step"])
		selected_classes = [int(index) for index in config["keep_largest_component"]]
	else:
		classes = args.classes
		label_step = args.label_step
		if classes < 2 or label_step <= 0:
			parser.error("--classes must be at least 2 and --label-step must be positive")
		targets, _ = load_volumes(args.gt_dir, classes, label_step)

	predictions, paths_by_patient = load_volumes(args.pred_dir, classes, label_step)
	if args.gt_dir:
		if predictions.keys() != targets.keys():
			raise ValueError("Prediction and ground-truth patient IDs do not match")
		for patient in predictions:
			if predictions[patient].shape != targets[patient].shape:
				raise ValueError(f"Prediction/ground-truth shape mismatch for {patient}")
		selected_classes, metrics = select_improving_classes(predictions, targets, classes)
		config = {
			"classes": classes,
			"label_step": label_step,
			"connectivity": 6,
			"keep_largest_component": selected_classes,
			"validation_metrics": metrics,
		}
		args.output_dir.mkdir(parents=True, exist_ok=True)
		config_path = args.output_dir / "postprocessing_config.json"
		config_path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
		for class_index, result in metrics.items():
			print(
				f"class {class_index}: baseline={result['baseline_mean_patient_dice']:.6f}, "
				f"largest_component={result['largest_component_mean_patient_dice']:.6f}, "
				f"selected={result['selected']}"
			)
		print(f"Saved selection config to: {config_path}")

	processed = predictions
	for class_index in selected_classes:
		processed = {
			patient: keep_largest_component(volume, class_index)
			for patient, volume in processed.items()
		}
	save_volumes(processed, paths_by_patient, args.output_dir, label_step)
	print(f"Applied largest-component cleanup for classes: {selected_classes}")
	print(f"Saved processed PNG slices to: {args.output_dir}")


if __name__ == "__main__":
	main()
