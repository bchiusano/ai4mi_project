import sys
from pathlib import Path

from postprocessing.connected_compontents import (
    keep_largest_components, load_volumes, save_volumes,
)

classes, label_step = 5, 63.0
src, dst = Path(sys.argv[1]), Path(sys.argv[2])

volumes, paths = load_volumes(src, classes, label_step)
processed = {
    patient: keep_largest_components(volume, classes, 6)
    for patient, volume in volumes.items()
}
save_volumes(processed, paths, dst, label_step)
print(f"Saved {len(processed)} patients to {dst}")