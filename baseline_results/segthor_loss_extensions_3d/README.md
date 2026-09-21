# SegTHOR loss extensions

This directory contains the compact results of the loss-function experiments on the final corrected SegTHOR dataset. The data split contains 15 training patients and 5 validation patients. Results are selected and reported using patient-level 3D Dice over the four foreground organs.

| Loss | Best epoch | Mean 3D Dice | Esophagus | Heart | Trachea | Aorta |
|---|---:|---:|---:|---:|---:|---:|
| Cross-Entropy | 24 | 0.524 | 0.316 | 0.794 | 0.490 | 0.497 |
| Dice | 24 | 0.553 | 0.461 | 0.669 | 0.485 | 0.595 |
| Cross-Entropy + Dice | 21 | 0.613 | 0.443 | 0.814 | 0.654 | 0.541 |

The `dice/` and `ce_dice/` directories contain the selected model, weights, metric arrays, plots, and patient-level evaluation tables. `ce_all_epochs/` contains the post-hoc 3D evaluation of all 25 epochs from the corrected Cross-Entropy baseline in `../segthor_clean_patient19_fixed/`.

The thousands of per-slice prediction PNGs are intentionally excluded. Predictions can be regenerated from `bestweights.pt` with `predict_segthor_from_weights.py`.

To regenerate the compact comparison figures from the repository root:

```bash
python make_segthor_comparison_figures.py \
  --results-root baseline_results/segthor_loss_extensions_3d \
  --ce-baseline-dir baseline_results/segthor_clean_patient19_fixed \
  --output-dir baseline_results/segthor_loss_comparison_3d
```

The primary results and presentation figures are in `../segthor_loss_comparison_3d/`.
