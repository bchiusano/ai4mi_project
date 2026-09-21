# SegTHOR loss comparison

All scores are from the same five-patient corrected validation split. The primary metric is the macro mean of patient-level 3D Dice over the four foreground organs.

| Loss | Best epoch | Mean 3D Dice | Esophagus | Heart | Trachea | Aorta |
|---|---:|---:|---:|---:|---:|---:|
| CE | 24 | 0.524 | 0.316 | 0.794 | 0.490 | 0.497 |
| Dice | 24 | 0.553 | 0.461 | 0.669 | 0.485 | 0.595 |
| CE + Dice | 21 | 0.613 | 0.443 | 0.814 | 0.654 | 0.541 |

CE + Dice achieved the highest overall score. Dice alone achieved the highest esophagus and aorta scores, while CE + Dice achieved the highest heart and trachea scores.

Validation losses are not compared across methods because each objective has a different numerical scale.
