"""Print the test-set metrics tracked by DVC (metrics/*.json) as a readable table.

  python scripts/show_metrics.py
"""
import json
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "metrics"
c = json.loads((M / "classification.json").read_text())
s = json.loads((M / "segmentation.json").read_text())
loc = c["gradcam_localisation"]
print(f"Chest X-ray classifier - test set ({c['test_images']} images, patient-disjoint)")
print(f"  {'Finding':<14}{'AUROC':>7}{'Sens':>7}{'Spec':>7}{'Grad-CAM pointing':>19}")
for k, v in sorted(c["per_finding"].items(), key=lambda kv: -kv[1]["auroc"]):
    print(f"  {k:<14}{v['auroc']:>7.3f}{v['sensitivity']:>7.2f}{v['specificity']:>7.2f}{loc['per_finding_pointing'][k]:>19.2f}")
print(f"  {'macro':<14}{c['macro_auroc']:>7.3f}{'':>14}{loc['pointing_game']:>19.2f}   (pointing chance ~0.07, {loc['boxes']} radiologist boxes)")
print(f"Spleen 3D U-Net - {s['test_volumes']} test CT volumes: Dice {s['mean_dice']:.3f} | HD95 {s['mean_hd95_mm']:.1f} mm")
