"""Quality gates on the trained models (read from the DVC metrics files)."""
import json

import numpy as np

from medai.explain import localisation


def test_classifier_quality_gate(trained):
    c = json.load(open(trained / "metrics/classification.json"))
    assert c["macro_auroc"] > 0.75
    assert all(v["auroc"] > 0.55 for v in c["per_finding"].values() if v["auroc"] is not None)   # every finding beats chance


def test_gradcam_localises_better_than_chance(trained):
    c = json.load(open(trained / "metrics/classification.json"))["gradcam_localisation"]
    # chance level for the pointing game = mean box area / image area (NIH boxes cover ~6% of the image)
    assert c["boxes"] > 100 and c["pointing_game"] > 0.25


def test_segmentation_quality_gate(trained):
    s = json.load(open(trained / "metrics/segmentation.json"))
    assert s["mean_dice"] > 0.8


def test_localisation_metric():
    cam = np.zeros((100, 100))
    cam[40:50, 40:50] = 1
    assert localisation(cam, (35, 35, 20, 20))["hit"]
    assert not localisation(cam, (0, 0, 20, 20))["hit"]
    assert abs(localisation(cam, (40, 40, 10, 10))["iou"] - 1.0) < 1e-6
