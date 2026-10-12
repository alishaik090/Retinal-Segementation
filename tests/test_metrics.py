import numpy as np

from retinal_ai.metrics import binary_metrics


def test_binary_metrics_perfect_prediction():
    truth = np.array([[1, 0], [0, 1]])
    metrics = binary_metrics(truth, truth.copy())
    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["pixel_accuracy"] == 1.0


def test_binary_metrics_no_overlap():
    truth = np.array([[1, 0], [0, 0]])
    prediction = np.array([[0, 1], [0, 0]])
    metrics = binary_metrics(truth, prediction)
    assert metrics["dice"] < 1e-6
    assert metrics["iou"] < 1e-6


def test_binary_metrics_exclude_pixels_outside_valid_mask():
    truth = np.array([[1, 0], [0, 0]])
    prediction = np.array([[1, 0], [0, 1]])
    valid_mask = np.array([[1, 1], [1, 0]])

    metrics = binary_metrics(truth, prediction, valid_mask)

    assert metrics["dice"] == 1.0

