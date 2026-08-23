import numpy as np

from driftsync.utils.metrics import compute_classification_metrics, compute_ece, compute_roc_curve


def test_classification_metrics_handle_single_class_auc():
    y_true = np.zeros(4, dtype=int)
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])

    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["roc_auc"] == 0.0


def test_ece_includes_probability_one_in_last_bin():
    y_true = np.array([0, 1, 1])
    y_pred = np.array([0.0, 0.5, 1.0])

    ece, confidences, accuracies, counts = compute_ece(y_true, y_pred, n_bins=2)

    assert counts.tolist() == [1, 2]
    assert confidences.tolist() == [0.0, 0.75]
    assert accuracies.tolist() == [0.0, 1.0]
    assert ece == (2 * 0.25) / 3


def test_roc_curve_returns_zero_auc_for_single_class():
    y_true = np.ones(3, dtype=int)
    y_pred = np.array([0.2, 0.5, 0.8])

    _, _, auc = compute_roc_curve(y_true, y_pred)

    assert auc == 0.0


def test_roc_curve_single_class_does_not_warn(recwarn):
    y_true = np.ones(3, dtype=int)
    y_pred = np.array([0.2, 0.5, 0.8])

    fpr, tpr, auc = compute_roc_curve(y_true, y_pred)

    assert len(recwarn) == 0
    np.testing.assert_array_equal(fpr, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(tpr, np.array([0.0, 1.0]))
    assert auc == 0.0
