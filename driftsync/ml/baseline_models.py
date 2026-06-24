"""
Baseline ML Models
==================
Simple models that serve as baselines alongside the LSTM and Transformer.

Models (in order of preference):
  1. RandomForestDriftModel  - sklearn RandomForest on last-timestep features
  2. LogisticRegressionModel - sklearn LogisticRegression on last-timestep features
  3. ThresholdModel          - rule-based fallback using rolling error rate

If not enough data exists to train sklearn models, the system falls back to
threshold logic and clearly indicates the fallback mode.

Trained models are saved with pickle in driftsync/results/checkpoints/.
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from driftsync.utils import get_logger

logger = get_logger(__name__)

CHECKPOINT_DIR = Path("driftsync/results/checkpoints")


# ---------------------------------------------------------------------------
# Threshold fallback
# ---------------------------------------------------------------------------

class ThresholdModel:
    """
    Rule-based fallback model.

    Fires a warning when the rolling error rate over the last N trials exceeds
    a fixed threshold. No training required. Used when insufficient data exists
    to train a statistical model.
    """

    MODE = "threshold"

    def __init__(self, threshold: float = 0.35):
        self.threshold = threshold

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Args:
            features: Feature vector of shape (num_features,) where index 3 is
                      rolling_error_rate_5 and index 4 is rolling_error_rate_10.
        Returns:
            Risk score in [0, 1].
        """
        if len(features) < 5:
            return 0.0
        err5             = float(features[3]) if len(features) > 3 else 0.0
        err10            = float(features[4]) if len(features) > 4 else 0.0
        streak_incorrect = float(features[8]) if len(features) > 8 else 0.0
        score = 0.5 * err5 + 0.3 * err10 + 0.2 * streak_incorrect
        return float(np.clip(score, 0.0, 1.0))

    def is_high_risk(self, features: np.ndarray) -> bool:
        return self.predict_proba(features) >= self.threshold


# ---------------------------------------------------------------------------
# Sklearn model wrapper
# ---------------------------------------------------------------------------

class SklearnDriftModel:
    """
    Wrapper for scikit-learn classifiers trained on last-timestep features.

    The model uses the feature vector from the most recent trial in the window.
    Since sklearn models do not natively process sequences, we use the last
    timestep of the sequence (shape: num_features) as the input feature vector.
    """

    MODE_LR = "logistic_regression"
    MODE_RF = "random_forest"

    def __init__(self, model_type: str = "rf"):
        """
        Args:
            model_type: "lr" for logistic regression, "rf" for random forest.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline

        self.model_type = model_type
        self._trained   = False

        if model_type == "lr":
            clf = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            )
            self.mode_name = self.MODE_LR
        else:
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=8, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            self.mode_name = self.MODE_RF

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    clf),
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train on sequence arrays.

        Args:
            X: Shape (N, seq_len, num_features) or (N, num_features).
            y: Shape (N,) binary labels.
        """
        if X.ndim == 3:
            X_flat = X[:, -1, :]
        else:
            X_flat = X
        self._pipeline.fit(X_flat, y)
        self._trained = True
        logger.info("SklearnDriftModel (%s) trained on %d samples.", self.model_type, len(y))

    def predict_proba(self, features: np.ndarray) -> float:
        """
        Predict risk from a single feature vector.

        Args:
            features: Shape (num_features,) — the latest trial's feature vector.
        Returns:
            Risk score in [0, 1].
        """
        if not self._trained:
            return 0.0
        x     = features.reshape(1, -1)
        proba = self._pipeline.predict_proba(x)
        return float(proba[0, 1]) if proba.shape[1] == 2 else float(proba[0, 0])

    def save(self, path: Optional[Path] = None) -> None:
        path = path or (CHECKPOINT_DIR / f"sklearn_{self.model_type}.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self._pipeline, "model_type": self.model_type}, f)
        logger.info("SklearnDriftModel saved -> %s", path)

    @classmethod
    def load(cls, model_type: str = "rf", path: Optional[Path] = None) -> Optional["SklearnDriftModel"]:
        path = path or (CHECKPOINT_DIR / f"sklearn_{model_type}.pkl")
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
            obj = cls(model_type=d["model_type"])
            obj._pipeline = d["pipeline"]
            obj._trained  = True
            logger.info("SklearnDriftModel loaded from %s", path)
            return obj
        except Exception as e:
            logger.warning("Failed to load sklearn model: %s", e)
            return None


# ---------------------------------------------------------------------------
# Model selection helpers
# ---------------------------------------------------------------------------

def train_baseline_models(X: np.ndarray, y: np.ndarray) -> Tuple[str, object]:
    """
    Train both LR and RF on the provided data. Falls back to threshold if
    not enough samples are available.

    Args:
        X: Sequence arrays (N, seq_len, num_features).
        y: Labels (N,).
    Returns:
        (mode_name, model) tuple — model is the best one trained.
    """
    MIN_SAMPLES = 50

    if len(X) < MIN_SAMPLES:
        logger.warning(
            "Only %d samples available (minimum %d). Using threshold fallback.",
            len(X), MIN_SAMPLES,
        )
        return ThresholdModel.MODE, ThresholdModel()

    rf = SklearnDriftModel(model_type="rf")
    rf.fit(X, y)
    rf.save()

    lr = SklearnDriftModel(model_type="lr")
    lr.fit(X, y)
    lr.save()

    logger.info("Trained baseline models: RandomForest and LogisticRegression.")
    return rf.mode_name, rf


def get_best_available_model() -> Tuple[object, str]:
    """
    Load the best available trained model.

    Preference order: RandomForest > LogisticRegression > threshold fallback.

    Returns:
        (model, mode_name) where model has a predict_proba(features) method.
    """
    rf = SklearnDriftModel.load("rf")
    if rf is not None:
        return rf, rf.mode_name

    lr = SklearnDriftModel.load("lr")
    if lr is not None:
        return lr, lr.mode_name

    fallback = ThresholdModel()
    logger.info("No trained sklearn models found. Using threshold fallback.")
    return fallback, ThresholdModel.MODE
