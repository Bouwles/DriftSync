# DriftSync: Real-Time Cognitive Drift and Human Error Risk Prediction

DriftSync is a real-time machine learning system that predicts when a user is likely to make a mistake before the mistake occurs. It models behavioral signals such as reaction time, accuracy, and response patterns over time using deep sequence models and lightweight baseline classifiers to estimate the probability of an error in the next K steps.

This is a performance and behavioral prediction tool, not a medical device. It is not clinically validated and makes no health claims.

---

## Table of Contents

- [Overview](#overview)
- [Why this project matters](#why-this-project-matters)
- [Main features](#main-features)
- [How the system works](#how-the-system-works)
- [Machine learning approach](#machine-learning-approach)
- [Feature engineering](#feature-engineering)
- [Calibration system](#calibration-system)
- [Prediction lead time](#prediction-lead-time)
- [Evaluation metrics](#evaluation-metrics)
- [Project structure](#project-structure)
- [How to run](#how-to-run)
- [Demo mode](#demo-mode)
- [Example outputs](#example-outputs)
- [Limitations](#limitations)
- [Future improvements](#future-improvements)

---

## Overview

DriftSync is a complete end-to-end pipeline for behavioral sequence modeling and real-time risk prediction. It includes:

- An interactive cognitive task simulator (click targets, respond to cues)
- A pre-session calibration phase to establish a personal behavioral baseline
- Temporal feature engineering over rolling windows of behavioral data
- Two deep learning architectures: LSTM and Transformer Encoder
- Two lightweight sklearn baselines: Logistic Regression and Random Forest
- A threshold fallback model requiring no training data
- Monte Carlo Dropout for uncertainty estimation at inference time
- A real-time inference engine with live risk overlay and warning triggers
- Rule-based explanation panel showing why risk is elevated
- Prediction lead time tracking (how far ahead warnings precede errors)
- Session export to CSV and JSON

---

## Why this project matters

Errors in sustained cognitive tasks follow predictable patterns. Reaction time increases before a mistake. Accuracy drops in clusters. Error streaks are more common after a period of unusually fast responding. These signals are detectable from behavioral data before the error itself occurs.

Most monitoring systems measure errors after the fact. DriftSync attempts to measure the conditions that precede errors and issue warnings while the user can still correct course.

The system is built around the question: given the last 20 timesteps of behavioral data, what is the probability that an error will occur in the next 5 steps?

---

## Main features

**Calibration:** A 25-trial warm-up phase before each live session. The system measures mean reaction time, accuracy, and error rate under normal conditions and stores a personal baseline. All subsequent comparisons are made relative to this baseline.

**Real-time inference:** During a live task, features are computed after each trial and fed to the loaded model. A risk score between 0 and 1 is updated continuously. When the score crosses a configurable threshold (default 0.65), a warning is displayed.

**Explanation panel:** When risk is elevated, a short overlay describes which behavioral signals are above normal. Examples: reaction time 25% above baseline, error streak of 3 consecutive mistakes, fatigue index elevated.

**Prediction lead time:** The system tracks which warnings preceded an actual error within a 20-trial window. Lead time is the gap in seconds between the first qualifying warning and the error. Aggregate lead time metrics are shown at session end and stored per session.

**Model fallback chain:** The system uses the best available model in order: Random Forest, Logistic Regression, threshold-based fallback. If no trained model exists, the threshold model activates automatically. The model mode is shown in the UI.

**Session metrics:** All session data is saved to JSON. A results screen lists all past sessions with accuracy, lead time, and prediction statistics. Data can be exported to CSV.

---

## How the system works

```
User interaction
    -> Task engine collects reaction time, correctness, timestamps
    -> Feature engineering produces 15-dimensional vectors
    -> Sliding window (L=20) creates input sequences
    -> Model produces P(error in next K=5 steps)
    -> Risk score drives warning display and explanation panel
    -> At session end: lead time metrics computed and saved
```

The live inference path adds a new datapoint every trial. It maintains a rolling buffer of recent features and passes the buffer to the model as a sequence. Uncertainty is estimated via Monte Carlo Dropout (50 forward passes) when a neural model is loaded.

---

## Machine learning approach

### LSTM predictor

A stacked LSTM with residual connections and layer normalization. Designed for temporal memory retention across long behavioral sequences. Orthogonal weight initialization. Trained with weighted binary cross-entropy to handle class imbalance (errors are less frequent than correct trials).

### Transformer Encoder predictor

Multi-head self-attention with causal masking, sinusoidal positional encoding, and pre-LayerNorm architecture. Global average pooling over the sequence before the output layer. Tends to perform slightly better than the LSTM at the cost of modestly higher inference latency.

### Sklearn baselines

Both the LSTM and Transformer require enough training data to be useful. For sessions where training data is scarce or models have not been trained, the system falls back to sklearn pipelines trained on the last timestep of each sequence:

- **Logistic Regression:** StandardScaler + L2 logistic regression. Fast, interpretable coefficients.
- **Random Forest:** 100 trees with max_depth=6. Handles non-linear feature interactions.

Training uses the last timestep feature vector from each sequence (`X[:, -1, :]`), which is a reasonable approximation when the goal is a single risk score per window.

### Threshold fallback

When no trained model is available, the system uses a weighted combination of rolling error rates:

```
score = 0.5 * error_rate_last_5 + 0.3 * error_rate_last_10 + 0.2 * streak_incorrect_norm
```

This requires no training and activates automatically. It is labeled "threshold" in the UI.

---

## Feature engineering

The system engineers 15 features from raw trial data. All features are normalized before use.

| Feature | Description |
|---------|-------------|
| `reaction_time_norm` | Reaction time normalized by IQR across the session |
| `correctness` | 1 if correct, 0 if incorrect |
| `elapsed_time_norm` | Time since session start, normalized |
| `rolling_error_rate_5` | Fraction of errors in the last 5 trials |
| `rolling_error_rate_10` | Fraction of errors in the last 10 trials |
| `inter_trial_interval_norm` | Time between trials, normalized |
| `cumulative_errors_norm` | Total errors so far, divided by trial count |
| `streak_correct` | Current consecutive correct trial count |
| `streak_incorrect` | Current consecutive incorrect trial count |
| `target_match` | Whether the displayed target matched the expected type |
| `action_click` | Whether the user clicked (vs. timed out) |
| `rolling_rt_variance` | Rolling standard deviation of reaction time (window=5), normalized by IQR |
| `time_since_last_error_norm` | Trials since the last error, divided by 20 |
| `rt_trend` | Slope of reaction time over the last 5 trials, clipped and rescaled to [0,1] |
| `fatigue_index` | Product of `elapsed_time_norm` and `cumulative_errors_norm` |

The last four features were added in v2. RT variance captures instability. Time since last error captures recovery patterns. RT trend detects whether the user is slowing down. Fatigue index combines time-on-task with accumulated errors as a proxy for overall load.

---

## Calibration system

Before a live session begins, the system runs a 25-trial calibration phase using the same task the user will perform. No risk scoring happens during calibration. The system measures:

- Mean and median reaction time
- IQR of reaction time (25th and 75th percentile)
- Accuracy and error rate
- Mean inter-trial interval

These values are stored as a baseline in `driftsync/sessions/calibration/baseline_latest.json`. During the live session, the explanation engine compares real-time stats against this baseline to generate specific reasons for elevated risk.

Calibration can be skipped per session via a toggle in the UI. If no baseline exists, the explanation panel falls back to generic thresholds.

---

## Prediction lead time

Lead time is the primary temporal evaluation metric. It measures how much warning the system gave before an actual error occurred.

For each error in a session, the system checks whether a warning was issued in the 20 trials preceding that error. If yes, the lead time is computed as:

```
lead_time = time_of_error - time_of_first_warning_in_window
```

The aggregate report includes:

- Total errors in session
- Errors predicted (at least one warning in the preceding window)
- Errors missed (no warning before the error)
- Average, minimum, and maximum lead time in seconds
- False positive warnings (warnings not followed by any error within 20 trials)

These metrics are shown in the session outro screen and saved to `driftsync/sessions/metrics_<session_id>.json`.

---

## Evaluation metrics

Models trained via the full pipeline are evaluated on a held-out test set:

| Metric | Description |
|--------|-------------|
| Accuracy | Fraction of trials classified correctly |
| Precision | Of predicted error events, fraction that were actual errors |
| Recall | Of actual error events, fraction that were predicted |
| F1 | Harmonic mean of precision and recall |
| ROC AUC | Area under the receiver operating characteristic curve |
| ECE | Expected Calibration Error (how well probabilities match observed frequencies) |
| Inference latency | Mean time per forward pass in milliseconds |

All metrics include 95% bootstrap confidence intervals. Evaluation plots are saved automatically to `driftsync/results/`.

Representative results on 20 synthetic sessions of 200 trials each:

| Model | Accuracy | F1 | AUC | ECE |
|-------|----------|----|-----|-----|
| LSTM | 0.74 | 0.70 | 0.81 | 0.06 |
| Transformer | 0.76 | 0.73 | 0.83 | 0.05 |

These are approximate values from synthetic data. Results will vary depending on session count, trial count, and individual behavioral patterns.

---

## Project structure

```
DriftSync/
├── launch.py                        # Main entry point (launches Pygame GUI)
├── run_experiment.py                # Headless training and evaluation pipeline
├── requirements.txt
│
└── driftsync/
    ├── app/
    │   └── application.py           # Main application state machine and UI
    ├── configs/
    │   └── config.py                # All configuration dataclasses
    ├── data/
    │   ├── dataset.py               # Dataset class and FEATURE_COLS definition
    │   ├── preprocessing.py         # Feature engineering pipeline
    │   └── loader.py                # Data loading utilities
    ├── demo/                        # Pre-recorded demo sessions
    ├── evaluation/
    │   ├── compare.py               # Model comparison logic
    │   └── plots.py                 # Evaluation visualizations
    ├── ml/
    │   ├── calibrator.py            # Calibration engine and BaselineStats
    │   ├── baseline_models.py       # SklearnDriftModel and ThresholdModel
    │   └── explainer.py             # Rule-based explanation engine
    ├── models/
    │   ├── lstm_model.py            # LSTM architecture
    │   └── transformer_model.py     # Transformer Encoder architecture
    ├── realtime/
    │   ├── inference_engine.py      # Live feature computation and prediction
    │   └── live_simulator.py        # Pygame real-time task and risk overlay
    ├── sessions/
    │   ├── calibration/             # Stored baseline JSON files
    │   └── metrics_*.json           # Per-session lead time and stats
    ├── simulator/
    │   ├── gui.py                   # Interactive task GUI with calibration
    │   ├── task_engine.py           # Trial logic and event generation
    │   └── headless_generator.py    # Synthetic session generation
    ├── training/
    │   ├── trainer.py               # Training loop
    │   └── pipeline.py              # End-to-end training orchestration
    └── utils/
        ├── metrics.py               # Bootstrap CI, lead time computation
        ├── logger.py                # Structured logging
        └── seed.py                  # Reproducibility seeding
```

---

## How to run

### Requirements

Python 3.11 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pygame`.

### Launch the GUI

```bash
python launch.py
```

From the main menu:

- **Run Demo**: Watch a pre-recorded synthetic session with risk scoring and explanations
- **Run Full Experiment**: Generate synthetic data, train models, evaluate, compare
- **Play Task**: Start an interactive session (runs calibration first by default)
- **Live Mode**: Real-time inference with the trained model
- **Results**: View all past sessions, lead time stats, export to CSV

### Run the training pipeline only

```bash
python run_experiment.py
```

This generates synthetic data, trains both models, evaluates them, and saves all results to `driftsync/results/`.

### Folder creation

All output directories (`sessions/`, `sessions/calibration/`, `results/checkpoints/`) are created automatically on first run.

---

## Demo mode

The Demo option in the main menu loads a pre-recorded synthetic session and replays it with full risk visualization. The session was generated with a deliberate drift pattern: stable performance in the first 60 trials, gradual increase in error rate and reaction time through the middle, and a high-error cluster near the end.

The demo shows:

- Risk score rising before error clusters
- Warning overlays appearing before errors occur
- The explanation panel listing specific behavioral signals
- A lead time summary at the end

The demo does not use a live model and does not require any prior training. It is intended to illustrate what the system looks like during normal use.

---

## Example outputs

**Session outro (after a live session):**

```
Errors predicted: 7 / 9
Errors missed:    2
Average lead time: 4.3 s
False positive warnings: 3
```

**Explanation panel (during live session, risk = 0.71):**

```
Risk elevated: 0.71
Reaction time 28% above baseline
Error streak: 3 consecutive mistakes
RT trend increasing over last 5 trials
```

**Session JSON (saved to driftsync/sessions/metrics_<id>.json):**

```json
{
  "session_id": "20260624_143201",
  "n_errors": 9,
  "n_predicted_before": 7,
  "n_missed": 2,
  "avg_lead_time_s": 4.3,
  "false_positive_warnings": 3,
  "model_mode": "random_forest"
}
```

---

## Limitations

**Synthetic data.** The default pipeline trains on synthetically generated behavioral data. This data approximates real human performance patterns but is not collected from actual users. Real-world performance may differ substantially.

**Task specificity.** The simulator uses a single click-response task. Features and models are tuned for this task structure. Generalization to other task types has not been tested.

**Calibration sensitivity.** The 25-trial calibration phase can be noisy if the user is still warming up or behaving unusually during calibration. A longer calibration window would reduce this noise.

**Class imbalance.** Errors are less frequent than correct trials. The system handles this with weighted loss during training, but precision and recall on the error class remain lower than overall accuracy.

**No temporal validation.** Current train/val/test splits are random across sessions rather than time-ordered. Proper temporal splitting would give a more honest estimate of how the model generalizes to future sessions.

**Lead time variability.** Lead time depends on where in a trial sequence warnings happen to fall. A single warning at trial N-1 before an error at trial N gives a very short lead time regardless of model quality.

---

## Future improvements

- Replace synthetic data with real user data collected across multiple task types
- Add temporal train/val/test splitting to avoid data leakage
- Extend calibration to track within-session drift from the calibration baseline
- Add a longer-horizon prediction target (K=10 or K=15) and evaluate separately
- Support multi-class prediction (mild drift vs. severe drift vs. no drift)
- Implement online model adaptation that fine-tunes on a user's own session history
- Add a fatigue recovery model that predicts when performance returns to baseline after a break
- Provide a REST API mode for integrating the inference engine with other applications
