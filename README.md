DriftSync is a research-grade machine learning system that predicts when a human is likely to make a mistake — *before it happens*. It models how reaction time, accuracy, and behavior evolve over time using deep sequence models to estimate the probability of an error occurring in the next K steps.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Core Idea](#core-idea)
- [Models](#models)
- [Evaluation](#evaluation)
- [Results](#results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)

---

## Overview

DriftSync is a complete end-to-end human-in-the-loop ML system. It includes:

- A cognitive task simulator (interactive and synthetic modes)
- Feature engineering and temporal sequence construction
- Two deep learning architectures: **LSTM** and **Transformer Encoder**
- Full training, evaluation, and calibration pipeline
- Monte Carlo Dropout uncertainty estimation
- Real-time inference with warning triggers
- Automatic experiment comparison and visualization

---

## Core Idea

Human performance degrades gradually during sustained tasks:

- Reaction times increase
- Micro-errors occur
- Fatigue
- Attention becomes unstable

These changes follow measurable temporal patterns. DriftSync learns those patterns and predicts future risk using sequence modeling.

### Prediction Task

Given the last `L` steps of user behavior:

```
S_t = [x_{t-L+1}, ..., x_t]
```

The model predicts:

```
P(error occurs in next K steps | S_t)
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `L` | Sequence length | 20 |
| `K` | Prediction horizon | 5 |
| `x_i` | Engineered behavioral feature vector | 11 features |

The output includes a **probability estimate** and an **uncertainty estimate**.

### System Flow

```
Simulator → Feature Engineering → Sequence Windowing
    → LSTM / Transformer → Risk Prediction
    → Evaluation + Calibration
    → Real-Time Warning System
```

---

## Models

### LSTM-Based Predictor
- Stacked recurrent architecture
- Residual connections
- Layer normalization
- Orthogonal weight initialization
- Designed for temporal memory retention

### Transformer Encoder Predictor
- Multi-head self-attention
- Causal masking (no future leakage)
- Sinusoidal positional encoding
- Pre-LayerNorm architecture
- Global average pooling

**Both models support:**
- Weighted binary cross-entropy (class imbalance handling)
- AdamW optimizer with learning rate scheduling
- Gradient clipping & early stopping
- Monte Carlo Dropout at inference time

---

## Evaluation

Models are evaluated using:

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness |
| Precision / Recall | Error class performance |
| F1 Score | Balanced precision-recall |
| ROC AUC | Discriminative ability |
| ECE | Expected Calibration Error |
| Latency | Inference speed benchmark |

All evaluation plots are automatically generated and saved.

---

## Results

Representative results on 20 sessions × 200 trials:

| Model | Accuracy | F1 | AUC | ECE |
|-------|----------|----|-----|-----|
| LSTM | ~0.74 | ~0.70 | ~0.81 | ~0.06 |
| Transformer | ~0.76 | ~0.73 | ~0.83 | ~0.05 |

> The Transformer typically achieves slightly higher AUC and calibration at modest additional inference cost.

---

## Project Structure

```
DriftSync/
├── launch.py                  # Main entry point
├── run_experiment.py
├── requirements.txt
│
└── driftsync/
    ├── simulator/             # Cognitive task simulator
    ├── data/                  # Data loading & preprocessing
    ├── models/                # LSTM & Transformer architectures
    ├── training/              # Training loop & optimization
    ├── evaluation/            # Metrics, calibration, plots
    ├── realtime/              # Live inference engine
    ├── utils/                 # Shared utilities
    └── results/               # Saved outputs & visualizations
```

---

## Getting Started

### Installation

```bash
git clone https://github.com/Bouwles/DriftSync.git
cd DriftSync
pip install -r requirements.txt
```

### Run

All functionality is launched from a single point:

```bash
python launch.py
```

From the menu, you can:
- Run the full experiment pipeline
- Train individual models
- Compare model performance
- Launch the interactive simulator
- Run real-time inference

No manual multi-step setup required.

### Real-Time Inference

DriftSync supports live inference mode:

1. The user interacts with the simulator
2. Features are streamed to the trained model
3. At each timestep, the system predicts future error risk
4. If risk exceeds a threshold, a **warning is triggered**

This simulates a cognitive AI co-pilot that monitors performance degradation in real time.

---

## Tech Stack

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| PyTorch | 2.x |
| NumPy | latest |
| Matplotlib | latest |
| Scikit-learn | latest |
| Pygame | latest |

---

<div align="center">
  <i>Built for research. Designed for real-time use.</i>
</div>
