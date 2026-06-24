# EXPLANATION.md

## What DriftSync is

DriftSync is a system that tries to predict when a person is about to make a mistake, before they actually make it. It watches how someone responds during a sustained attention task and uses machine learning to estimate whether the conditions for an error are building up.

The core output is a probability score, updated after every trial, representing how likely the user is to make an error in the next few steps. When that score crosses a threshold, the system issues a warning.

The project covers the full pipeline from data generation to real-time inference: a custom task simulator, feature engineering, sequence model training, calibration, and a live session viewer with explanation overlays.

---

## The problem it tries to solve

When people work on tasks that require sustained attention, their performance does not degrade suddenly. It degrades gradually. Reaction times start creeping up. Small errors begin clustering. The person may not notice. By the time an obvious mistake occurs, the underlying state that caused it has often been present for several seconds or longer.

Most error-detection systems wait for the error to happen, then flag it. That is reactive. The goal here is to be predictive: catch the degradation while there is still time to do something about it, whether that means prompting the user to slow down, take a break, or simply pay more attention to a critical step.

This is not a new idea in the research literature. It is more commonly called cognitive state monitoring or human performance modeling. What makes DriftSync interesting as a project is that it connects that research idea to a complete working implementation with a real UI, real-time inference, calibration, and measurable lead time.

---

## How calibration works

The problem with predicting errors from behavioral signals is that everyone has a different baseline. A fast typist with a 180ms average reaction time looks completely different from a slower, more deliberate user with a 400ms average. If the model compares everyone to the same absolute threshold, it will be badly calibrated for most people.

The calibration phase addresses this. Before a live session begins, the user completes 25 trials of the same task, but with no risk scoring. The system measures their mean reaction time, accuracy, IQR, and error rate under normal conditions. These numbers become their personal baseline.

During the live session, the explanation engine compares real-time stats against this baseline. Instead of saying "reaction time is 350ms, which is slow," it says "reaction time is 28% above your baseline of 270ms." That is a meaningful signal regardless of absolute level.

The baseline is saved to a JSON file so it persists across sessions. It can be regenerated any time by running calibration again.

---

## The ML pipeline

The prediction task is framed as binary classification: given a sequence of the last 20 trials, predict whether an error will occur in the next 5 trials.

Each trial produces a 15-dimensional feature vector. Features include normalized reaction time, correctness, rolling error rates, streak counts, and four newer features: reaction time variance, time since the last error, RT slope over the last 5 trials, and a fatigue index that combines elapsed time with cumulative error count.

These feature vectors are stacked into sequences of length 20 and fed into one of several models.

**LSTM.** A stacked recurrent network with residual connections. Suitable for temporal patterns where what happened several steps ago still matters. The recurrent structure naturally handles varying sequence dependencies.

**Transformer Encoder.** Multi-head self-attention over the full 20-step window. Tends to perform slightly better in evaluation, likely because the attention mechanism can selectively weight the most informative steps rather than compressing everything into a hidden state.

**Sklearn baselines.** Logistic Regression and Random Forest, trained on just the last timestep's features rather than the full sequence. These are much simpler but still useful when training data is limited or the neural models have not been trained yet.

**Threshold fallback.** A weighted combination of rolling error rates, requiring no training at all. This activates automatically if nothing else is available.

The system always tries to use the best available model, falling through the chain until something works.

---

## Features used

The original 11 features covered the obvious signals: normalized reaction time, correctness, elapsed time, rolling error rates (two windows), inter-trial interval, cumulative errors, correct/incorrect streaks, and two binary flags for the task stimuli.

The four new features in v2 were added because the original set was missing some signals that clearly matter:

- **RT variance:** A user whose reaction times are stable is different from one who is all over the place. High variance often precedes errors.
- **Time since last error:** The recency of the most recent error matters. A fresh error cluster is different from an isolated error 30 trials ago.
- **RT trend:** If reaction time has been rising steadily over the last 5 trials, that is a different signal from a sudden spike. The slope captures the direction of change.
- **Fatigue index:** This is a simple multiplicative combination of elapsed time and cumulative errors. Both are correlated with fatigue, so their product is a rough proxy for accumulated load.

None of these are novel signal choices. They are derived from what the literature says tends to predict error onset in sustained attention tasks.

---

## The dashboard

The live session screen shows several things simultaneously.

On the left side, there is a running plot of the risk score over the last 30 trials, updated in real time. When the score is low, it stays in a neutral color. When it crosses the warning threshold, the bar turns red and a warning message appears.

Below the plot, a status panel shows current accuracy, rolling error rate, and which model is active.

When risk is high (above 0.40), an explanation overlay appears. This is a semi-transparent box that lists the specific reasons the risk is elevated, phrased relative to the user's calibration baseline. It updates every trial.

In the bottom right corner, the model mode is shown: whether the system is using a neural model, logistic regression, random forest, or the threshold fallback. This makes it clear to the user what is driving the prediction.

---

## How lead time is measured

Lead time is the gap in seconds between the first warning and the error it preceded.

After a session ends, the system goes through all warning events and all error events. For each error, it looks back 20 trials and checks whether any warning was issued in that window. If yes, the lead time is the difference between the error timestamp and the timestamp of the first warning in the window.

The session summary shows how many errors were predicted this way, how many were missed (no warning in the window before the error), and the average lead time across predicted errors.

False positive warnings are also tracked: warnings that were issued but no error followed in the next 20 trials.

This metric is important because it is the most direct answer to the core question: when the system says something is about to go wrong, does it say so in time to be useful? A system with 80% accuracy but 0.2 second average lead time is not particularly useful. A system with 65% accuracy and 5 second average lead time is much more actionable.

---

## Limitations

The biggest limitation is that the training data is synthetic. The simulator generates sessions with a programmed drift pattern (performance degrades after trial 50, noise increases, errors cluster). The models learn those patterns extremely well, which is partly why the numbers look reasonable. Real human behavioral data is noisier and more variable.

The calibration phase is only 25 trials. That is short enough to be practical but long enough to pick up baseline characteristics if the person is performing normally. If someone is distracted or anxious during calibration, the baseline will be wrong, and comparisons during the live session will be off.

The train/val/test split is random across sessions rather than time-ordered. This means the evaluation is slightly optimistic: in a real deployment, the model would only have access to past sessions, not future ones mixed in during training.

The lead time metric is also sensitive to threshold tuning. A lower warning threshold means more warnings, more predicted errors, but also more false positives. The current default (0.65) was chosen as a reasonable tradeoff on the synthetic data. It would need re-tuning for real users.

---

## Future improvements

The most valuable next step would be real user data. Even a small dataset of 10 to 20 people doing a few sessions each would reveal whether the features and model structure transfer to real behavioral patterns.

Beyond that:

- Temporal validation splits (train on older sessions, test on newer ones)
- Online adaptation (the model fine-tunes slightly on each new session)
- Longer prediction horizons (can the system predict errors 10 or 15 trials ahead?)
- A REST API mode so the inference engine can plug into other applications
- Support for tasks other than the current click-response format

The architecture is modular enough that most of these could be added incrementally without restructuring what already exists.
