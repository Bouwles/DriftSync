import numpy as np
import pandas as pd

from driftsync.data.dataset import FEATURE_COLS, build_sequences_from_df, extract_sequences, split_data
from driftsync.data.preprocessing import add_target_label, engineer_features
from tests.test_preprocessing import sample_trials


def processed_session(session_id: str, offset: int = 0) -> pd.DataFrame:
    df = sample_trials().copy()
    df["session_id"] = session_id
    df["trial_idx"] = df["trial_idx"] + offset
    return add_target_label(engineer_features(df), horizon=2)


def test_extract_sequences_uses_requested_window_length():
    processed = processed_session("s1")

    X, y = extract_sequences(processed, seq_len=3, feature_cols=FEATURE_COLS)

    assert X.shape == (4, 3, len(FEATURE_COLS))
    assert y.shape == (4,)
    np.testing.assert_array_equal(X[0, -1], processed.loc[2, FEATURE_COLS].to_numpy(dtype=np.float32))


def test_build_sequences_does_not_cross_session_boundaries():
    combined = pd.concat(
        [processed_session("s1", offset=0), processed_session("s2", offset=100)],
        ignore_index=True,
    )

    X, y = build_sequences_from_df(combined, seq_len=4)

    assert X.shape == (6, 4, len(FEATURE_COLS))
    assert y.shape == (6,)
