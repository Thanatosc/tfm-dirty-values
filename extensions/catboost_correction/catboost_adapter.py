from __future__ import annotations
from importlib import metadata
import numpy as np

MODEL_PARAMETERS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "thread_count": 1,
    "allow_writing_files": False,
    "verbose": False,
}

def to_frame(values: np.ndarray, categorical_indices: list[int]):
    import pandas as pd

    matrix = np.asarray(values, dtype=np.float64)
    frame = pd.DataFrame(matrix.copy(), columns=list(range(matrix.shape[1])))
    for index in categorical_indices:
        tokens: list[str] = []
        for value in matrix[:, index]:
            if np.isnan(value):
                tokens.append("__CATBOOST_MISSING__")
            elif value == -1:
                tokens.append("__CATBOOST_UNSEEN__")
            else:
                if not float(value).is_integer():
                    raise ValueError(f"non-integral category code {value}")
                tokens.append(f"__CATBOOST_LEVEL_{int(value)}__")
        frame[index] = tokens
    return frame

def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    categorical_indices: list[int],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(**MODEL_PARAMETERS, random_seed=int(seed))
    model.fit(to_frame(X_train, categorical_indices), y_train, cat_features=categorical_indices)
    probabilities = model.predict_proba(to_frame(X_test, categorical_indices))[:, 1]
    if not np.isfinite(probabilities).all():
        raise ValueError("CatBoost returned non-finite probabilities")
    return probabilities, {
        "package": f"catboost=={metadata.version('catboost')}",
        "parameters": {**MODEL_PARAMETERS, "random_seed": int(seed)},
        "categorical_interface": "deterministic_string_tokens",
        "class_order": [0, 1],
    }
