"""Load and sample ISOT CSVs for the closed-corpus demo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = Path("data")


def load_isot(data_dir: Path | str = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load True.csv / Fake.csv and attach source-bucket labels.

    label 1 / label_name 'true'  → Reuters-style bucket
    label 0 / label_name 'fake'  → unreliable-outlet bucket

    These are corpus construction labels, not claim-level fact-check verdicts.
    """
    data_dir = Path(data_dir)
    true_path = data_dir / "True.csv"
    fake_path = data_dir / "Fake.csv"
    if not true_path.exists() or not fake_path.exists():
        raise FileNotFoundError(
            f"Missing {true_path} and/or {fake_path}. "
            "Run: python scripts/download_data.py"
        )

    df_true = pd.read_csv(true_path)
    df_fake = pd.read_csv(fake_path)
    df_true = df_true.copy()
    df_fake = df_fake.copy()
    df_true["label"] = 1
    df_fake["label"] = 0
    df_true["label_name"] = "true"
    df_fake["label_name"] = "fake"

    df = pd.concat([df_true, df_fake], ignore_index=True)
    df["article_id"] = np.arange(len(df), dtype=np.int64)
    for col in ("title", "text", "subject", "date"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    return df


def stratified_sample(
    df: pd.DataFrame,
    n_articles: int,
    random_state: int = 7,
) -> pd.DataFrame:
    """Stratified sample by label (approx. half true / half fake)."""
    n_per = max(1, n_articles // 2)
    parts = []
    for _, group in df.groupby("label"):
        take = min(n_per, len(group))
        parts.append(group.sample(n=take, random_state=random_state))
    out = pd.concat(parts, ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
