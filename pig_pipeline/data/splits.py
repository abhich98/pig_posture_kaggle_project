from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def build_single_split(
    train_df: pd.DataFrame,
    val_ratio: float,
    seed: int,
    out_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_part, val_part = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        stratify=train_df["class_id"],
    )
    train_part = train_part.reset_index(drop=True)
    val_part = val_part.reset_index(drop=True)

    train_part.to_csv(out_dir / "train.csv", index=False)
    val_part.to_csv(out_dir / "val.csv", index=False)
    return train_part, val_part


def build_cv_splits(
    train_df: pd.DataFrame,
    n_folds: int,
    seed: int,
    out_dir: str | Path,
) -> list[tuple[pd.DataFrame, pd.DataFrame, int]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y = train_df["class_id"].to_numpy()

    folds: list[tuple[pd.DataFrame, pd.DataFrame, int]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_df, y)):
        fold_dir = out_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_part = train_df.iloc[train_idx].reset_index(drop=True)
        val_part = train_df.iloc[val_idx].reset_index(drop=True)

        train_part.to_csv(fold_dir / "train.csv", index=False)
        val_part.to_csv(fold_dir / "val.csv", index=False)
        folds.append((train_part, val_part, fold_idx))

    return folds
