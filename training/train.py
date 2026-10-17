"""Train and evaluate deployment-aligned taxi-time regression models.

Run from the project root:
    python -m training.train --data data/private/model_table.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_policy import PUBLIC_FEATURES, TARGET_COLUMN, TIME_COLUMN, validate_feature_policy

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def load_data(path: Path) -> tuple[pd.DataFrame, int]:
    frame = pd.read_csv(path)
    validate_feature_policy(frame.columns)
    frame[TIME_COLUMN] = pd.to_datetime(frame[TIME_COLUMN], errors="coerce", utc=True)
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    before = len(frame)
    frame = frame.dropna(subset=[TIME_COLUMN, TARGET_COLUMN]).copy()
    frame = frame[frame[TARGET_COLUMN].between(1.0, 60.0)]
    removed = before - len(frame)
    frame = frame.sort_values(TIME_COLUMN).reset_index(drop=True)
    if len(frame) < 50:
        raise ValueError("At least 50 valid chronological rows are required.")
    return frame, removed


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [column for column in features.columns if column not in numeric]
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", encoder),
                    ]
                ),
                categorical,
            ),
        ]
    )


def model_candidates() -> dict[str, object]:
    return {
        "Dummy median": DummyRegressor(strategy="median"),
        "Ridge": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=7,
            min_samples_leaf=5,
            max_features=0.7,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
        ),
    }


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae_min": float(mean_absolute_error(actual, predicted)),
        "rmse_min": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate(
    features: pd.DataFrame,
    target: pd.Series,
    timestamps: pd.Series,
    folds: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Pipeline]]:
    splitter = TimeSeriesSplit(n_splits=folds)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    fitted: dict[str, Pipeline] = {}

    for model_name, estimator in model_candidates().items():
        for fold, (train_index, validation_index) in enumerate(splitter.split(features), start=1):
            pipeline = Pipeline(
                [("preprocess", build_preprocessor(features.iloc[train_index])), ("model", estimator)]
            )
            pipeline.fit(features.iloc[train_index], target.iloc[train_index])
            prediction = pipeline.predict(features.iloc[validation_index])
            fold_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "train_rows": len(train_index),
                    "validation_rows": len(validation_index),
                    "validation_start": timestamps.iloc[validation_index].min().isoformat(),
                    "validation_end": timestamps.iloc[validation_index].max().isoformat(),
                    **metrics(target.iloc[validation_index], prediction),
                }
            )
            for row_index, actual, predicted in zip(
                validation_index, target.iloc[validation_index], prediction
            ):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "split_time_utc": timestamps.iloc[row_index].isoformat(),
                        "actual_taxi_time_minutes": float(actual),
                        "predicted_taxi_time_minutes": float(predicted),
                        "error_minutes": float(actual - predicted),
                    }
                )

        final_pipeline = Pipeline(
            [("preprocess", build_preprocessor(features)), ("model", estimator)]
        )
        final_pipeline.fit(features, target)
        fitted[model_name] = final_pipeline
    return pd.DataFrame(fold_rows), pd.DataFrame(prediction_rows), fitted


def summarise(fold_results: pd.DataFrame) -> pd.DataFrame:
    return (
        fold_results.groupby("model", as_index=False)
        .agg(
            cv_mae_mean=("mae_min", "mean"),
            cv_mae_std=("mae_min", "std"),
            cv_rmse_mean=("rmse_min", "mean"),
            cv_rmse_std=("rmse_min", "std"),
            cv_r2_mean=("r2", "mean"),
            cv_r2_std=("r2", "std"),
        )
        .sort_values("cv_mae_mean")
        .reset_index(drop=True)
    )


def save_plots(summary: pd.DataFrame, predictions: pd.DataFrame, best_model: str, output: Path) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.bar(summary["model"], summary["cv_mae_mean"], yerr=summary["cv_mae_std"], capsize=4)
    plt.ylabel("Forward-CV MAE (minutes)")
    plt.title("Model comparison")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output / "model_comparison.png", dpi=180)
    plt.close()

    selected = predictions[predictions["model"] == best_model]
    plt.figure(figsize=(6, 6))
    plt.scatter(
        selected["actual_taxi_time_minutes"],
        selected["predicted_taxi_time_minutes"],
        alpha=0.65,
    )
    lower = min(selected["actual_taxi_time_minutes"].min(), selected["predicted_taxi_time_minutes"].min())
    upper = max(selected["actual_taxi_time_minutes"].max(), selected["predicted_taxi_time_minutes"].max())
    plt.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    plt.xlabel("Actual taxi time (minutes)")
    plt.ylabel("Out-of-fold prediction (minutes)")
    plt.title(f"{best_model}: forward-CV predictions")
    plt.tight_layout()
    plt.savefig(output / "actual_vs_predicted.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, removed_rows = load_data(args.data)
    features = frame[PUBLIC_FEATURES].copy()
    target = frame[TARGET_COLUMN].copy()
    timestamps = frame[TIME_COLUMN].copy()

    fold_results, predictions, fitted = evaluate(features, target, timestamps, args.folds)
    summary = summarise(fold_results)
    best = summary[summary["model"] != "Dummy median"].iloc[0]
    best_model = str(best["model"])

    summary.to_csv(args.output_dir / "model_comparison.csv", index=False)
    fold_results.to_csv(args.output_dir / "fold_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)
    joblib.dump(fitted[best_model], args.output_dir / "best_model.joblib")
    save_plots(summary, predictions, best_model, args.output_dir)

    result = {
        "evaluation": "forward-chaining cross-validation",
        "rows_used": len(frame),
        "rows_removed_by_quality_filter": removed_rows,
        "date_start_utc": timestamps.min().isoformat(),
        "date_end_utc": timestamps.max().isoformat(),
        "folds": args.folds,
        "features": PUBLIC_FEATURES,
        "selected_model": best_model,
        "cv_mae_mean_min": float(best["cv_mae_mean"]),
        "cv_mae_std_min": float(best["cv_mae_std"]),
        "cv_rmse_mean_min": float(best["cv_rmse_mean"]),
        "cv_rmse_std_min": float(best["cv_rmse_std"]),
        "cv_r2_mean": float(best["cv_r2_mean"]),
        "cv_r2_std": float(best["cv_r2_std"]),
        "note": "No untouched final test claim is made after the reproducibility audit.",
    }
    (args.output_dir / "verified_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
