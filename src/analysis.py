import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TARGET = "imdb_score"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/movie_metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    return parser.parse_args()


def save_score_distribution(data: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(data[TARGET], kde=True)
    plt.title("Distribution of IMDb ratings")
    plt.xlabel("IMDb score")
    plt.ylabel("Number of movies")
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=160)
    plt.close()


def save_target_correlations(data: pd.DataFrame, output_dir: Path) -> pd.Series:
    correlations = (
        data.select_dtypes(include=[np.number])
        .corr()[TARGET]
        .drop(TARGET)
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    top_correlations = correlations.head(10).sort_values()
    plt.figure(figsize=(8, 6))
    colors = ["#c44e52" if value < 0 else "#4c72b0" for value in top_correlations]
    top_correlations.plot.barh(color=colors)
    plt.title("Strongest numeric correlations with IMDb score")
    plt.xlabel("Pearson correlation")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "target_correlations.png", dpi=160)
    plt.close()
    return correlations


def prepare_model_data(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    complete_numeric_data = data.select_dtypes(include=[np.number]).dropna()
    features = complete_numeric_data.drop(columns=[TARGET])
    target = complete_numeric_data[TARGET]
    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    features_train_full, features_test, target_train_full, target_test = (
        train_test_split(features, target, test_size=0.2, random_state=42)
    )
    features_train, features_validation, target_train, target_validation = (
        train_test_split(
            features_train_full,
            target_train_full,
            test_size=0.25,
            random_state=42,
        )
    )
    return (
        features_train,
        features_validation,
        features_test,
        target_train,
        target_validation,
        target_test,
    )


def train_models(
    features_train: pd.DataFrame,
    target_train: pd.Series,
) -> dict[str, object]:
    linear_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
    linear_model.fit(features_train, target_train)
    random_forest.fit(features_train, target_train)
    return {
        "linear_regression": linear_model,
        "random_forest": random_forest,
    }


def evaluate_model(
    model: object,
    features: pd.DataFrame,
    target: pd.Series,
) -> dict[str, float]:
    predictions = model.predict(features)
    return {
        "r2": float(r2_score(target, predictions)),
        "mae": float(mean_absolute_error(target, predictions)),
        "rmse": float(mean_squared_error(target, predictions) ** 0.5),
    }


def save_prediction_plot(
    actual: pd.Series,
    predicted: np.ndarray,
    output_dir: Path,
) -> None:
    minimum = min(actual.min(), predicted.min())
    maximum = max(actual.max(), predicted.max())
    plt.figure(figsize=(7, 7))
    plt.scatter(actual, predicted, alpha=0.3)
    plt.plot([minimum, maximum], [minimum, maximum], "r--")
    plt.xlabel("Actual IMDb score")
    plt.ylabel("Predicted IMDb score")
    plt.title("Predicted vs. actual scores")
    plt.tight_layout()
    plt.savefig(output_dir / "predicted_vs_actual.png", dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data).drop_duplicates().reset_index(drop=True)
    save_score_distribution(data, args.output)
    correlations = save_target_correlations(data, args.output)

    features, target = prepare_model_data(data)
    (
        features_train,
        features_validation,
        features_test,
        target_train,
        target_validation,
        target_test,
    ) = split_data(features, target)

    models = train_models(features_train, target_train)
    validation_metrics = {
        name: evaluate_model(model, features_validation, target_validation)
        for name, model in models.items()
    }
    selected_name = max(
        validation_metrics,
        key=lambda name: validation_metrics[name]["r2"],
    )
    selected_model = models[selected_name]
    test_metrics = evaluate_model(selected_model, features_test, target_test)
    test_predictions = selected_model.predict(features_test)
    save_prediction_plot(target_test, test_predictions, args.output)

    results = {
        "rows_after_duplicate_removal": int(len(data)),
        "numeric_features": int(features.shape[1]),
        "complete_numeric_rows": int(len(features)),
        "split_rows": {
            "train": int(len(features_train)),
            "validation": int(len(features_validation)),
            "test": int(len(features_test)),
        },
        "strongest_numeric_correlations": {
            name: float(value) for name, value in correlations.head(5).items()
        },
        "validation_metrics": validation_metrics,
        "selected_model": selected_name,
        "test_metrics": test_metrics,
    }

    with (args.output / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
