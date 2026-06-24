import copy
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset_clean.csv"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def build_preprocessor(x_train: pd.DataFrame) -> ColumnTransformer:
    nominal_cols = [
        col
        for col in x_train.columns
        if "City_" in col or "EmploymentType_" in col or col == "Gender"
    ]
    num_cols = [col for col in x_train.columns if col not in nominal_cols]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("nom", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), nominal_cols),
        ]
    )


def make_pipeline(preprocessor: ColumnTransformer, classifier) -> Pipeline:
    return Pipeline(steps=[("preprocessor", copy.deepcopy(preprocessor)), ("classifier", classifier)])


def evaluate_pipeline(pipe: Pipeline, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train, y_test) -> dict:
    y_pred_train = pipe.predict(x_train)
    y_pred_test = pipe.predict(x_test)
    y_prob_test = pipe.predict_proba(x_test)[:, 1]

    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    gap = acc_train - acc_test

    return {
        "Acc Train": round(acc_train, 4),
        "Acc Test": round(acc_test, 4),
        "Brecha (gap)": round(gap, 4),
        "Overfitting OK": "✔" if gap < 0.05 else "✘",
        "Precision": round(precision_score(y_test, y_pred_test), 4),
        "Recall": round(recall_score(y_test, y_pred_test), 4),
        "F1-Score": round(f1_score(y_test, y_pred_test), 4),
        "ROC AUC": round(roc_auc_score(y_test, y_prob_test), 4),
    }


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATA_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if "LoanApproved" not in df.columns:
        raise ValueError("El dataset debe contener la columna objetivo 'LoanApproved'.")

    x = df.drop(columns=["LoanApproved"])
    y = df["LoanApproved"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(x_train)

    model_definitions = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=4,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=4,
            random_state=RANDOM_STATE,
        ),
        "rf_optimizado": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            max_depth=20,
            min_samples_leaf=1,
        ),
    }

    model_display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "rf_optimizado": "RF Optimizado",
    }

    summary_rows = []

    for file_stem, classifier in model_definitions.items():
        pipe = make_pipeline(preprocessor, classifier)
        pipe.fit(x_train, y_train)

        metrics = evaluate_pipeline(pipe, x_train, x_test, y_train, y_test)
        summary_rows.append({"Modelo": model_display_names[file_stem], **metrics})

        joblib.dump(pipe, MODELS_DIR / f"{file_stem}.pkl")

    summary_df = pd.DataFrame(summary_rows).set_index("Modelo")
    summary_df.to_csv(ASSETS_DIR / "model_comparison_summary.csv")

    metadata = {
        "dataset": DATA_PATH.name,
        "train_size": len(x_train),
        "test_size": len(x_test),
        "models": summary_df.reset_index().to_dict(orient="records"),
        "best_model_by_roc_auc": summary_df["ROC AUC"].idxmax(),
        "best_model_by_f1": summary_df["F1-Score"].idxmax(),
    }

    with open(MODELS_DIR / "model_comparison_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("Modelos exportados en /models:")
    for file_stem in model_definitions:
        print(f" - {file_stem}.pkl")
    print()
    print("Tabla comparativa guardada en assets/model_comparison_summary.csv")
    print("Metadata guardada en models/model_comparison_metadata.json")
    print()
    print(summary_df.to_string())


if __name__ == "__main__":
    main()
