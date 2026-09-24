
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PREPARED_DATA_PATH = (
    BASE_DIR
    / "data"
    / "pipeline_artifacts"
    / "prepared_data.joblib"
)

MODEL_DIR = (
    BASE_DIR
    / "model_building"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# Check prepared data
# ============================================================

if not PREPARED_DATA_PATH.exists():
    raise FileNotFoundError(
        f"Prepared data not found: {PREPARED_DATA_PATH}"
    )


# ============================================================
# Load prepared data
# ============================================================

prepared_data = joblib.load(
    PREPARED_DATA_PATH
)

X_train = prepared_data["X_train_processed"]
X_test = prepared_data["X_test_processed"]

y_train = prepared_data["y_train"]
y_test = prepared_data["y_test"]

preprocessor = prepared_data["preprocessor"]

print("Prepared data loaded successfully.")
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")


# ============================================================
# Configure MLflow
# ============================================================

MLFLOW_DB = (
    MODEL_DIR
    / "mlflow.db"
)

MLFLOW_ARTIFACTS = (
    MODEL_DIR
    / "mlruns"
)

MLFLOW_ARTIFACTS.mkdir(
    parents=True,
    exist_ok=True
)

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

EXPERIMENT_NAME = (
    "Tourism Package Purchase Prediction"
)


# Create experiment with project-local artifact storage
experiment = mlflow.get_experiment_by_name(
    EXPERIMENT_NAME
)

if experiment is None:

    mlflow.create_experiment(
        name=EXPERIMENT_NAME,
        artifact_location=MLFLOW_ARTIFACTS.as_uri()
    )

else:

    print(
        f"Using existing MLflow experiment: "
        f"{EXPERIMENT_NAME}"
    )


mlflow.set_experiment(
    EXPERIMENT_NAME
)

print(f"MLflow database: {MLFLOW_DB}")
print(f"MLflow artifacts: {MLFLOW_ARTIFACTS}")


# ============================================================
# Evaluation function
# ============================================================

def evaluate_model(
    model,
    X_data,
    y_data
):

    predictions = model.predict(
        X_data
    )

    probabilities = model.predict_proba(
        X_data
    )[:, 1]

    return {
        "accuracy": accuracy_score(
            y_data,
            predictions
        ),

        "precision": precision_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_data,
            predictions,
            zero_division=0
        ),

        "roc_auc": roc_auc_score(
            y_data,
            probabilities
        ),

        "pr_auc": average_precision_score(
            y_data,
            probabilities
        )
    }


# ============================================================
# Baseline models
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE
    ),

    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),

    "XGBoost": XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    ),

    "KNN": KNeighborsClassifier()
}


# ============================================================
# Baseline model training
# ============================================================

baseline_results = []

print("\nStarting baseline model training...")


for model_name, model in models.items():

    print(
        f"\nTraining baseline model: "
        f"{model_name}"
    )

    with mlflow.start_run(
        run_name=f"Baseline - {model_name}"
    ):

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        mlflow.log_param(
            "model",
            model_name
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_metrics(
            metrics
        )

        mlflow.sklearn.log_model(
            model,
            name="model"
        )

        baseline_results.append({

            "Model": model_name,

            "Test Accuracy": metrics[
                "accuracy"
            ],

            "Test Precision": metrics[
                "precision"
            ],

            "Test Recall": metrics[
                "recall"
            ],

            "Test F1": metrics[
                "f1"
            ],

            "Test ROC-AUC": metrics[
                "roc_auc"
            ],

            "Test PR-AUC": metrics[
                "pr_auc"
            ]
        })


# ============================================================
# Baseline results
# ============================================================

baseline_df = pd.DataFrame(
    baseline_results
)

baseline_df = baseline_df.sort_values(
    by="Test ROC-AUC",
    ascending=False
)

baseline_df.to_csv(
    MODEL_DIR / "baseline_results.csv",
    index=False
)

print("\nBaseline Results:")

print(
    baseline_df.to_string(
        index=False
    )
)


# ============================================================
# Hyperparameter search spaces
# ============================================================

rf_param_grid = {

    "n_estimators": [
        300,
        500,
        700,
        900
    ],

    "max_depth": [
        None,
        10,
        15,
        20,
        25
    ],

    "min_samples_split": [
        2,
        5,
        10
    ],

    "min_samples_leaf": [
        1,
        2,
        4
    ],

    "max_features": [
        "sqrt",
        "log2",
        None
    ],

    "class_weight": [
        None,
        "balanced",
        "balanced_subsample"
    ]
}


knn_param_grid = {

    "n_neighbors": list(
        range(3, 21)
    ),

    "weights": [
        "uniform",
        "distance"
    ],

    "p": [
        1,
        2
    ]
}


# ============================================================
# Cross-validation
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# Random Forest tuning
# ============================================================

print(
    "\nStarting Random Forest "
    "hyperparameter tuning..."
)

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    param_distributions=rf_param_grid,

    n_iter=30,

    scoring="roc_auc",

    cv=cv,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1
)


with mlflow.start_run(
    run_name="Tuned - Random Forest"
):

    rf_search.fit(
        X_train,
        y_train
    )

    rf_best = (
        rf_search.best_estimator_
    )

    rf_metrics = evaluate_model(
        rf_best,
        X_test,
        y_test
    )

    mlflow.log_params(
        rf_search.best_params_
    )

    mlflow.log_metric(
        "cv_roc_auc",
        rf_search.best_score_
    )

    mlflow.log_metrics(
        rf_metrics
    )

    mlflow.sklearn.log_model(
        rf_best,
        name="model"
    )


print("\nRandom Forest best parameters:")

print(
    rf_search.best_params_
)

print(
    f"Random Forest CV ROC-AUC: "
    f"{rf_search.best_score_:.4f}"
)


# ============================================================
# KNN tuning
# ============================================================

print(
    "\nStarting KNN "
    "hyperparameter tuning..."
)

knn_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),

    param_distributions=knn_param_grid,

    n_iter=30,

    scoring="roc_auc",

    cv=cv,

    random_state=RANDOM_STATE,

    n_jobs=-1,

    verbose=1
)


with mlflow.start_run(
    run_name="Tuned - KNN"
):

    knn_search.fit(
        X_train,
        y_train
    )

    knn_best = (
        knn_search.best_estimator_
    )

    knn_metrics = evaluate_model(
        knn_best,
        X_test,
        y_test
    )

    mlflow.log_params(
        knn_search.best_params_
    )

    mlflow.log_metric(
        "cv_roc_auc",
        knn_search.best_score_
    )

    mlflow.log_metrics(
        knn_metrics
    )

    mlflow.sklearn.log_model(
        knn_best,
        name="model"
    )


print("\nKNN best parameters:")

print(
    knn_search.best_params_
)

print(
    f"KNN CV ROC-AUC: "
    f"{knn_search.best_score_:.4f}"
)


# ============================================================
# Tuned model comparison
# ============================================================

tuned_results = pd.DataFrame([

    {
        "Model": "Random Forest",

        "CV ROC-AUC":
            rf_search.best_score_,

        "Test Accuracy":
            rf_metrics["accuracy"],

        "Test Precision":
            rf_metrics["precision"],

        "Test Recall":
            rf_metrics["recall"],

        "Test F1":
            rf_metrics["f1"],

        "Test ROC-AUC":
            rf_metrics["roc_auc"],

        "Test PR-AUC":
            rf_metrics["pr_auc"]
    },

    {
        "Model": "KNN",

        "CV ROC-AUC":
            knn_search.best_score_,

        "Test Accuracy":
            knn_metrics["accuracy"],

        "Test Precision":
            knn_metrics["precision"],

        "Test Recall":
            knn_metrics["recall"],

        "Test F1":
            knn_metrics["f1"],

        "Test ROC-AUC":
            knn_metrics["roc_auc"],

        "Test PR-AUC":
            knn_metrics["pr_auc"]
    }

])


tuned_results = tuned_results.sort_values(
    by="Test ROC-AUC",
    ascending=False
)


tuned_results.to_csv(
    MODEL_DIR / "tuned_results.csv",
    index=False
)


print("\nTuned Results:")

print(
    tuned_results.to_string(
        index=False
    )
)


# ============================================================
# Select final model
# ============================================================

if (
    rf_metrics["roc_auc"]
    >=
    knn_metrics["roc_auc"]
):

    final_model = rf_best

    final_model_name = (
        "Random Forest"
    )

    final_metrics = rf_metrics

    final_cv_score = (
        rf_search.best_score_
    )

    final_params = (
        rf_search.best_params_
    )

else:

    final_model = knn_best

    final_model_name = "KNN"

    final_metrics = knn_metrics

    final_cv_score = (
        knn_search.best_score_
    )

    final_params = (
        knn_search.best_params_
    )


print("\nSelected model:")
print(final_model_name)

print("\nFinal model parameters:")
print(final_params)


# ============================================================
# Save final model
# ============================================================

joblib.dump(
    final_model,
    MODEL_DIR / "final_model.pkl"
)


# ============================================================
# Save preprocessor
# ============================================================

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)


# ============================================================
# Save final metrics
# ============================================================

metrics_output = {

    "model":
        final_model_name,

    "accuracy":
        final_metrics["accuracy"],

    "precision":
        final_metrics["precision"],

    "recall":
        final_metrics["recall"],

    "f1":
        final_metrics["f1"],

    "roc_auc":
        final_metrics["roc_auc"],

    "pr_auc":
        final_metrics["pr_auc"],

    "cv_roc_auc":
        final_cv_score,

    "best_params":
        final_params
}


with open(
    MODEL_DIR / "final_metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics_output,
        f,
        indent=4,
        default=str
    )


# ============================================================
# Save model metadata
# ============================================================

model_info = {

    "model_name":
        final_model_name,

    "random_state":
        RANDOM_STATE,

    "input_features":
        prepared_data[
            "input_features"
        ],

    "processed_features":
        prepared_data[
            "processed_features"
        ],

    "training_samples":
        prepared_data[
            "training_samples"
        ],

    "testing_samples":
        prepared_data[
            "testing_samples"
        ]
}


with open(
    MODEL_DIR / "model_info.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_info,
        f,
        indent=4
    )


# ============================================================
# Save confusion matrix
# ============================================================

predictions = final_model.predict(
    X_test
)

cm = confusion_matrix(
    y_test,
    predictions
)

confusion_output = {

    "true_negative":
        int(cm[0, 0]),

    "false_positive":
        int(cm[0, 1]),

    "false_negative":
        int(cm[1, 0]),

    "true_positive":
        int(cm[1, 1])
}


with open(
    MODEL_DIR / "confusion_matrix.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        confusion_output,
        f,
        indent=4
    )


# ============================================================
# Final output
# ============================================================

print("\nFinal Model Performance:")

print(
    f"Model: {final_model_name}"
)

print(
    f"Accuracy: "
    f"{final_metrics['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{final_metrics['precision']:.4f}"
)

print(
    f"Recall: "
    f"{final_metrics['recall']:.4f}"
)

print(
    f"F1 Score: "
    f"{final_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC: "
    f"{final_metrics['roc_auc']:.4f}"
)

print(
    f"PR-AUC: "
    f"{final_metrics['pr_auc']:.4f}"
)

print(
    "\nModel artifacts saved to:"
)

print(
    MODEL_DIR
)

print(
    "\nTraining completed successfully."
)
