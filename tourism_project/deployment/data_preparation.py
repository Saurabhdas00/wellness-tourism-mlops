
import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = (
    BASE_DIR
    / "data"
    / "tourism.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "pipeline_artifacts"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RANDOM_STATE = 42


# ============================================================
# Load dataset
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

original_rows = len(df)
original_columns = len(df.columns)

print("Dataset loaded successfully.")
print(f"Original dataset shape: {df.shape}")


# ============================================================
# Remove unwanted index column
# ============================================================

if "Unnamed: 0" in df.columns:
    df = df.drop(
        columns=["Unnamed: 0"]
    )


# ============================================================
# Clean string columns
# ============================================================

for column in df.select_dtypes(
    include="object"
).columns:

    df[column] = (
        df[column]
        .str.strip()
    )


# ============================================================
# Standardize categorical values
# ============================================================

if "Gender" in df.columns:

    df["Gender"] = df["Gender"].replace(
        {
            "Fe Male": "Female"
        }
    )


# ============================================================
# Remove duplicate rows
# ============================================================

duplicate_rows = int(
    df.duplicated().sum()
)

if duplicate_rows > 0:
    df = df.drop_duplicates()


# ============================================================
# Remove identifier column
# ============================================================

if "CustomerID" in df.columns:

    df = df.drop(
        columns=["CustomerID"]
    )


# ============================================================
# Validate target
# ============================================================

TARGET = "ProdTaken"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' not found."
    )


# ============================================================
# Separate features and target
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET]


# ============================================================
# Train/test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# ============================================================
# Identify feature types
# ============================================================

numerical_features = (
    X_train
    .select_dtypes(
        include=["int64", "float64"]
    )
    .columns
    .tolist()
)

categorical_features = (
    X_train
    .select_dtypes(
        include=["object"]
    )
    .columns
    .tolist()
)


print(
    f"Numerical features: "
    f"{len(numerical_features)}"
)

print(
    f"Categorical features: "
    f"{len(categorical_features)}"
)


# ============================================================
# Numerical preprocessing
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# ============================================================
# Categorical preprocessing
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# ============================================================
# Combined preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# Fit preprocessing on training data
# ============================================================

X_train_processed = (
    preprocessor.fit_transform(
        X_train
    )
)

X_test_processed = (
    preprocessor.transform(
        X_test
    )
)


# ============================================================
# Processed feature count
# ============================================================

processed_features = (
    X_train_processed.shape[1]
)


# ============================================================
# Prepare artifacts
# ============================================================

prepared_data = {
    "X_train_processed": X_train_processed,
    "X_test_processed": X_test_processed,
    "y_train": y_train,
    "y_test": y_test,
    "preprocessor": preprocessor,
    "input_features": X.columns.tolist(),
    "processed_features": processed_features,
    "training_samples": len(X_train),
    "testing_samples": len(X_test)
}


# ============================================================
# Save prepared data
# ============================================================

prepared_data_path = (
    OUTPUT_DIR
    / "prepared_data.joblib"
)

joblib.dump(
    prepared_data,
    prepared_data_path
)


# ============================================================
# Save preparation metadata
# ============================================================

metadata = {
    "source_file": str(DATA_PATH),
    "original_rows": int(original_rows),
    "original_columns": int(original_columns),
    "cleaned_rows": int(len(df)),
    "cleaned_columns": int(len(df.columns)),
    "duplicate_rows_removed": int(duplicate_rows),
    "target": TARGET,
    "input_features": X.columns.tolist(),
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "processed_features": int(processed_features),
    "random_state": RANDOM_STATE
}


metadata_path = (
    OUTPUT_DIR
    / "preparation_metadata.json"
)

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# Verification
# ============================================================

print()
print("Data preparation completed successfully.")
print(f"Training data: {X_train_processed.shape}")
print(f"Testing data: {X_test_processed.shape}")
print(
    f"Prepared data saved to: "
    f"{prepared_data_path}"
)
print(
    f"Metadata saved to: "
    f"{metadata_path}"
)
