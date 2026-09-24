
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "tourism.csv"
OUTPUT_DIR = BASE_DIR / "data" / "pipeline_artifacts"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


# Load dataset
df = pd.read_csv(DATA_PATH)

df_clean = df.copy()


# Remove unwanted index column
if "Unnamed: 0" in df_clean.columns:
    df_clean.drop(columns=["Unnamed: 0"], inplace=True)


# Standardize categorical text
categorical_columns = df_clean.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    df_clean[column] = (
        df_clean[column]
        .astype(str)
        .str.strip()
    )


# Standardize Gender values
if "Gender" in df_clean.columns:
    df_clean["Gender"] = df_clean["Gender"].replace({
        "Fe Male": "Female",
        "fe male": "Female",
        "FE MALE": "Female",
        "female": "Female",
        "male": "Male"
    })


# Remove duplicate rows
df_clean = df_clean.drop_duplicates()


# Remove identifier column
identifier_columns = [
    column
    for column in ["CustomerID"]
    if column in df_clean.columns
]

df_model = df_clean.drop(
    columns=identifier_columns
)


# Separate features and target
X = df_model.drop(
    columns=["ProdTaken"]
)

y = df_model["ProdTaken"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


# Identify feature types
numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object"]
).columns.tolist()


# Numerical preprocessing
numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing
categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_pipeline,
            numerical_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)


# Fit preprocessing on training data
X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


# Check for missing values
if np.isnan(X_train_processed).sum() != 0:
    raise ValueError(
        "Missing values remain in processed training data."
    )

if np.isnan(X_test_processed).sum() != 0:
    raise ValueError(
        "Missing values remain in processed testing data."
    )


# Store prepared data
prepared_data = {
    "X_train_processed": X_train_processed,
    "X_test_processed": X_test_processed,
    "y_train": y_train.to_numpy(),
    "y_test": y_test.to_numpy(),
    "preprocessor": preprocessor,
    "input_features": X_train.shape[1],
    "processed_features": X_train_processed.shape[1],
    "training_samples": X_train.shape[0],
    "testing_samples": X_test.shape[0],
    "random_state": RANDOM_STATE
}


joblib.dump(
    prepared_data,
    OUTPUT_DIR / "prepared_data.joblib"
)


# Save preparation metadata
metadata = {
    "original_shape": list(df.shape),
    "cleaned_shape": list(df_clean.shape),
    "modeling_shape": list(df_model.shape),
    "training_shape": list(X_train.shape),
    "testing_shape": list(X_test.shape),
    "processed_training_shape": list(
        X_train_processed.shape
    ),
    "processed_testing_shape": list(
        X_test_processed.shape
    ),
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "random_state": RANDOM_STATE
}


with open(
    OUTPUT_DIR / "preparation_metadata.json",
    "w"
) as f:
    json.dump(
        metadata,
        f,
        indent=4
    )


print("Data preparation completed successfully.")
print(f"Original dataset: {df.shape}")
print(f"Cleaned dataset: {df_clean.shape}")
print(f"Training data: {X_train.shape}")
print(f"Testing data: {X_test.shape}")
print(
    f"Processed training data: "
    f"{X_train_processed.shape}"
)
print(
    f"Processed testing data: "
    f"{X_test_processed.shape}"
)
