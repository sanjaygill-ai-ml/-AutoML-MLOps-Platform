import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def engineer_features(df, config):
    target = config["data"]["target_column"]
    os.makedirs("artifacts", exist_ok=True)

    # Drop duplicates
    df = df.drop_duplicates()

    # Drop Id column if exists
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])
        logger.info("Dropped 'Id' column")

    # Split features and target
    X = df.drop(columns=[target])
    y = df[target]

    # Encode target labels to numbers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    joblib.dump(le, "artifacts/label_encoder.pkl")
    logger.info(f"Target classes: {le.classes_}")

    # Find numerical and categorical columns
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    logger.info(f"Numerical columns: {num_cols}")
    logger.info(f"Categorical columns: {cat_cols}")

    # Build preprocessing pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    transformers = [("num", num_pipeline, num_cols)]

    if cat_cols:
        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipeline, cat_cols))

    preprocessor = ColumnTransformer(transformers)

    # Transform features
    X_transformed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, "artifacts/preprocessor.pkl")
    logger.info(f"Features shape after transformation: {X_transformed.shape}")
    logger.info("Preprocessor saved to artifacts/preprocessor.pkl")

    return X_transformed, y_encoded, preprocessor


if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv(config["data"]["raw_path"])
    X, y, preprocessor = engineer_features(df, config)
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Sample y values: {y[:5]}")