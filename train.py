import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
import yaml
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
import sys
import os
sys.path.insert(0, os.path.abspath("."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


MODEL_MAP = {
    "RandomForest": RandomForestClassifier,
    "XGBoost": XGBClassifier,
    "LogisticRegression": LogisticRegression,
    "SVC": SVC,
}


def train_all_models(X, y, config):
    mlflow_cfg = config["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    best_score = -1
    best_model = None
    best_model_name = None

    for model_cfg in config["model_training"]["models"]:
        name = model_cfg["name"]
        params = model_cfg.get("params", {})

        with mlflow.start_run(run_name=name):
            logger.info(f"Training {name}...")
            model = MODEL_MAP[name](**params)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_param("model_name", name)
            mlflow.log_params(params)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, "model")

            logger.info(f"{name} — F1: {f1:.4f} | Accuracy: {acc:.4f}")

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_model_name = name

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(best_model, "artifacts/best_model.pkl")
    logger.info(f"Best model: {best_model_name} with F1={best_score:.4f}")
    logger.info("Best model saved to artifacts/best_model.pkl")

    return best_model, best_model_name, best_score


if __name__ == "__main__":
    from src.feature_engineering.features import engineer_features
    config = load_config()
    df = pd.read_csv(config["data"]["raw_path"])
    X, y, _ = engineer_features(df, config)
    best_model, best_name, best_score = train_all_models(X, y, config)
    print(f"\nWinner: {best_name} with F1 = {best_score:.4f}")