import joblib
import json
import os
import yaml
import logging
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate_model(config):
    os.makedirs("artifacts", exist_ok=True)

    model = joblib.load("artifacts/best_model.pkl")
    preprocessor = joblib.load("artifacts/preprocessor.pkl")
    label_encoder = joblib.load("artifacts/label_encoder.pkl")

    df = pd.read_csv(config["data"]["raw_path"])
    if "Id" in df.columns:
        df = df.drop(columns=["Id"])

    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = label_encoder.transform(df[target])

    X_transformed = preprocessor.transform(X)

    _, X_test, _, y_test = train_test_split(
        X_transformed, y,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred,
                target_names=label_encoder.classes_, output_dict=True)

    metrics = {
        "f1_score": round(f1, 4),
        "accuracy": round(acc, 4),
        "model_type": type(model).__name__,
        "classification_report": report
    }

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"Accuracy : {acc:.4f}")
    logger.info("Metrics saved to artifacts/metrics.json")
    return metrics


if __name__ == "__main__":
    config = load_config()
    evaluate_model(config)