import pandas as pd
import yaml
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ingest_data(config):
    raw_path = config["data"]["raw_path"]

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"File not found: {raw_path}")

    ext = os.path.splitext(raw_path)[-1].lower()

    if ext == ".csv":
        df = pd.read_csv(raw_path)
    elif ext == ".parquet":
        df = pd.read_parquet(raw_path)
    elif ext == ".json":
        df = pd.read_json(raw_path)
    else:
        raise ValueError(f"Unsupported format: {ext}")

    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    target = config["data"]["target_column"]
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available: {df.columns.tolist()}")

    logger.info(f"Target column '{target}' found OK")
    return df


if __name__ == "__main__":
    config = load_config()
    df = ingest_data(config)
    print(df.head())