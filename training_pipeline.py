import sys
import os
sys.path.insert(0, os.path.abspath("."))

import yaml
import logging
import pandas as pd

from src.data_ingestion.ingest import load_config, ingest_data
from src.data_validation.validate import DataValidator
from src.feature_engineering.features import engineer_features
from src.model_training.train import train_all_models
from src.model_evaluation.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("="*50)
    logger.info("   AUTOML PIPELINE STARTING")
    logger.info("="*50)

    # Step 1 - Load config
    config = load_config()

    # Step 2 - Ingest data
    logger.info("STEP 1: Data Ingestion")
    df = ingest_data(config)

    # Step 3 - Validate data
    logger.info("STEP 2: Data Validation")
    validator = DataValidator(config)
    if config["data"]["target_column"] not in df.columns:
        config["data"]["target_column"] = df.columns[-1]
        logger.info(f"Auto-detected target column: {df.columns[-1]}")

    validator = DataValidator(config)
    if not validator.validate(df):
        raise ValueError("Data validation failed! Pipeline stopped.")
    # Step 4 - Feature engineering
    logger.info("STEP 3: Feature Engineering")
    X, y, preprocessor = engineer_features(df, config)

    # Step 5 - Train models
    logger.info("STEP 4: Model Training")
    best_model, best_name, best_score = train_all_models(X, y, config)

    # Step 6 - Evaluate
    logger.info("STEP 5: Model Evaluation")
    metrics = evaluate_model(config)

    logger.info("="*50)
    logger.info(f"PIPELINE COMPLETE!")
    logger.info(f"Best Model : {best_name}")
    logger.info(f"F1 Score   : {best_score:.4f}")
    logger.info(f"API ready  : run 'uvicorn src.deployment.api:app --port 8000'")
    logger.info("="*50)


if __name__ == "__main__":
    run_pipeline()