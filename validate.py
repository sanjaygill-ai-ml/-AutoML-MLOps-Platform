import pandas as pd
import yaml
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class DataValidator:
    def __init__(self, config):
        self.target = config["data"]["target_column"]
        self.errors = []

    def check_target_exists(self, df):
        if self.target not in df.columns:
            self.errors.append(f"Target column '{self.target}' missing")

    def check_missing_values(self, df):
        for col in df.columns:
            missing = df[col].isna().mean()
            if missing > 0.4:
                self.errors.append(f"Column '{col}' has too many missing values: {missing:.0%}")

    def check_minimum_rows(self, df):
        if len(df) < 50:
            self.errors.append(f"Too few rows: {len(df)} — need at least 50")

    def check_duplicates(self, df):
        dupes = df.duplicated().sum()
        if dupes > 0:
            logger.warning(f"Found {dupes} duplicate rows — will be dropped later")

    def validate(self, df):
        self.check_target_exists(df)
        self.check_missing_values(df)
        self.check_minimum_rows(df)
        self.check_duplicates(df)

        if self.errors:
            for err in self.errors:
                logger.error(f"Validation failed: {err}")
            return False

        logger.info("All validation checks passed!")
        return True


if __name__ == "__main__":
    import pandas as pd
    config = load_config()
    df = pd.read_csv(config["data"]["raw_path"])
    validator = DataValidator(config)
    result = validator.validate(df)
    print("Validation passed!" if result else "Validation failed!")