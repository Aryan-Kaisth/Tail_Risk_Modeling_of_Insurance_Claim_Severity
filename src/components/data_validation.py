# src/components/data_validation.py

import sys

from pandas.api.types import (
    is_float_dtype,
    is_string_dtype
)

from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import CustomException
from src.logger import get_logger
from src.entity.artifact_entity import DataValidationArtifact
from src.utils.io import (
    read_csv_file,
    read_yaml_file,
    write_json_file,
    write_text_file,
)

logger = get_logger(__name__)


class DataValidation:
    def __init__(
        self,
        ingestion_artifact: DataIngestionArtifact,
        validation_config: DataValidationConfig = DataValidationConfig(),
    ):
        self.validation_config = validation_config
        self.ingestion_artifact = ingestion_artifact

    def validate_schema(self) -> bool:
        logger.info("Starting schema validation.")

        try:
            schema = read_yaml_file(self.validation_config.schema_file_path)

            expected_columns = set(schema["columns"].keys())

            datasets = {
                "Train": read_csv_file(self.ingestion_artifact.train_data_path),
                "Test": read_csv_file(self.ingestion_artifact.test_data_path),
            }

            validation_passed = True

            for dataset_name, df in datasets.items():

                actual_columns = set(df.columns)

                missing_columns = expected_columns - actual_columns
                unexpected_columns = actual_columns - expected_columns

                if missing_columns:
                    logger.error(
                        "%s dataset is missing columns: %s",
                        dataset_name,
                        sorted(missing_columns),
                    )
                    validation_passed = False

                if unexpected_columns:
                    logger.error(
                        "%s dataset contains unexpected columns: %s",
                        dataset_name,
                        sorted(unexpected_columns),
                    )
                    validation_passed = False

            if validation_passed:
                logger.info("Schema validation passed successfully.")
            else:
                logger.error("Schema validation failed.")

            return validation_passed

        except Exception as e:
            raise CustomException(e, sys)

    def validate_dtypes(self) -> bool:
        logger.info("Starting datatype validation.")

        try:
            schema = read_yaml_file(self.validation_config.schema_file_path)

            datasets = {
                "Train": read_csv_file(self.ingestion_artifact.train_data_path),
                "Test": read_csv_file(self.ingestion_artifact.test_data_path),
            }

            validation_passed = True

            dtype_validators = {
                "float64": is_float_dtype,
                "string": is_string_dtype,
            }

            for dataset_name, df in datasets.items():

                for column, metadata in schema["columns"].items():

                    expected_dtype = metadata["dtype"]

                    validator = dtype_validators.get(expected_dtype)

                    if validator is None:
                        logger.warning(
                            "Unsupported dtype '%s' for column '%s'.",
                            expected_dtype,
                            column,
                        )
                        continue

                    if not validator(df[column]):
                        logger.error(
                            "%s dataset: Column '%s' has dtype '%s'. Expected '%s'.",
                            dataset_name,
                            column,
                            df[column].dtype,
                            expected_dtype,
                        )
                        validation_passed = False

            if validation_passed:
                logger.info("Datatype validation passed successfully.")
            else:
                logger.error("Datatype validation failed.")

            return validation_passed

        except Exception as e:
            raise CustomException(e, sys)

    def validate_missing_values(self) -> bool:
        logger.info("Starting missing value validation.")

        try:
            schema = read_yaml_file(self.validation_config.schema_file_path)

            id_col = schema["id_col"]
            target_col = schema["target_col"]

            datasets = {
                "Train": read_csv_file(self.ingestion_artifact.train_data_path),
                "Test": read_csv_file(self.ingestion_artifact.test_data_path),
            }

            validation_passed = True

            for dataset_name, df in datasets.items():

                logger.info("Checking missing values in %s dataset.", dataset_name)

                missing_counts = df.isnull().sum()

                for column, missing_count in missing_counts.items():

                    if missing_count == 0:
                        continue

                    logger.warning(
                        "%s dataset: Column '%s' contains %d missing values.",
                        dataset_name,
                        column,
                        missing_count,
                    )

                    # ID column cannot contain nulls
                    if column == id_col:
                        logger.error(
                            "%s dataset: ID column contains missing values.",
                            dataset_name,
                        )
                        validation_passed = False

                    # Target column cannot contain nulls
                    if column == target_col:
                        logger.error(
                            "%s dataset: Target column contains missing values.",
                            dataset_name,
                        )
                        validation_passed = False

                    # Entire column is null
                    if missing_count == len(df):
                        logger.error(
                            "%s dataset: Column '%s' is completely null.",
                            dataset_name,
                            column,
                        )
                        validation_passed = False

            if validation_passed:
                logger.info("Missing value validation passed.")
            else:
                logger.error("Missing value validation failed.")

            return validation_passed

        except Exception as e:
            raise CustomException(e, sys)

    def validate_duplicates(self) -> bool:
        logger.info("Starting duplicate validation.")

        try:

            datasets = {
                "Train": read_csv_file(self.ingestion_artifact.train_data_path),
                "Test": read_csv_file(self.ingestion_artifact.test_data_path),
            }

            validation_passed = True

            for dataset_name, df in datasets.items():

                duplicate_count = df.duplicated().sum()

                if duplicate_count > 0:
                    logger.error(
                        "%s dataset contains %d duplicate rows.",
                        dataset_name,
                        duplicate_count,
                    )
                    validation_passed = False
                else:
                    logger.info(
                        "%s dataset contains no duplicate rows.",
                        dataset_name,
                    )

            if validation_passed:
                logger.info("Duplicate validation passed.")
            else:
                logger.error("Duplicate validation failed.")

            return validation_passed

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        logger.info("Starting data validation pipeline.")

        try:

            validation_results = {
                "schema_validation": self.validate_schema(),
                "datatype_validation": self.validate_dtypes(),
                "missing_value_validation": self.validate_missing_values(),
                "duplicate_validation": self.validate_duplicates(),
            }

            validation_results["overall_validation"] = all(
                validation_results.values()
            )

            # Save validation report
            write_json_file(
                file_path=self.validation_config.validation_report_file_path,
                data=validation_results,
            )

            # Save validation status
            status = (
                "Validation Status : PASSED"
                if validation_results["overall_validation"]
                else "Validation Status : FAILED"
            )

            write_text_file(
                file_path=self.validation_config.validation_status_file_path,
                content=status,
            )

            logger.info(
                "Data validation completed with status: %s",
                status,
            )

            return DataValidationArtifact(
                validation_status=validation_results["overall_validation"],
                validation_report_path=self.validation_config.validation_report_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)