from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    train_data_path: Path
    test_data_path: Path

@dataclass(frozen=True)
class DataValidationArtifact:
    validation_status: bool
    validation_report_path: Path