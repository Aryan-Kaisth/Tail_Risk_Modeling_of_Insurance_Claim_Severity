from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path = Path("artifacts") / "data_ingestion" / "raw.csv"
    train_data_path: Path = Path("artifacts") / "data_ingestion" / "train.csv"
    test_data_path: Path = Path("artifacts") / "data_ingestion" / "test.csv"
    test_size: float = 0.2
    random_state: int = 42
    shuffle: bool = True