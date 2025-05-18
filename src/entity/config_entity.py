from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class dataingestion_config:
    root_dir : Path
    source_url : str
    local_data_file : Path
    unzip_dir : Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir : Path
    unzip_data_dir : Path
    STATUS_FILE : str
    all_schema : dict 

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir : Path
    data_dir : Path
    scaler_path : Path
    model_feature_path : Path
    one_hot_encoder_path :Path
    transformed_data : Path
    preprocessed_dir : Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    max_features: int
    min_samples_split: int
    n_estimators : int
    target_column: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: dict
    metric_file_name: Path
    target_column: str

