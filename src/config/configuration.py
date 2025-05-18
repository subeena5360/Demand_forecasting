from src.entity.config_entity import dataingestion_config,DataValidationConfig,DataTransformationConfig
from src.entity.config_entity import ModelTrainerConfig,ModelEvaluationConfig
from src.constants import *
from src.utils.common import read_yaml,create_directories


class ConfiguarationManager:
    def __init__(
        self,
        config_file = config_file_path,
    
        schema_file = schema_file_path,
        params_file = params_file_path):

        self.config = read_yaml(config_file)
        self.schema = read_yaml(schema_file)
        self.params = read_yaml(params_file)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self)->dataingestion_config:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = dataingestion_config(
            root_dir = config.root_dir,
            source_url = config.source_url,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_config
    
    def get_data_validation_config(self)->DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS
        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir = config.root_dir,
            unzip_data_dir = config.unzip_data_dir,
            STATUS_FILE = config.STATUS_FILE,
            all_schema = schema
        )
        return data_validation_config
    
    def get_data_transformation_config(self)->DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir = config.root_dir,
            data_dir = config.data_dir,
            scaler_path = config.scaler_path,
            model_feature_path = config.model_feature_path,
            one_hot_encoder_path = config.one_hot_encoder_path,
            transformed_data = config.transformed_data,
            preprocessed_dir = config.preprocessed_dir
        )
        return data_transformation_config
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.GradientBoostingRegressor
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config.root_dir,
            train_data_path = config.train_data_path,
            test_data_path = config.test_data_path,
            model_name = config.model_name,
            min_samples_split = params.min_samples_split,
            max_features = params.max_features,
            n_estimators = params.n_estimators,
            target_column = schema.name
            
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.GradientBoostingRegressor
        schema =  self.schema.TARGET_COLUMN

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            test_data_path=config.test_data_path,
            model_path = config.model_path,
            all_params=params,
            metric_file_name = config.metric_file_name,
            target_column = schema.name
           
        )

        return model_evaluation_config