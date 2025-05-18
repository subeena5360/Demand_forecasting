from src.config.configuration import ConfiguarationManager
from src.components.data_validation import DataValiadtion
from src import logger



STAGE_NAME = " Data Ingestion Stage"

class DataValidationTrainingPipeline:
    def _init_(self):
        pass
    
    def main(self):
        config = ConfiguarationManager()
        data_validation_config = config.get_data_validation_config()
        data_validation = DataValiadtion(config=data_validation_config)
        data_validation.validate_all_columns()
        STAGE_NAME = "Data Validation stage"
try:
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
        logger.exception(e)
        raise e