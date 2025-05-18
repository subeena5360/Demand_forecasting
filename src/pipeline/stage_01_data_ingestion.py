from src.config.configuration import ConfiguarationManager
from src.components.data_ingestion import DataIngestion

from src import logger



STAGE_NAME = " Data Ingestion Stage"

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfiguarationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()
if __name__ =="__main__":
     STAGE_NAME = "Data Ingestion stage"
     try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
     except Exception as e:
        logger.exception(e)
        raise e