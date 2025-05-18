import pandas as pd
import os
from src import logger
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from src.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        
        test_y = test_data[[self.config.target_column]]


        grb = GradientBoostingRegressor(max_features=self.config.max_features, 
                                       min_samples_split=self.config.min_samples_split,
                                       n_estimators=self.config.n_estimators,
                                       random_state=42)
        grb.fit(train_x, train_y.values.ravel())

        joblib.dump({'model': grb,'feature_names': train_x.columns.tolist()}, os.path.join(self.config.root_dir, self.config.model_name))