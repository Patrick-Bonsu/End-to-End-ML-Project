import pandas as pd
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str  # Specify the path to your training dataset
    test_data_path: str  # Specify the path to your testing dataset

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.ingestion_config = config

    def initiate_data_ingestion(self):
        try:
            train_data = pd.read_csv(self.ingestion_config.train_data_path)
            test_data = pd.read_csv(self.ingestion_config.test_data_path)

            # Handle missing values in the training and testing datasets as needed.

            return train_data, test_data
        except Exception as e:
            print(f"An error occurred: {str(e)}")
