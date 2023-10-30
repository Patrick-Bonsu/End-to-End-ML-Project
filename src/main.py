import os
from components.Data_ingestion import DataIngestion, DataIngestionConfig
from components.Data_transformation import DataTransformation, DataTransformationConfig
from components.model_trainer import ModelTrainer
from pipeline.predict_pipeline import predict_and_save_data

# Specify file paths
train_data_path = r"C:\Users\TRY'S COMPUTERS\Desktop\Titanic\train.csv"
test_data_path = r"C:\Users\TRY'S COMPUTERS\Desktop\Titanic\test.csv"
save_directory = r"C:\Users\TRY'S COMPUTERS\Desktop\Titanic\src\artifacts"

# Data Ingestion
data_ingestion_config = DataIngestionConfig(train_data_path=train_data_path, test_data_path=test_data_path)
data_ingestion = DataIngestion(data_ingestion_config)
train_data, test_data = data_ingestion.initiate_data_ingestion()

# Data Transformation
data_transformation_config = DataTransformationConfig(preprocessor_obj_file_path=os.path.join(save_directory, "preprocessor.pkl"))
data_transformation = DataTransformation(data_transformation_config)
X_train_transformed, y_train, X_test_transformed = data_transformation.initiate_data_transformation(train_data, test_data)

# Model Training
trainer = ModelTrainer()
best_model = trainer.initiate_model_trainer(X_train_transformed, y_train, save_directory)


# Prediction Pipeline
preprocessor_path = os.path.join(save_directory, 'preprocessor.pkl')  # Provide the correct preprocessor path
predict_and_save_data(test_data_path, save_directory, preprocessor_path)
