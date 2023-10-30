import os
import joblib
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.data_transformation_config = config

    def get_data_transformer_object(self):
        try:
            # Define your numerical and categorical columns
            numerical_columns = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
            categorical_columns = ["Sex", "Embarked"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(drop="first"))
                ]
            )

            # Combine num_pipeline and cat_pipeline using ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            print(f"An error occurred: {str(e)}")

    def initiate_data_transformation(self, train_data, test_data):
        try:
            # Assuming "Survived" is the target column in the training dataset
            target_column_name = "Survived"

            # Prepare your training and testing data
            X_train = train_data.drop(columns=[target_column_name], axis=1)
            y_train = train_data[target_column_name]
            X_test = test_data  # We don't have the target column in the testing data

            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # Save the preprocessor object using joblib
            joblib.dump(preprocessor, self.data_transformation_config.preprocessor_obj_file_path)

            return X_train_transformed, y_train, X_test_transformed

        except Exception as e:
            print(f"An error occurred: {str(e)}")
