import os
import pandas as pd
from joblib import load

def predict_and_save_data(data_path, output_directory, preprocessor_path, input_data):

    # Load the best model
    best_model = load(os.path.join(output_directory, 'best_model.joblib'))

    # Load the preprocessor to transform the test data
    preprocessor = load(preprocessor_path)  # Use the provided preprocessor_path

    # Read the test data
    X_new_test = pd.read_csv(data_path)

    # Transform the test data
    X_new = preprocessor.transform(X_new_test)

    # Make predictions using the best model
    predictions = best_model.predict(X_new)

    # Apply a threshold of 0.7 to convert predictions to 0 or 1
    predictions_binary = [1 if p >= 0.7 else 0 for p in predictions]

    # Create a DataFrame with 'PassengerId' and 'Survived' columns
    passenger_ids = X_new_test['PassengerId']
    output_df = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': predictions_binary})

    # Save the DataFrame to a CSV file in the specified output directory
    output_df.to_csv(os.path.join(output_directory, 'output.csv'), index=False)



def predict_from_input(input_data, preprocessor_path, model_path):
    # Load the preprocessor
    preprocessor = load(preprocessor_path)

    # Prepare the input data as a DataFrame
    input_data_df = pd.DataFrame([input_data])

    # Transform the input data using the preprocessor
    X_new = preprocessor.transform(input_data_df)

    # Load the best model
    best_model = load(model_path)

    # Make predictions using the best model
    predictions = best_model.predict(X_new)

    # Apply a threshold of 0.7 to convert predictions to 0 or 1
    predictions_binary = [1 if p >= 0.7 else 0 for p in predictions]

    return predictions_binary
