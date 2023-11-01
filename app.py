from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import predict_from_input
import os
import pandas as pd
from joblib import load


app = Flask(__name__)
# Define the path for the preprocessor
preprocessor_path = r"C:\Users\TRY'S COMPUTERS\Desktop\Titanic\src\artifacts\preprocessor.pkl"  # Update with the actual path
model_path=r"C:\Users\TRY'S COMPUTERS\Desktop\Titanic\src\artifacts\best_model.joblib"

def prepare_input_data(Pclass, Age, SibSp, Parch, Fare, Sex, Embarked):
    # Create a dictionary with the input data
    input_data = {
        "Pclass": Pclass,
        "Age": Age,
        "SibSp": SibSp,
        "Parch": Parch,
        "Fare": Fare,
        "Sex": Sex,
        "Embarked": Embarked
    }

    return input_data

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = []  # Initialize an empty list for predictions

    if request.method == 'POST':
        # Retrieve input values from the form
        Pclass = int(request.form['Pclass'])
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Sex = request.form['Sex']
        Embarked = request.form['Embarked']

        # Prepare the input data
        input_data = prepare_input_data(Pclass, Age, SibSp, Parch, Fare, Sex, Embarked)

        # Make predictions
        predictions = predict_from_input(input_data, preprocessor_path,model_path)

    return render_template('index.html', predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
