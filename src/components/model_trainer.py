import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump

class ModelTrainer:
    def initiate_model_trainer(self, X_train, y_train, save_directory):
        # Split data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        # Models to train
        models = [
            ('Random Forest', RandomForestRegressor(), {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20],
            }),
            ('Decision Tree', DecisionTreeRegressor(), {
                'max_depth': [None, 10, 20],
            }),
            ('Gradient Boosting', GradientBoostingRegressor(), {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
            }),
            ('Linear Regression', LinearRegression(), {}),
            ('AdaBoost Regressor', AdaBoostRegressor(), {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
            }),
        ]

        best_model = None
        best_mse = float('inf')  # Initialize with a large value

        for model_name, model, param_grid in models:
            # Hyperparameter Tuning
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)

            # Select the Best Model
            best_model_candidate = grid_search.best_estimator_
            val_predictions = best_model_candidate.predict(X_val)
            mse = mean_squared_error(y_val, val_predictions)

            # Check if this model is better than the current best
            if mse < best_mse:
                best_mse = mse
                best_model = best_model_candidate

        # Save the Best Model
        dump(best_model, os.path.join(save_directory, 'best_model.joblib'))

        return best_model
