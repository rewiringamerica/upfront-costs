"""A utility file to provide common modeling helper classes and functions in the upfront cost repo."""

import json
import os
import tempfile

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, median_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split


class ModelHelper:
    """
    A helper class for training machine learning models, tuning hyperparameters, and logging results to MLflow.

    Attributes
    ----------
    data : pandas.DataFrame
        The dataset containing features and target variable.
    features : list
        A list of feature column names.
    target : str
        The name of the target column.
    experiment_name : str
        The name of the MLflow experiment.
    model : estimator
        The machine learning model to be trained. Can be a sklearn estimator
    param_grid : dict
        The hyperparameters to be tuned.
    scorer : sklearn scorer
        The scoring function for model evaluation.
    X : pandas.DataFrame
        The feature data.
    y : pandas.Series
        The target data.
    X_train : pandas.DataFrame
        The training feature data.
    X_test : pandas.DataFrame
        The test feature data.
    y_train : pandas.Series
        The training target data.
    y_test : pandas.Series
        The test target data.
    best_model : sklearn estimator
        The best model after hyperparameter tuning.
    best_params : dict
        The best hyperparameters after tuning.
    val_predictions : numpy.ndarray
        The predictions on the test set.
    """

    def __init__(self, data, features, target, experiment_name, *, test_size=0.2, random_state=42):
        """
        Initialize the ModelHelper class with data, features, target, and experiment name.

        Splits the data into training and test sets.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset containing features and target variable.
        features : list
            A list of feature column names.
        target : str
            The name of the target column.
        experiment_name : str
            The name of the MLflow experiment.
        test_size : float, optional (default=0.2)
            The proportion of the dataset to include in the validation set.
        random_state : int, optional (default=42)
            The seed used by the random number generator.

        Returns
        -------
            None
        """
        self.data = data
        self.features = features
        self.target = target
        self.experiment_name = experiment_name
        self.model = None
        self.param_grid = None

        self.X = self.data[self.features]
        self.y = self.data[self.target]

        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, shuffle=True
        )

    def train_model(self, model, scorer, param_grid, cv=5, n_iter=100, random_state=42):
        """
        Set the machine learning model and hyperparameters to be tuned.

        Performs a randomized search and logs the metrics of each permutation of hyperparameters.

        Parameters
        ----------
        model : estimator
            The machine learning model to be trained. Can be a sklearn estimator.
        param_grid : dict
            The hyperparameters to be tuned.
        scorer : callable
            Scorer created by sklearn.metrics.make_scorer().
        cv : int
            Number of folds to use in k-folds cross validation. Defaults to 5.
        n_iter : int
            Number of parameter settings that are sampled. Defaults to 100.
        random_state: int
            Random state to be used to recreate model training.

        Returns
        -------
            None
        """
        self.model = model
        self.param_grid = param_grid
        self.scorer = scorer

        mlflow.autolog(disable=True)

        randomized_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=self.param_grid,
            cv=cv,
            scoring=self.scorer,
            verbose=3,
            n_iter=n_iter,
            random_state=random_state,
        )  # Set a seed for reproducibility

        randomized_search.fit(self.X_train, self.y_train)

        # Create a list to store search results
        results = []

        for i, params in enumerate(randomized_search.cv_results_["params"]):
            mean_test_score = -randomized_search.cv_results_["mean_test_score"][i]
            std_test_score = randomized_search.cv_results_["std_test_score"][i]
            params["mean_test_mae"] = mean_test_score
            params["std_test_mae"] = std_test_score

            # Append the result to the list
            results.append(params)

        # Convert the results list to a DataFrame
        self.results_df = pd.DataFrame(results)

        # Save the best model and best parameters to the object
        self.best_model = randomized_search.best_estimator_
        self.best_params = randomized_search.best_params_
        self.test_predictions = randomized_search.predict(self.X_test)

    def log_model(self, model_name):
        """
        Log the best model found during grid search along with its parameters and metrics to MLflow.

        This method performs the following steps:
        1. Logs the best hyperparameters found during the grid search.
        2. Uses the best model to make predictions on the validation set.
        3. Calculates evaluation metrics on the test set:
        - Median Absolute Error (MAE)
        - Weighted Median Absolute Percentage Error (WMAPE)
        - Median Absolute Percentage Error (MdAPE)
        - Variance of the percentage errors (MdAPE Variance)
        - Root Mean Squared Error (RMSE)
        4. Logs the evaluation metrics to MLflow.
        5. Logs the best model to MLflow.
        6. Logs the features used for training as a JSON artifact.
        7. Logs the predictions, actual values, and errors as a CSV artifact.

        Assumes the grid search has been completed and the best model and parameters are stored in `self.best_model`
        and `self.best_params` respectively.

        Parameters
        ----------
        model_name : string
            The name of the model to be logged.

        Returns
        -------
            None
        """
        mlflow.set_experiment(self.experiment_name)
        # start and name the model run
        with mlflow.start_run(run_name=model_name) as _:
            # Log best parameters
            mlflow.log_params(self.best_params)

            # Predict using the best model
            predictions = self.best_model.predict(self.X_test)

            # Calculate metrics
            test_val_mae = median_absolute_error(self.y_test, predictions)
            test_val_wmape = self.weighted_mean_absolute_percentage_error(self.y_test, predictions)
            test_val_mdape = self.median_absolute_percentage_error(self.y_test, predictions)
            test_percentage_errors = np.abs((self.y_test - predictions) / self.y_test) * 100
            test_val_mdape_variance = np.var(test_percentage_errors)
            test_val_rmse = np.sqrt(mean_squared_error(self.y_test, predictions))

            # Log metrics
            mlflow.log_metric("val_mae", test_val_mae)
            mlflow.log_metric("val_wmape", test_val_wmape)
            mlflow.log_metric("val_mdape", test_val_mdape)
            mlflow.log_metric("val_mdape_variance", test_val_mdape_variance)
            mlflow.log_metric("val_rmse", test_val_rmse)

            # Log the best model
            mlflow.sklearn.log_model(self.best_model, "model", input_example=self.X_test.iloc[[0]])

            # Log the features used
            features_dict = {"features": self.features}
            with tempfile.TemporaryDirectory() as tempdir:
                features_path = os.path.join(tempdir, "features.json")
                with open(features_path, "w") as f:
                    json.dump(features_dict, f)
                mlflow.log_artifact(features_path)

            # Log the grid search results using tempdirectory
            with tempfile.TemporaryDirectory() as tempdir:
                results_path = os.path.join(tempdir, "randomized_search_results.csv")
                self.results_df.to_csv(results_path, index=False)
                mlflow.log_artifact(results_path)

            # Log predictions, actual values, and errors
            results_df = pd.DataFrame(
                {
                    "actual": self.y_test,
                    "predicted": predictions,
                    "error": self.y_test - predictions,
                }
            )
            # Log the predictions using tempdirectory
            with tempfile.TemporaryDirectory() as tempdir:
                predictions_path = os.path.join(tempdir, "predictions.csv")
                results_df.to_csv(predictions_path, index=False)
                mlflow.log_artifact(predictions_path)

            print(
                f"Metrics logged: val_mae={test_val_mae},\
                    val_wmape={test_val_wmape},\
                    val_mdape={test_val_mdape},\
                    val_mdape_variance={test_val_mdape_variance},\
                    val_rmse={test_val_rmse}"
            )

    @staticmethod
    def weighted_mean_absolute_percentage_error(y_true, y_pred):
        """
        Calculate the Weighted Median Absolute Percentage Error (WMAPE).

        Parameters
        ----------
        y_true : array-like
            The true target values.
        y_pred : array-like
            The predicted target values.

        Returns
        -------
        float
            The WMAPE value.
        """
        return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true) * 100

    @staticmethod
    def median_absolute_percentage_error(y_true, y_pred):
        """
        Calculate the Median Absolute Percentage Error (MdAPE).

        Parameters
        ----------
        y_true : array-like
            The true target values.
        y_pred : array-like
            The predicted target values.

        Returns
        -------
        float
            The MdAPE value.
        """
        return np.median(np.abs((y_true - y_pred) / y_true)) * 100
