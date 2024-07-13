#!/usr/bin/env python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import csv


class LinearRegressionModel:
    """
    A class for managing the lifecycle of a linear regression model including training, prediction, and evaluation.

    Attributes:
        model (LinearRegression): An instance of scikit-learn's LinearRegression.
        logger (Logger): A logger for recording model performance and operational events.

    Methods:
        read_csv(filename): Reads and processes data from a CSV file.
        train(r, rosette, test_size): Trains the model using specified data and test size.
        predict(X): Predicts target values using the model.
        score(r, X, y): Computes and logs performance metrics of the model.
    """

    def __init__(self, logger):
        """
        Initializes a LinearRegressionModel instance with a logging object.

        Parameters:
            logger (Logger): An initialized and configured Logger object.
        """
        self.model = LinearRegression()
        self.logger = logger


    def read_csv(self, filename):
        """
        Reads a CSV file and preprocesses its contents for model training or prediction.

        Parameters:
            filename (str): The path to the CSV file.

        Returns:
            list: A list of preprocessed data points.

        Raises:
            IOError: If the file cannot be read.
        """
        rosette = []
        try:
            with open(filename, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    if row:
                        values = list(row.values())
                        n = [float(values[0]), float(values[1])]
                        n.extend(22.5 - 2.5 * np.log10([float(v) for v in values[2:-1]]))
                        n.append(float(values[-1]))
                        rosette.append(n)
        except Exception as e:
            raise IOError(f"Failed to read or process the file {filename}: {e}")
        return rosette


    def train(self, r, rosette, test_size):
        """
        Trains the linear regression model using the provided dataset.

        Parameters:
            r (int): The identifier for the dataset.
            rosette (list): The dataset to train the model on.
            test_size (float): The proportion of the dataset to be used as test set.

        Returns:
            LinearRegression: The trained model instance.
        """
        X, y = [n[2:] for n in rosette], [n[1] for n in rosette]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        pc = pearsonr(y_test, y_pred)[0]

        train_data = {
            'train-rosette': r,
            'train_size': 1 - test_size,
            'r2': r2,
            'mse': mse,
            'pc': pc
        }
        self.logger.info(train_data)
        return self.model


    def predict(self, X):
        """
        Predicts the target values using the trained model.

        Parameters:
            X (array-like): The input data for making predictions.

        Returns:
            array: The predicted values.
        """
        return self.model.predict(X)


    def score(self, r, X, y):
        """
        Evaluates the model's performance on the specified dataset.

        Parameters:
            r (int): The identifier for the dataset being evaluated.
            X (array-like): The input features for evaluation.
            y (array-like): The actual target values for evaluation.

        Returns:
            dict: A dictionary containing evaluation metrics such as MSE, R^2, and Pearson correlation.
            array: The predicted values.
        """
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        pc = pearsonr(y, predictions)[0]

        test_data = {
            'test-rosette': r,
            'r2': r2,
            'mse': mse,
            'pc': pc
        }
        self.logger.info(test_data)
        return test_data, predictions