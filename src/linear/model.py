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
        train(rosette_ids, all_data, test_size): Trains the model on the combined dataset.
        predict(X): Predicts target values using the model.
        score(r, X, y): Computes and logs performance metrics of the model on a test set.
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
            list: A list of preprocessed data points in the form [galaxy_id, mass, mag_r, mag_g, ... mag_w2, redshift].
                  (Exact structure depends on how many columns are in the CSV.)
        """
        rosette = []
        try:
            with open(filename, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    if row:
                        values = list(row.values())  # might be [id, mass, flux_r, flux_g, flux_z, flux_w1, flux_w2, z]
                        n = [float(values[0]), float(values[1])]
                        flux_values = [float(v) for v in values[2:-1]]
                        mags = 22.5 - 2.5 * np.log10(flux_values)
                        n.extend(mags)
                        n.append(float(values[-1]))
                        rosette.append(n)
        except Exception as e:
            raise IOError(f"Failed to read or process the file {filename}: {e}")
        return rosette

    def train(self, rosette_ids, all_data, test_size):
        """
        Trains the linear regression model on the combined data of possibly multiple rosettes.

        Parameters:
            rosette_ids (list of int): The identifiers of rosettes used for training.
            all_data (list): The combined dataset (each entry is [galaxy_id, mass, mag_r, ..., redshift]).
            test_size (float): If > 0, we do an internal train/test split for sanity-check metrics.

        Returns:
            None
        """
        X = [n[2:] for n in all_data]
        y = [n[1] for n in all_data]

        if test_size > 0:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=True)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            pc = pearsonr(y_test, y_pred)[0]
            self.logger.info({'train-rosettes': rosette_ids,
                              'train_size': 1 - test_size,
                              'r2': r2,
                              'mse': mse,
                              'pc': pc})
        else:
            self.model.fit(X, y)
            self.logger.info({'train-rosettes': rosette_ids,
                              'train_size': '100%',
                              'r2': None,
                              'mse': None,
                              'pc': None})

    def predict(self, X):
        """
        Predicts the target values using the trained model.

        Parameters:
            X (array-like): The input features for making predictions.

        Returns:
            array: The predicted values.
        """
        return self.model.predict(X)

    def score(self, r, X, y):
        """
        Evaluates the model's performance on the specified dataset.

        Parameters:
            r (int): The identifier for the rosette being evaluated (for logging).
            X (array-like): The input features for evaluation.
            y (array-like): The actual target values for evaluation.

        Returns:
            dict: A dictionary containing evaluation metrics (MSE, R^2, Pearson correlation).
            array: The predicted values.
        """
        predictions = self.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        pc = pearsonr(y, predictions)[0] if len(y) > 1 else 0.0

        test_data = {'test-rosette': r,
                     'r2': r2,
                     'mse': mse,
                     'pc': pc}
        self.logger.info(test_data)
        return test_data, predictions