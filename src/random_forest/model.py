from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

import pandas as pd
import numpy as np
import logging
import csv

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 360
matplotlib.rcParams['text.usetex'] = True


class RandomForestModel:
    def __init__(self, params, logger):
        """
        Initialize the RandomForestModel with specified parameters and a logger.

        Args:
            params (dict): Parameters to configure the RandomForestRegressor.
            logger (logging.Logger): Logger object for logging information during processing.
        """
        self.model = RandomForestRegressor(**params)
        self.logger = logger


    def read_csv(self, filename):
        """
        Read data from a CSV file and preprocess it into a format suitable for regression analysis.

        Args:
            filename (str): The path to the CSV file.

        Returns:
            list: A list of lists, where each sublist represents a data row with numerical features.

        Raises:
            IOError: If there is an issue reading the file or processing its contents.
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
            raise IOError(f'Failed to read or process the file {filename}: {e}')
        return rosette


    def train(self, r, x_train, y_train, test_size=0.8):
        """
        Train the RandomForestRegressor model using the provided dataset.

        Args:
            x_train (list of lists): Training features.
            y_train (list): Target variable for training.
            test_size (float): The proportion of the dataset to use as test set.

        Notes:
            This method also logs the training results.
        """
        X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        pc = pearsonr(y_test, y_pred)[0]

        train_data = {'train-rosette': r,
                      'train_size': 1-test_size,
                      'r2': r2, 'mse': mse, 'pc': pc}
        self.logger.info(train_data)


    def predict(self, x_test):
        """
        Predict the target values using the trained model for the provided test data.

        Args:
            x_test (array-like): Test features.

        Returns:
            array: Predicted values.
        """
        return self.model.predict(x_test)

    def score(self, x_test, y_test):
        """
        Evaluate the model performance using the test dataset.

        Args:
            x_test (array-like): Test features.
            y_test (array-like): True values for the test set.

        Returns:
            dict: A dictionary containing R-squared, MSE, and Pearson correlation values.
        """
        predictions = self.predict(x_test)
        r2 = r2_score(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        pc = pearsonr(y_test, predictions)[0]
        return {'r2': r2, 'mse': mse, 'pc': pc}

    def plot_feature_importances(self, feature_names, path):
        """
        Plot and save the feature importances of the trained RandomForest model.

        Args:
            feature_names (list): Names of the features corresponding to importances.
            path (str): Path where the plot will be saved.
        """
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        cmap = sns.color_palette("mako", as_cmap=True)
        colors = cmap(importances/importances.max())  # normalize importances for cmap
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(feature_names, importances, yerr=std, color=colors, capsize=5)
        plt.title('Random Forest Model - Feature Importances', y=1.03)
        ax.set_ylabel('Mean decrease in impurity')
        fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=importances.min(), vmax=importances.max())),
                    ax=ax, orientation='vertical', label='Importance')
        fig.tight_layout(pad=3)

        plt.savefig(path)
        plt.close()