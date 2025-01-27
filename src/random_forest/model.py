from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import numpy as np
import logging

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 360
matplotlib.rcParams['text.usetex'] = True


class RandomForestModel:
    def __init__(self, params, logger):
        """
        Initialize the RandomForestModel with specified hyperparams and a logger.
        """
        self.model = RandomForestRegressor(**params)
        self.logger = logger

    def train(self, r, x_train, y_train, test_size=0.0):
        """
        Train the RandomForestRegressor on x_train, y_train.
        If test_size>0, do an internal check. Otherwise train on full x_train.
        """
        if test_size > 0:
            from sklearn.model_selection import train_test_split
            X_tr, X_ts, y_tr, y_ts = train_test_split(x_train, y_train, test_size=test_size, random_state=42)
            self.model.fit(X_tr, y_tr)
            y_pred = self.model.predict(X_ts)

            r2 = r2_score(y_ts, y_pred)
            mse = mean_squared_error(y_ts, y_pred)
            pc = pearsonr(y_ts, y_pred)[0] if len(y_ts) > 1 else 0.0

            train_data = {
                'train-rosette': r,
                'train_size': f"{(1 - test_size)*100}%",
                'r2': r2,
                'mse': mse,
                'pc': pc
            }
            if self.logger:
                self.logger.info(train_data)
        else:
            self.model.fit(x_train, y_train)
            if self.logger:
                self.logger.info({
                    'train-rosette': r,
                    'train_size': '100%',
                    'r2': None,
                    'mse': None,
                    'pc': None
                })

    def predict(self, x_test):
        """
        Predict target values for x_test.
        """
        return self.model.predict(x_test)

    def score(self, x_test, y_test):
        """
        Evaluate model on x_test, y_test and return dict of R^2, MSE, Pearson correlation.
        """
        y_pred = self.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        pc = pearsonr(y_test, y_pred)[0] if len(y_test) > 1 else 0.0
        return {'r2': r2, 'mse': mse, 'pc': pc}

    def plot_feature_importances(self, feature_names, path):
        """
        Saves a bar plot of feature importances.
        """
        importances = self.model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)

        cmap = sns.color_palette("mako", as_cmap=True)
        colors = cmap(importances / importances.max())

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(feature_names, importances, yerr=std, color=colors, capsize=5)
        ax.set_ylabel('Mean decrease in impurity')
        plt.title('Random Forest - Feature Importances', y=1.03)
        fig.tight_layout(pad=3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=importances.min(), vmax=importances.max()))
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Feature Importance')

        plt.savefig(path, dpi=300)
        plt.close(fig)