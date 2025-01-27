from optimization import ModelOptimization
from model import RandomForestModel
from sklearn.model_selection import train_test_split
import optuna
import shap

from datetime import datetime
import pandas as pd
import numpy as np
import argparse
import logging
import csv
import os

import matplotlib
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 360
matplotlib.rcParams['text.usetex'] = True
logging.getLogger('shap').setLevel(logging.WARNING)


def setup_logging():
    """
    Sets up a centralized logging system for the entire application.
    """
    log_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'job_management', 'logs')
    )
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'random_forest.log'),
                        level=logging.INFO,
                        format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger('random_forest')


def configure_args():
    parser = argparse.ArgumentParser(description='Model training and evaluation on rosettes data')
    parser.add_argument('--train_rosettes', type=str, required=True,
                        help='Comma-separated list of rosettes IDs for training')
    parser.add_argument('--test_rosettes', type=str, required=False, default='',
                        help='Comma-separated list of rosettes IDs for testing')
    parser.add_argument('--test_size', type=float, required=False, default=0.2,
                        help='Proportion of the dataset for internal validation (0.0 means no split)')
    parser.add_argument('--change_data', type=str, required=False, default='',
                        help='Columns to remove from training, e.g., flux_r,flux_g, etc.')
    parser.add_argument('--optimize', action='store_true', required=False, default=False,
                        help='Optimize hyperparameters using Optuna')
    parser.add_argument('--add_rand_col', action='store_true', required=False, default=False,
                        help='Include a random column in the dataset')
    parser.add_argument('--plot', action='store_true', required=False, default=False,
                        help='Enable plotting of results (predictions)')
    parser.add_argument('--feat_imp', action='store_true', required=False, default=False,
                        help='Enable plotting of feature importances')
    parser.add_argument('--shap', action='store_true', required=False, default=False,
                        help='Enable plotting of SHAP values')
    return parser.parse_args()


def read_csv_file(filename):
    """
    Reads numerical data from a CSV file, converting fluxes to magnitudes.

    Returns:
        list: Each row is [id, mass, mag_r, mag_g, mag_z, mag_w1, mag_w2, redshift]
    """
    rosette = []
    try:
        with open(filename, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if row:
                    values = list(row.values())
                    n = [float(values[0]), float(values[1])]
                    fluxes = [float(v) for v in values[2:-1]]
                    mags = 22.5 - 2.5 * np.log10(fluxes)
                    n.extend(mags)
                    n.append(float(values[-1]))
                    rosette.append(n)
    except Exception as e:
        raise IOError(f'Failed to read or process file {filename}: {e}')
    return rosette


def add_rand_column(X):
    """
    Appends a column of random values to a list of data rows.
    """
    random_col = np.random.rand(len(X))
    for i in range(len(X)):
        X[i].append(random_col[i])
    return X


def load_and_concatenate(rosettes_list, base_path):
    """
    Loads and concatenates data from each rosette ID in `rosettes_list`.
    Returns a single list of rows (each row is [id, mass, mag_r, mag_g, mag_z, mag_w1, mag_w2, z]).
    """
    all_rows = []
    for r in rosettes_list:
        path = os.path.join(base_path, f'rosette{r}.csv')
        data = read_csv_file(path)
        all_rows.extend(data)
    return all_rows


def quit_columns(rosette, remove_cols, columns_dict):
    """
    Removes specified columns from rosette data. Also extracts X and y.

    rosette: list of rows [id, mass, mag_r, mag_g, mag_z, mag_w1, mag_w2, z]
    remove_cols: e.g. ['flux_r', 'flux_g'] but you must map them to actual indexes
    columns_dict: e.g. {'flux_r':2,'flux_g':3,'flux_z':4,'flux_w1':5,'flux_w2':6,'z':7}

    Returns (X, y, indices_used).
    """
    keep_indices = []
    for name, idx in columns_dict.items():
        if name not in remove_cols:
            keep_indices.append(idx)

    X = []
    y = []
    for row in rosette:
        y.append(row[1])  # mass
        feats = [row[i] for i in keep_indices]
        X.append(feats)
    return X, y, keep_indices


def plot(model, x, y, columns, quit_columns, args, r):
    """
    Handles the plotting of predictions and SHAP values based on the trained model and user specs.

    Parameters:
        model (RandomForestModel): The trained model.
        x (list): Feature data for plotting.
        y (list): Target values (masses).
        columns (dict): Mapping of feature names to their indexes.
        quit_columns (list): Columns excluded from training.
        args (Namespace): Command-line arguments.
        r (int): Rosette number (for naming plots).

    Description:
        - Generates prediction plots (true vs. predicted).
        - Generates SHAP beeswarm plots (if requested).
    """
    def prepare_plot_path(suffix, drop_info=''):
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'random_forest'))
        os.makedirs(base_dir, exist_ok=True)
        now = datetime.now()
        fl = now.strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'{suffix}_{fl}.png'
        return os.path.join(base_dir, file_name)

    def plot_predictions(x, y, quit_columns, model, args, r):
        pred = model.predict(x)
        slope, intercept = np.polyfit(y, pred, 1)
        fit = slope * np.array(y) + intercept
        dif = np.abs(pred - fit)
        rng = (np.max(dif) - np.min(dif)) or 1e-12
        color_values = 1 - (dif - np.min(dif)) / rng

        fig, ax = plt.subplots(figsize=(8, 6))
        cmap = sns.color_palette("mako", as_cmap=True)
        scatter = ax.scatter(y, pred, c=color_values, cmap=cmap, s=5.5)
        ax.plot(y, fit, color='black', ls='--', linewidth=1)
        ax.set_ylabel('Prediction')
        ax.set_xlabel('Data')
        plt.colorbar(scatter, ax=ax, label='Proximity to Fit', pad=0.07)
        plt.grid(True)

        drop_info = 'Excluding ' + ', '.join(quit_columns) if quit_columns else ''
        plt.title(f'Rosette {r} - $\\log M_{{*}}$ (Random Forest)\n{drop_info}', y=1.03)
        plot_path = prepare_plot_path('pred', drop_info)
        plt.savefig(plot_path, dpi=360)
        plt.close(fig)

    def plot_shap_values(x, y, columns, quit_columns, model, args, r):
        _, x_test, _, _ = train_test_split(x, y, test_size=args.test_size, random_state=42)
        feature_cols = [col for col in columns if col not in quit_columns]
        new_x_test = pd.DataFrame(x_test, columns=feature_cols)

        explainer = shap.Explainer(model.model, new_x_test)
        shap_values = explainer(new_x_test)

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values,
                            color=sns.color_palette("mako", as_cmap=True),
                            plot_size=(9, 5),
                            show=False)
        fig.tight_layout(pad=3)

        drop_info = 'Excluding ' + ', '.join(quit_columns) if quit_columns else ''
        plt.title(f'Rosette {r} - SHAP Values\n{drop_info}')
        plot_path = prepare_plot_path('shap', drop_info)
        plt.savefig(plot_path, dpi=360)
        plt.close(fig)

    if args.plot:
        plot_predictions(x, y, quit_columns, model, args, r)

    if args.shap:
        plot_shap_values(x, y, columns, quit_columns, model, args, r)


def test_model_on_rosettes(model, test_rosettes, base_path, columns_dict, quit_cols, add_rand, args):
    """
    Evaluate the trained model on each rosette in test_rosettes.
    Calls 'plot()' to generate the desired predictions/SHAP plots if enabled.
    """
    for r in test_rosettes:
        path = os.path.join(base_path, f'rosette{r}.csv')
        rosette_data = read_csv_file(path)

        X_temp, y_temp, _ = quit_columns(rosette_data, quit_cols, columns_dict)
        if add_rand:
            X_temp = add_rand_column(X_temp)

        metrics = model.score(X_temp, y_temp)
        print(f'Metrics for rosette {r}: {metrics}')

        if args.plot or args.shap:
            plot(model, X_temp, y_temp, columns_dict, quit_cols, args, r)


def main():
    args = configure_args()
    logger = setup_logging()

    columns_dict = {'mag_r': 2, 'mag_g': 3,
                    'mag_z': 4, 'mag_w1':5,
                    'mag_w2':6, 'z':7}

    train_list = [int(x) for x in args.train_rosettes.split(',')]
    test_list = [int(x) for x in args.test_rosettes.split(',')] if args.test_rosettes else []
    quit_cols = [c.strip() for c in args.change_data.split(',') if c.strip()]

    base_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    all_rows = load_and_concatenate(train_list, base_path)
    X_all, y_all, used_indices = quit_columns(all_rows, quit_cols, columns_dict)
    if args.add_rand_col:
        X_all = add_rand_column(X_all)

    if args.test_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_all, y_all,
                                                          test_size=args.test_size,
                                                          random_state=42)
    else:
        X_train, y_train = X_all, y_all
        X_val, y_val = [], []

    best_params = {'n_estimators':100, 'random_state':42}
    if args.optimize:
        if len(X_val) == 0:
            print("[WARNING] test_size=0.0 but optimization is True. "
                  "Optuna might rely on cross-validation in optimization.py or produce less reliable results.")
        optimizer = ModelOptimization(X_train, y_train, X_val, y_val)
        optimizer.create_study(n_trials=10)
        best_params = optimizer.best_params
        print(f'Optuna best params: {best_params}, best value: {optimizer.best_value}')

    model = RandomForestModel(best_params, logger)
    model.train(r='combined', x_train=X_train, y_train=y_train, test_size=0.2)

    if len(X_val) > 0:
        val_metrics = model.score(X_val, y_val)
        print("[Internal Validation]", val_metrics)

    if test_list:
        test_model_on_rosettes(model, test_list, base_path, columns_dict, quit_cols, args.add_rand_col, args)

    if args.feat_imp:
        feat_names = []
        sorted_cols = sorted(columns_dict.items(), key=lambda x: x[1])
        for name, idx in sorted_cols:
            if name not in quit_cols and idx in used_indices:
                feat_names.append(name)
        if args.add_rand_col:
            feat_names.append("Random")

        now = datetime.now()
        fl = now.strftime('%Y-%m-%d_%H-%M-%S')
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'random_forest'))
        os.makedirs(out_dir, exist_ok=True)
        path_imp = os.path.join(out_dir, f'feature_importances_{fl}.png')
        model.plot_feature_importances(feat_names, path_imp)


if __name__ == '__main__':
    main()