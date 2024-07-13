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
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 360
matplotlib.rcParams['text.usetex'] = True
logging.getLogger('shap').setLevel(logging.WARNING)


def setup_logging():
    """
    Sets up a centralized logging system for the entire application.

    Creates a directory for logs if it doesn't exist and configures logging settings.

    Returns:
        logging.Logger: A logger object configured to write to 'random_forest.log' in the specified logs directory.
    """
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'job_management', 'logs'))
    logging.basicConfig(
        filename=os.path.join(log_dir, 'random_forest.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger('random_forest')


def read_csv(filename):
    """
    Reads numerical data from a CSV file and preprocesses it for model usage.

    Parameters:
        filename (str): The path to the CSV file to be read.

    Returns:
        list: A list of lists, each containing numerical data from one row of the CSV.

    Raises:
        IOError: If the file cannot be read or processed.
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


def configure_args():
    """
    Configures the command-line argument parser for the application.

    Specifies the necessary command-line arguments needed for running the model training and evaluation.

    Returns:
        argparse.Namespace: An object containing all the command-line arguments parsed according to the specifications.
    """
    parser = argparse.ArgumentParser(description='Model training and evaluation on rosettes data')
    parser.add_argument('--train_rosettes', type=str, required=True,
                        help='Comma-separated list of rosettes IDs for training')
    parser.add_argument('--test_rosettes', type=str, required=False, default='',
                        help='Comma-separated list of rosettes IDs for testing, if any')
    parser.add_argument('--test_size', type=float, required=False, default=0.8,
                        help='Proportion of the dataset to include in the test split')
    parser.add_argument('--change_data', type=str, required=False, default='',
                        help='Columns to take out of the training, i.e., flux_r,flux_g,flux_z,flux_w1,flux_w2,z')
    parser.add_argument('--optimize', action='store_true', required=False, default=True,
                        help='Optimize hyperparameters using Optuna')
    parser.add_argument('--add_rand_col', action='store_true', required=False, default=False,
                        help='Include a random column in the dataset')
    parser.add_argument('--plot', action='store_true', required=False, default=False,
                        help='Enable plotting of results')
    parser.add_argument('--feat_imp', action='store_true', required=False, default=False,
                        help='Enable plotting of feature importances')
    parser.add_argument('--shap', action='store_true', required=False, default=False,
                        help='Enable plotting of shap values')
    return parser.parse_args()


def add_rand_column(x):
    """
    Appends a column of random values to a list of data rows.

    Parameters:
        x (list of list): The existing dataset where each inner list represents a data row.

    Returns:
        list: The modified dataset with an additional column of random values.
    """
    random_column = np.random.rand(len(x))
    [x[i].append(random_column[i]) for i in range(len(x))]
    return x


def quit_columns(rosette, args, columns):
    """
    Filters out specified columns from the dataset based on user-defined arguments.

    Parameters:
        rosette (list): Data loaded from a CSV, expected to be a list of lists.
        args (argparse.Namespace): Parsed command-line arguments that may specify columns to exclude.
        columns (dict): Mapping of column names to their indices in the data rows.

    Returns:
        tuple: Contains processed feature vectors (x), target variable vector (y), and indices of used columns (index_i).
    """
    y = [mass for [_, mass, _, _, _, _, _, _] in rosette]
    index_i = [index for name, index in columns.items()]
    if args.change_data:
        x = [[row[i] for i in sorted(index_i[:-1])] for row in rosette]
        if args.add_rand_col:
            x = add_rand_column(x)
    else:
        x = [[flux_g, flux_r, flux_z, flux_w1, flux_w2, z] for [_, _, flux_g, flux_r, flux_z, flux_w1, flux_w2, z] in rosette]
    return x, y, index_i


def train(path, train_rosettes, model, index_i, test_size, random):
    """
    Trains a Random Forest regressor model using data from specified rosettes.

    Parameters:
        path (str): Base directory path where data files are stored.
        train_rosettes (list of int): Identifiers for rosettes to train on.
        model (RandomForestModel): The model instance to train.
        index_i (list of int): Indices of features to be used from the data.
        test_size (float): Proportion of data to hold out as a test set.
        random (bool): Flag to determine if a random column should be added.

    Description:
        Iteratively processes each specified rosette, loads data, modifies it if necessary, and performs model training.
    """
    for r in train_rosettes:
        rosette_path = os.path.join(path, f'rosette{r}.csv')
        rosette = model.read_csv(rosette_path)
        y = [n[1] for n in rosette]
        if random:
            x = [[row[i] for i in sorted(index_i[:-1])] for row in rosette]
            x = add_rand_column(x)
        else:
            x = [[row[i] for i in sorted(index_i)] for row in rosette]
        model.train(r, x, y, test_size)


def plot(model, x, y, columns, quit_columns, args, r):
    """
    Handles the plotting of predictions and SHAP values based on the trained model and user specifications.

    Parameters:
        model (RandomForestModel): The trained model to use for predictions and SHAP analysis.
        x (list): Feature data used for making predictions.
        y (list): Actual target values used for comparison.
        columns (dict): Dictionary mapping of data columns.
        quit_columns (list): List of columns that have been removed from the data.
        args (argparse.Namespace): Command-line arguments to determine which plots to generate.
        r (int): Roosette number.

    Description:
        Generates and saves prediction plots and SHAP value plots as specified by the user.
    """
    def prepare_plot_path(suffix, drop_info=''):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'random_forest'))
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
        color_values = 1 - (dif - np.min(dif)) / (np.max(dif) - np.min(dif))

        fig, ax = plt.subplots(figsize=(8,7))
        cmap = plt.get_cmap('viridis')
        scatter = ax.scatter(y, pred, c=color_values, cmap=cmap, s=5.5)
        ax.plot(y, fit, color='black', ls='--', linewidth=1)
        ax.set_ylabel('Prediction')
        ax.set_xlabel('Data')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax, label='Proximity to Fit', pad=0.07)
        plt.grid(True)

        drop_info = 'Excluding ' + ', '.join(quit_columns) if quit_columns else ''
        plt.title(f'Rosette {r} - $\\log M_{{*}}$ (Random Forest)\n{drop_info}', y=1.03)
        plot_path = prepare_plot_path('pred', drop_info)
        plt.savefig(plot_path)
        plt.close(fig)

    def plot_shap_values(x, y, columns, quit_columns, model, args, r):
        _, x_test, _, _ = train_test_split(x, y, test_size=args.test_size)
        new_x_test = pd.DataFrame(x_test, columns=[col for col in columns if col not in quit_columns])
        explainer = shap.Explainer(model.model, new_x_test)
        shap_values = explainer(new_x_test)

        fig, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, color=cm.viridis, plot_size=(9,5), show=False)
        fig.tight_layout(pad=3)

        drop_info = 'Excluding ' + ', '.join(quit_columns) if quit_columns else ''
        plt.title(f'Rosette {r} - SHAP Values\n{drop_info}')
        plot_path = prepare_plot_path('shap', drop_info)
        plt.savefig(plot_path)
        plt.close(fig)

    if args.plot:
        plot_predictions(x, y, quit_columns, model, args, r)

    if args.shap:
        plot_shap_values(x, y, columns, quit_columns, model, args, r)


def test(path, test_rosettes, model, index_i, columns, quit_col, args):
    """
    Evaluates a trained RandomForestModel using data from specified test rosettes.

    Parameters:
        path (str): Directory path where data files are stored.
        test_rosettes (list of int): Identifiers for rosettes to be used for testing.
        model (RandomForestModel): The model to use for evaluation.
        index_i (list of int): Indices of the features included in the model.
        columns (dict): Mapping of column names to indices.
        quit_col (list): Columns excluded from the model training.
        args (argparse.Namespace): Command-line arguments specifying additional functionalities like plotting.

    Description:
        Processes test data, applies the model to generate predictions, evaluates these predictions, and optionally generates plots.
    """
    for r in test_rosettes:
        rosette_path = os.path.join(path, f'rosette{r}.csv')
        rosette = model.read_csv(rosette_path)
        y = [n[1] for n in rosette]
        if args.add_rand_col:
            x = [[row[i] for i in sorted(index_i[:-1])] for row in rosette]
            x = add_rand_column(x)
        else:
            x = [[row[i] for i in sorted(index_i)] for row in rosette]
        metrics = model.score(x, y)
        print(f'Metrics for rosette {r}: {metrics}')
        if args.plot or args.shap:
            plot(model, x, y, columns, quit_col, args, r)


def main():
    args = configure_args()
    train_rosettes = [int(r) for r in args.train_rosettes.split(',')]
    test_rosettes = [int(r) for r in args.test_rosettes.split(',') if r]
    quit_col = [str(col) for col in args.change_data.split(',') if col]
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    columns = {'flux_r':2,'flux_g':3,'flux_z':4,'flux_w1':5,'flux_w2':6,'z':7}

    x, y, index_i = quit_columns(read_csv(os.path.join(path, f'rosette{train_rosettes[0]}.csv')), args, columns)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=args.test_size, random_state=42)
    optimizer = ModelOptimization(x_train, y_train, x_test, y_test)
    logger = setup_logging()

    if quit_col:
        index_i = [index for name, index in columns.items() if name not in quit_col]
    if args.add_rand_col:
        columns['Random'] = 8
        index_i.append(index_i[-1]+1)

    if args.optimize:
        optimizer.create_study(n_trials=10)
        model = RandomForestModel(optimizer.best_params, logger)
        model.logger.info({'quit':args.change_data,'best_param':optimizer.best_params, 'best_values':optimizer.best_value})
        print(f'Optuna best param:{optimizer.best_params}, best values:{optimizer.best_value}')
    else:
        model = RandomForestModel(optimizer.best_params, logger)

    train(path, train_rosettes, model, index_i, args.test_size, args.add_rand_col)
    if test_rosettes:
        test(path, test_rosettes, model, index_i, columns, quit_col, args)

    if args.feat_imp:
        feature_names = [col for col in columns if col not in quit_col]
        now = datetime.now()
        fl = now.strftime('%Y-%m-%d_%H-%M-%S')
        file_name = f'features_{fl}.png'
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'random_forest'))
        plot_path = os.path.join(base_dir, file_name)
        model.plot_feature_importances(feature_names, plot_path)


if __name__ == '__main__':
    main()