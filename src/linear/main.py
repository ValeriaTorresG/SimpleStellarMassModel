#!/usr/bin/env python
from model import LinearRegressionModel
from datetime import datetime
import numpy as np
import argparse
import logging
import os

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 360
matplotlib.rcParams['text.usetex'] = True


def parse_arguments():
    """
    Parses command line arguments for the model training and evaluation script.

    Returns:
        argparse.Namespace: Contains all the command-line arguments provided.
    """
    parser = argparse.ArgumentParser(description='Model training and evaluation on rosettes data')
    parser.add_argument('--train_rosettes', type=str, required=True, help='Comma-separated list of rosettes IDs for training')
    parser.add_argument('--test_rosettes', type=str, required=False, default='', help='Comma-separated list of rosettes IDs for testing, if any')
    parser.add_argument('--test_size', type=float, required=False, default=0.8, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--plot', action='store_true', required=False, default=False, help='Enable plotting of results')
    return parser.parse_args()


def setup_logging():
    """
    Sets up a centralized logging system for tracking the application's operations.

    Returns:
        logging.Logger: Configured logger object for recording operations.

    Description:
        - Creates a log directory if it doesn't exist.
        - Configures logging to file with a specific format and date/time stamping.
    """
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'job_management', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, 'linear.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    return logging.getLogger('linear')


def get_rosette_paths(base_path, rosettes):
    """
    Constructs file paths for rosette data files based on their identifiers.

    Parameters:
        base_path (str): The base directory where rosette data files are stored.
        rosettes (list of int): List of rosette identifiers.

    Returns:
        list of str: Full file paths to the rosette data files.
    """
    return [os.path.join(base_path, f'rosette{r}.csv') for r in rosettes]


def train_model(model, rosette_paths, test_size):
    """
    Trains the provided LinearRegressionModel using data from specified rosette files.

    Parameters:
        model (LinearRegressionModel): The model to train.
        rosette_paths (list of str): Paths to the CSV files containing rosette data.
        test_size (float): Proportion of the dataset to reserve for testing.

    Description:
        - Reads data from each rosette file.
        - Calls the train method on the model for each dataset.
    """
    for rosette_path in rosette_paths:
        rosette = model.read_csv(rosette_path)
        rosette_id = int(os.path.splitext(os.path.basename(rosette_path))[0].replace('rosette', ''))
        model.train(rosette_id, rosette, test_size)


def plot_results(rosette_id, masses, pred, train_rosettes, test_size):
    """
    Plots predictions from the model against actual target values.

    Parameters:
        rosette_id (int): Identifier for the current rosette being processed.
        masses (list): Actual target values from the rosette data.
        pred (list): Predicted values from the model.
        train_rosettes (str): Identifiers for training rosettes, used in the plot title.
        test_size (float): Proportion of the test set, used in the plot title.

    Description:
        - Creates a scatter plot with a color-coded fit line indicating prediction accuracy.
        - Saves the plot to a specified directory with a timestamped filename.
    """
    slope, intercept = np.polyfit(masses, pred, 1)
    fit = slope*np.array(masses)+intercept
    dif = np.abs(pred-fit)
    norm = (dif-np.min(dif))/(np.max(dif)-np.min(dif))
    color_values = 1-norm

    fig, ax = plt.subplots(figsize=(8,7))
    cmap = plt.get_cmap('viridis')
    scatter = ax.scatter(masses, pred, c=color_values, cmap=cmap, s=5.5)
    ax.plot(masses, fit, color='black', ls='--', linewidth=1)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Data')
    ax.set_aspect('equal')
    plt.title(f'Rosette {rosette_id} - ' + r'$\log M_{*}$ (Linear)', y=1.03)
    plt.colorbar(scatter, ax=ax, label='Proximity to Fit', pad=0.07)
    plt.grid(True)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'linear'))
    now = datetime.now()
    print(base_dir)
    fl = now.strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'pred_{fl}.png'
    plt.savefig(os.path.join(base_dir, file_name))
    plt.close()


def test_model(model, rosette_paths, args, logger):
    """
    Evaluates the trained model using specified test rosette files and logs the results.

    Parameters:
        model (LinearRegressionModel): The model to evaluate.
        rosette_paths (list of str): Paths to the CSV files containing test rosette data.
        args (argparse.Namespace): Command-line arguments, used to determine plotting requirements.
        logger (logging.Logger): Logger for recording test outcomes.

    Description:
        - For each rosette file, predicts target variables and calculates metrics.
        - Optionally plots the results if specified.
    """
    for rosette_path in rosette_paths:
        rosette = model.read_csv(rosette_path)
        rosette_id = int(os.path.splitext(os.path.basename(rosette_path))[0].replace('rosette', ''))
        fluxes, masses = [n[2:] for n in rosette], [n[1] for n in rosette]
        metrics, pred = model.score(rosette_id, fluxes, masses)
        print(f'Metrics for rosette {rosette_id}: {metrics}')
        if args.plot:
            plot_results(rosette_id, masses, pred, args.train_rosettes, args.test_size)


def main():
    args = parse_arguments()
    logger = setup_logging()
    model = LinearRegressionModel(logger)
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    train_rosettes_paths = get_rosette_paths(base_path, [int(r) for r in args.train_rosettes.split(',')])
    test_rosettes_paths = get_rosette_paths(base_path, [int(r) for r in args.test_rosettes.split(',') if r])

    train_model(model, train_rosettes_paths, args.test_size)
    test_model(model, test_rosettes_paths, args, logger)

if __name__ == "__main__":
    main()