from model import LinearRegressionModel
from datetime import datetime
import numpy as np
import argparse
import logging
import os

import matplotlib
import seaborn as sns
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
    parser.add_argument('--train_rosettes', type=str, required=True,
                        help='Comma-separated list of rosettes IDs for training (e.g., 1,2,3)')
    parser.add_argument('--test_rosettes', type=str, required=False, default='',
                        help='Comma-separated list of rosettes IDs for testing (e.g., 4,5)')
    parser.add_argument('--test_size', type=float, required=False, default=0.0,
                        help='Proportion of the dataset to include in the test split (0.0 means train on all).')
    parser.add_argument('--plot', action='store_true', required=False, default=False,
                        help='Enable plotting of results')
    return parser.parse_args()


def setup_logging():
    """
    Sets up a centralized logging system for tracking the application's operations.

    Returns:
        logging.Logger: Configured logger object for recording operations.
    """
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'job_management', 'logs'))
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(filename=os.path.join(log_dir, 'linear.log'),
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
    Loads data from each rosette in rosette_paths, concatenates them,
    and then trains the linear regression model (with optional train/test split).

    Parameters:
        model (LinearRegressionModel): The model to train.
        rosette_paths (list of str]): Paths to CSV files for training rosettes.
        test_size (float): Proportion of the dataset to reserve for internal test.

    Description:
        - Concatenates data from all provided rosettes into a single dataset.
        - If test_size > 0, it splits that combined dataset internally for a sanity-check test set.
        - Otherwise, it trains on the entire combined dataset.
    """
    all_data = []
    rosette_ids = []
    for rosette_path in rosette_paths:
        rosette = model.read_csv(rosette_path)
        all_data.extend(rosette)
        rosette_id = int(os.path.splitext(os.path.basename(rosette_path))[0].replace('rosette', ''))
        rosette_ids.append(rosette_id)

    model.train(rosette_ids, all_data, test_size)


def plot_results(rosette_id, masses, pred):
    """
    Plots predictions from the model against actual target values.

    Parameters:
        rosette_id (int): Identifier for the current rosette being processed.
        masses (list): Actual target values from the rosette data.
        pred (list): Predicted values from the model.
    """
    slope, intercept = np.polyfit(masses, pred, 1)
    fit = slope * np.array(masses) + intercept
    dif = np.abs(pred - fit)
    norm = (dif - np.min(dif)) / (np.max(dif) - np.min(dif) + 1e-12)
    color_values = 1 - norm

    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = sns.color_palette("mako", as_cmap=True)
    scatter = ax.scatter(masses, pred, c=color_values, cmap=cmap, s=5.5)
    ax.plot(masses, fit, color='black', ls='--', linewidth=1)
    ax.set_ylabel('Prediction')
    ax.set_xlabel('Data')
    plt.title(f'Rosette {rosette_id} - ' + r'$\log M_{*}$ (Linear)', y=1.03)
    plt.colorbar(scatter, ax=ax, label='Proximity to Fit', pad=0.07)
    plt.grid(True)

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'plots', 'linear'))
    os.makedirs(base_dir, exist_ok=True)
    now = datetime.now()
    fl = now.strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f'pred_r{rosette_id}_{fl}.png'
    plt.savefig(os.path.join(base_dir, file_name), dpi=360)
    plt.close()


def test_model(model, rosette_paths, plot=False):
    """
    Evaluates the trained model on each specified test rosette and logs/prints the results.

    Parameters:
        model (LinearRegressionModel): The already-trained model.
        rosette_paths (list of str): Paths to the CSV files containing test rosette data.
        plot (bool): If True, create a scatter-plot of predicted vs. true values.
    """
    for rosette_path in rosette_paths:
        rosette_id = int(os.path.splitext(os.path.basename(rosette_path))[0].replace('rosette', ''))
        rosette = model.read_csv(rosette_path)
        fluxes = [n[2:] for n in rosette]
        masses = [n[1] for n in rosette]

        metrics, pred = model.score(rosette_id, fluxes, masses)
        print(f"Test on Rosette {rosette_id}: R^2={metrics['r2']:.3f}, "
              f"MSE={metrics['mse']:.3f}, Pearson={metrics['pc']:.3f}")

        if plot:
            plot_results(rosette_id, np.array(masses), pred)


def main():
    args = parse_arguments()
    logger = setup_logging()
    model = LinearRegressionModel(logger)

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

    train_rosettes = [int(r) for r in args.train_rosettes.split(',')]
    train_rosettes_paths = get_rosette_paths(base_path, train_rosettes)

    test_rosettes = [int(r) for r in args.test_rosettes.split(',')] if args.test_rosettes else []
    test_rosettes_paths = get_rosette_paths(base_path, test_rosettes)

    train_model(model, train_rosettes_paths, args.test_size)

    if test_rosettes_paths:
        test_model(model, test_rosettes_paths, plot=args.plot)


if __name__ == "__main__":
    main()