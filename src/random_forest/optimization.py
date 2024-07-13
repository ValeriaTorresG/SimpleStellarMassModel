from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelOptimization:
    def __init__(self, x_train, y_train, x_test, y_test):
        """
        Initializes the ModelOptimization class with training and testing data.

        Parameters:
            x_train (array-like): Training feature dataset.
            y_train (array-like): Training target dataset.
            x_test (array-like): Testing feature dataset.
            y_test (array-like): Testing target dataset.

        Description:
            - Stores the training and testing data as instance variables.
            - Sets initial sklearn default parameters for tracking the optimization process.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.best_params = {'criterion': 'squared_error',
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2}
        self.best_value = 0.0

    def objective(self, trial):
        """
        Objective function for hyperparameter optimization using Optuna.

        Parameters:
            trial (optuna.trial.Trial): A trial object which suggests hyperparameters.

        Returns:
            float: The R-squared score of the model evaluated using the suggested parameters.

        Description:
            - This method is called by Optuna to evaluate a set of parameters.
            - The goal is to maximize the R-squared score, which is returned to Optuna.
        """
        criterion = trial.suggest_categorical('criterion', ['squared_error', 'absolute_error', 'friedman_mse', 'poisson'])
        n_estimators = trial.suggest_int('n_estimators', 5, 100)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

        regressor = RandomForestRegressor(
            criterion=criterion, n_estimators=n_estimators,
            max_depth=max_depth, min_samples_split=min_samples_split,
            random_state=0
        )
        regressor.fit(self.x_train, self.y_train)
        return r2_score(self.y_test, regressor.predict(self.x_test))

    def create_study(self, n_trials):
        """
        Creates and conducts a Optuna study to find the best hyperparameters for the RandomForestRegressor.

        Parameters:
            n_trials (int): The number of trials Optuna should run to optimize the parameters.

        Description:
            - Initializes an Optuna study aimed at maximizing the objective function.
            - Runs the study for a defined number of trials.
            - Stores the best parameters and best objective value found during the study in instance variables.
            - Reduces verbosity of Optuna's logging to focus on warnings only.
        """
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_params
        self.best_value = study.best_value
