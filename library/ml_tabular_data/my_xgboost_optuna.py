from dataclasses import dataclass
from typing import List, Union, Tuple, Any
import numpy as np
import pandas as pd
from pandas import DataFrame
import optuna
from itertools import product
from src.ml_models.data_class import Datasets
from typing import List, Optional
from xgboost import XGBRegressor
import xgboost as xgb
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, \
    balanced_accuracy_score

from sklearn.model_selection import KFold, ParameterGrid
from tqdm import tqdm
import numpy.typing as npt
import pathlib
import seaborn as sns
from matplotlib import pyplot as plt
import operator
import json


# class LRScheduler(xgb.callback.TrainingCallback):
#     """Custom learning rate scheduler for dynamic learning rate adjustment."""
#
#     def __init__(self, initial_lr, decay_rate):
#         self.initial_lr = initial_lr
#         self.decay_rate = decay_rate
#
#     def before_iteration(self, model, epoch, evals_log):
#         """Update learning rate before each iteration."""
#         new_lr = self.initial_lr * (self.decay_rate ** epoch)
#         model.set_param('learning_rate', new_lr)
#         print(f'LR: {new_lr}')
#         return False  # Continue training


class LRScheduler(xgb.callback.TrainingCallback):
    """Custom callback to reduce the learning rate based on validation performance."""

    def __init__(self,
                 n_boosting: float,
                 monitor: str = 'eval-mlogloss',
                 delta: float = 0.001,
                 patience: int = 25,
                 decay_factor: float = 0.5,
                 min_lr: float = 1e-3,
                 look_back=10):
        """
        Parameters:
            monitor (str): Name of the evaluation metric to monitor.
            delta (float): Minimum relative change in the monitored metric to qualify as an improvement.
            patience (int): Number of rounds with no significant improvement after which learning rate will be reduced.
            decay_factor (float): Factor by which the learning rate will be reduced.
            min_lr (float): Minimum learning rate below which the learning rate will not be reduced.
            look_back (int): Number of past observations to consider for trend analysis.
        """
        self.monitor = monitor
        self.delta = delta
        self.patience = patience
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.look_back = look_back
        self.stagnant_rounds = 0
        self.validation_scores = [[]] * n_boosting

    def before_training(self, model):
        """Called before training starts."""
        self.initial_lr = self._get_learning_rate(model=model)
        return model

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration."""
        current_score = evals_log['valid'][self.monitor][-1]
        self.validation_scores[epoch] = current_score

        # Only start checking once we have enough data
        if epoch >= self.look_back and self.initial_lr > self.min_lr:
            recent_scores = [x for x in self.validation_scores if x != []]
            recent_scores = recent_scores[-self.look_back:]
            # Check if improvement is smaller than delta over the last few epochs
            print(f'{epoch}: {np.abs(max(recent_scores) - min(recent_scores))} | {self.delta}')
            if np.abs(max(recent_scores) - min(recent_scores)) < self.delta:
                self.stagnant_rounds += 1
            else:
                self.stagnant_rounds = 0  # Reset counter if recent trend shows sufficient variability

            # Reduce learning rate if no significant changes for 'patience' rounds
            if self.stagnant_rounds >= self.patience:
                current_lr = self.initial_lr
                # new_lr = max(current_lr * self.decay_factor, self.min_lr)
                new_lr = (self.decay_factor**epoch)*current_lr
                if new_lr < current_lr:
                    print(f'Updating LR :{current_lr} to {new_lr}')
                    model.set_param('learning_rate', new_lr)
                    print(f"Reducing learning rate to {new_lr} at epoch {epoch}")
                    self.stagnant_rounds = 0  # Reset after adjustment

        return False  # Continue training

    def after_training(self, model):
        """Called after training ends."""
        return model

    def _get_learning_rate(self, model) -> float:
        model_params = json.loads(model.save_config())
        learning_rate = float(
            model_params.get('learner').get('gradient_booster').get('tree_train_param').get('learning_rate'))
        return learning_rate


class XGBoostModel:
    def __init__(self,
                 n_boosting: int = 1000,
                 ds: Optional[Datasets] = None,
                 learning_task: str = 'regression',
                 num_classes: Optional[int] = None,
                 path_model_save: pathlib.Path = None,
                 path_model_results: pathlib.Path = None):
        self.seed = 42

        self.learning_task = learning_task
        learning_tasks = {
            'regression': {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
            },
            'classification': {
                'objective': 'binary:logistic',  # For binary classification
                'eval_metric': 'error',  # Binary classification error rate
            },

            'classification_multiclass': {
                'objective': 'multi:softmax',  # Use softmax for multiclass classification
                'eval_metric': 'mlogloss',  # Multiclass classification error rate
            }
        }

        if not learning_task in learning_tasks.keys():
            raise ValueError(f'The learning task must be one of the following \n\t\t{learning_tasks.keys()}')

        if 'classification' in self.learning_task:
            num_classes = num_classes
        else:
            num_classes = None

        self.params = learning_tasks[learning_task]
        self.params.update({'device': 'cuda',
                            'num_class': num_classes,
                            'learning_rate': 0.01})

        self.n_boosting = n_boosting  # num_boost_round

        if not self.learning_task == 'regression':
            self.params['num_class'] = ds.train_y.nunique()

        self.xgb_model = None
        if ds:
            self.dtrain = xgb.DMatrix(data=ds.train_X,
                                      label=ds.train_y)
            self.deval = xgb.DMatrix(data=ds.valid_X,
                                     label=ds.valid_y)
            self.dtest = xgb.DMatrix(ds.test_X,
                                     label=ds.test_y)
        else:
            self.dtrain = None
            self.dval = None
            self.dtest = None
        self.path_model_save = path_model_save
        self.path_model_results = path_model_results
        self.predictions = {}

    def train_and_eval_model(self,
                             ds: Datasets = None,
                             features_to_include: Optional[list[str]] = None
                             ):
        """
        Fits AdaBoostRegressor on a subset of features using specified hyperparameters

        :param ds: Datasets for training, validation and testing
        :param features_to_include: List of features to include in the model
        :return: predictions and evaluation metrics
        """
        if features_to_include is None:
            features_to_include = ds.train_X.columns.tolist()

        assert all([feature in ds.train_X.columns for feature in features_to_include])

        self.dtrain = xgb.DMatrix(data=ds.train_X[features_to_include],
                                  label=ds.train_y)

        self.deval = xgb.DMatrix(data=ds.valid_X[features_to_include],
                                 label=ds.valid_y)

        self.dtest = xgb.DMatrix(ds.test_X[features_to_include],
                                 label=ds.test_y)

        watchlist = [(self.dtrain, 'train'), (self.deval, 'valid')]
        eval_results = {}
        self.xgb_model = xgb.train(
            params=self.params,
            dtrain=self.dtrain,
            evals=watchlist,
            evals_result=eval_results,
            num_boost_round=self.n_boosting,
            verbose_eval=200,
        )

        self.plot_training_val_loss(results=eval_results,
                                    plot_path=self.path_model_results.joinpath('train_val_curve.png'))

        pred_dtrain = self.xgb_model.predict(data=self.dtrain,
                                             output_margin=False, )

        pred_dval = self.xgb_model.predict(data=self.deval,
                                           output_margin=False, )
        pred_dtest = self.xgb_model.predict(data=self.dtest,
                                            output_margin=False, )

        predictions = {"train": pred_dtrain,
                       "valid": pred_dval,
                       "test": pred_dtest}

        return predictions, self.evaluate_model(ds=ds, pred=predictions, output_dtype='frame')

    def train_and_eval_model_lr_scheduler(self,
                             ds: Datasets = None,
                             features_to_include: Optional[list[str]] = None
                             ):
        """
        Fits AdaBoostRegressor on a subset of features using specified hyperparameters

        :param ds: Datasets for training, validation and testing
        :param features_to_include: List of features to include in the model
        :return: predictions and evaluation metrics
        """
        if features_to_include is None:
            features_to_include = ds.train_X.columns.tolist()

        assert all([feature in ds.train_X.columns for feature in features_to_include])

        self.dtrain = xgb.DMatrix(data=ds.train_X[features_to_include],
                                  label=ds.train_y)

        self.deval = xgb.DMatrix(data=ds.valid_X[features_to_include],
                                 label=ds.valid_y)

        self.dtest = xgb.DMatrix(ds.test_X[features_to_include],
                                 label=ds.test_y)

        lr_scheduler = LRScheduler(monitor=self.params.get('eval_metric'),
                                   delta=0.001,
                                   patience=20,
                                   decay_factor=0.008,
                                   min_lr=1e-2,
                                   n_boosting=self.n_boosting)

        params = {
            # Add Optuna suggested parameters
            'max_depth': 4,
            'min_child_weight': 13,
            # 'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'gamma': 3.0186694836771587e-06,
            'subsample': 0.6,
            'colsample_bytree':0.6,
            'reg_alpha': 7,
            'reg_lambda': 2,
            **self.params,
        }

        watchlist = [(self.dtrain, 'train'), (self.deval, 'valid')]
        eval_results = {}
        self.xgb_model = xgb.train(
            params=params,
            dtrain=self.dtrain,
            evals=watchlist,
            evals_result=eval_results,
            num_boost_round=self.n_boosting,
            callbacks=[lr_scheduler],
            verbose_eval=200,
        )

        self.plot_training_val_loss(results=eval_results,
                                    plot_path=self.path_model_results.joinpath('train_val_curve.png'))

        pred_dtrain = self.xgb_model.predict(data=self.dtrain,
                                             output_margin=False, )

        pred_dval = self.xgb_model.predict(data=self.deval,
                                           output_margin=False, )
        pred_dtest = self.xgb_model.predict(data=self.dtest,
                                            output_margin=False, )

        self.predictions = {"train": pred_dtrain,
                       "valid": pred_dval,
                       "test": pred_dtest}

        return self.predictions, self.evaluate_model(ds=ds, pred=self.predictions, output_dtype='frame')

    @staticmethod
    def custom_ordinal_loss(preds, dtrain):
        """
        The custom_loss function calculates the loss based on the ordinal distance. The gradient and hessian are also
        computed for use in the XGBoost optimization process.
        :param preds:
        :param dtrain:
        :return:
        """
        labels = dtrain.get_label()
        preds = preds.reshape(-1, 4)

        # Calculate softmax probabilities
        exp_preds = np.exp(preds)
        preds_prob = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)

        # Predicted classes (as integers)
        pred_classes = np.argmax(preds_prob, axis=1)

        # Calculate the weighted squared error
        diff = pred_classes - labels
        loss = np.sum(diff ** 2) / len(labels)

        # Gradient and Hessian
        grad = np.zeros_like(preds)
        hess = np.zeros_like(preds)

        for i in range(len(labels)):
            for j in range(4):
                if j == labels[i]:
                    grad[i, j] = 2 * (j - pred_classes[i])
                    hess[i, j] = 2
                else:
                    grad[i, j] = 2 * (pred_classes[i] - j)
                    hess[i, j] = 2

        return grad.ravel(), hess.ravel()
    def _objective(self,
                   trial: optuna.Trial,
                   X: pd.DataFrame,
                   y: pd.Series,
                   kfold,
                   early_stopping_rounds: int,
                   lr_decay: float = 0.09,
                   patience: int = 10) -> float:

        # def create_lr_scheduler(initial_lr, lr_decay):
        #     """ Returns a callback that updates the learning rate dynamically. """
        #
        #     def callback(env):
        #         # Update learning rate by decay factor every round
        #         lr = initial_lr * (lr_decay ** env.iteration)
        #         env.model.set_param('learning_rate', lr)
        #
        #     return callback

        # Define parameters to optimize
        params = {
            # Add Optuna suggested parameters
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
            # 'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-10, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 10.0, log=True),
            **self.params,
        }

        # Create a learning rate scheduler
        lr_scheduler = LRScheduler(monitor=self.params.get('eval_metric'),
                                   delta=0.0001,
                                   patience=patience,
                                   decay_factor=lr_decay,
                                   min_lr=1e-5,
                                   n_boosting=self.n_boosting)

        # Perform cross-validation
        maes = []
        for train_index, valid_index in kfold.split(X, y):
            dtrain = xgb.DMatrix(X.iloc[train_index], label=y.iloc[train_index])
            dvalid = xgb.DMatrix(X.iloc[valid_index], label=y.iloc[valid_index])
            watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
            model = xgb.train(params,
                              dtrain,
                              self.n_boosting,
                              evals=watchlist,
                              evals_result={},
                              early_stopping_rounds=early_stopping_rounds,
                              callbacks=[lr_scheduler],
                              verbose_eval=False)
            valid_pred = model.predict(dvalid)
            valid_mae = mean_absolute_error(y.iloc[valid_index], valid_pred)
            maes.append(valid_mae)
        mean_mae = np.mean(maes)
        return mean_mae

    def run_hparam_search_optuna(self,
                                 ds: Any,
                                 n_splits: int = 10,
                                 early_stopping_rounds: Optional[int] = 100,
                                 n_trials: int = 10,
                                 lr_decay: float = 0.09) -> Tuple[Any, pd.DataFrame]:
        """
        Run hyperparameter search using Optuna to find the best combination of hyperparameters for the XGBoost model.

        Args:
        - ds (Any): Dataset object containing training data.
        - n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        - n_trials (int, optional): Set total number of trials for Optuna
        - early_stopping_rounds (int, optional): Number of rounds for early stopping during model training.
                                                 Defaults to 100.

        Returns:
        - Tuple[Any, pd.DataFrame]: Tuple containing the best hyperparameters found and a DataFrame with the
                                    optimization results.
        """
        X = ds.train_X
        y = ds.train_y
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        # Create a tqdm progress bar
        with tqdm(total=n_trials, desc="Optimizing", unit="trial") as progress_bar:
            # Define objective function
            def objective(trial):
                # Inside the objective function, you can use tqdm to track the progress
                progress_bar.update(1)  # Update tqdm progress bar
                return self._objective(trial, X, y, kfold, early_stopping_rounds, lr_decay)

            # Create an Optuna study
            study = optuna.create_study(direction='minimize')
            # Optimize hyperparameters
            study.optimize(objective, n_trials=n_trials)

        # Extract best parameters
        best_params = study.best_params
        print(f'Best params: {best_params}')

        # Update model with best parameters
        self._update_params(best_params=best_params)

        # Optionally, you can also evaluate the model with the best parameters on the entire dataset here

        # Create DataFrame of results (optional)
        cv_results_df = pd.DataFrame.from_dict(study.trials_dataframe())

        return best_params, cv_results_df

    def _update_params(self, best_params: dict):
        """Update the params in the class"""
        self.params.update(best_params)
        print(f'Initial parameters have been updated to')
        for key, val in self.params.items(): print(f'{key}: {val} \n')

    def feature_selection_cv_sklearn(self,
                                     ds: Datasets,
                                     feature_imp_score: str = 'gain',
                                     n_splits: Optional[int] = 10) -> pd.DataFrame:
        """
        Perform cross-validation and extract the most important features on each fold alongside with the metrics of the
        fold.
        :param ds: Dataset class
        :return:
            dataframe, each row is a cv result, the columns have the features and the metrics. The cells
            on the features of the columns have the feature importance scores.
        """

        def evaluate_cv(y_pred: npt.NDArray, y_true: npt.NDArray) -> dict:
            return {
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
            }

        imp_score_options = {'weight', 'gain', 'cover', 'total_gain', 'total_cover'}

        if any(set(imp_score_options).intersection(feature_imp_score)):
            raise ValueError(f'Invalid importance score {feature_imp_score}. Valid options are :{imp_score_options}')

        # Create a KFold object
        kf = KFold(n_splits=n_splits)

        # Hold feature importance
        feature_importance = []
        # Loop through each fold
        for train_index, test_index in kf.split(ds.train_X, ds.train_y):
            X_train, X_test = ds.train_X.iloc[train_index], ds.train_X.iloc[test_index]
            y_train, y_test = ds.train_y.iloc[train_index], ds.train_y.iloc[test_index]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Train the model
            model = xgb.train(self.params, dtrain, )

            # Feature importance
            f_importance = model.get_score(importance_type=feature_imp_score)

            # metrics of the current training
            pred_dtest = model.predict(data=dtest,
                                       output_margin=False, )

            metrics = evaluate_cv(y_pred=y_test, y_true=pred_dtest)

            # compile features scores and metrics
            f_importance.update(metrics)
            feature_importance.append(f_importance)
        # Combine feature importances for all folds
        df_fi = pd.DataFrame(feature_importance)
        df_fi['score'] = feature_imp_score
        return df_fi

    def evaluate_model(self, ds: Datasets, pred: dict, output_dtype: Optional[str] = 'dict') -> Union[
        dict, pd.DataFrame]:
        metrics = {}
        if not any(set(pred.keys()).intersection({'train', 'test', 'val'})):
            raise ValueError('The dictionary must contain at least one of the following keys: ["train", "test", "val"]')

        if 'train' in pred:
            if self.learning_task == 'regression':
                metrics["train"] = {
                    "RMSE": np.sqrt(mean_squared_error(ds.train_y, pred['train'])),
                    "MAE": mean_absolute_error(ds.train_y, pred['train']),
                    "R2": r2_score(ds.train_y, pred['train']),
                }

            else:
                metrics["train"] = {
                    "Accuracy": accuracy_score(ds.train_y, pred['train']),
                    "F1 Score": f1_score(ds.train_y, pred['train'], average='weighted'),
                    "Balanced Accuracy": balanced_accuracy_score(ds.train_y, pred['train'])
                }

        if 'valid' in pred:
            if self.learning_task == 'regression':
                metrics["valid"] = {
                    "RMSE": np.sqrt(mean_squared_error(ds.valid_y, pred['valid'])),
                    "MAE": mean_absolute_error(ds.valid_y, pred['valid']),
                    "R2": r2_score(ds.valid_y, pred['valid']),
                }
            else:
                metrics["valid"] = {
                    "Accuracy": accuracy_score(ds.valid_y, pred['valid']),
                    "F1 Score": f1_score(ds.valid_y, pred['valid'], average='weighted'),
                    "Balanced Accuracy": balanced_accuracy_score(ds.valid_y, pred['valid'])
                }

        if 'test' in pred:
            if self.learning_task == 'regression':
                metrics["test"] = {
                    "RMSE": np.sqrt(mean_squared_error(ds.test_y, pred['test'])),
                    "MAE": mean_absolute_error(ds.test_y, pred['test']),
                    "R2": r2_score(ds.test_y, pred['test'])}
            else:
                metrics["test"] = {
                    "Accuracy": accuracy_score(ds.test_y, pred['test']),
                    "F1 Score": f1_score(ds.test_y, pred['test'], average='weighted'),
                    "Balanced Accuracy": balanced_accuracy_score(ds.test_y, pred['test'])
                }

        if output_dtype == 'dict':
            return metrics
        else:
            metrics_df = pd.DataFrame(metrics).T  # Convert dict to DataFrame and transpose
            return metrics_df

    def plot_training_val_loss(self, results: dict,
                               plot_path: Optional[pathlib.Path] = None):
        """Result from the fit src. Plot the trainning and validation loss"""

        plt.figure(figsize=(10, 7))
        eval_metric = results
        metric = [*eval_metric.get('train').keys()][0]
        if ('valid' in results.keys()) and ("train" in results.keys()):
            # validation and training loss
            plt.plot(eval_metric.get('train').get(metric),
                     label="Training loss",
                     linewidth=2.5)
            plt.plot(eval_metric.get('valid').get(metric),
                     label="Validation loss",
                     linewidth=2.5)
            if hasattr(self, 'es_round'):
                if self.es_round == self.n_boosting:
                    # no early stop on validation, identify the best src
                    early_stop_point = {'x': np.max(results["validation_1"][eval_metric]),
                                        'y': np.argmax(results["validation_1"][eval_metric])}
                else:
                    early_stop_point = {'x': np.max(results["validation_1"][eval_metric]),
                                        'y': len(results["validation_1"][eval_metric]) - self.es_round}
                plt.plot(early_stop_point['y'], early_stop_point['x'], marker="o",
                         markersize=7, markeredgecolor="green", markerfacecolor="green")
        elif "train" in results.keys():
            # only training loss
            plt.plot(eval_metric.get('train').get(metric),
                     label="Training loss",
                     linewidth=2.5)
        else:
            print(f'Expected keys for curves not defined, unable to plot loss curves')
            return eval_metric

        # plt.axvline(21, color="gray", label="Optimal tree number")
        plt.title(f"Learning Curve XGB",
                  fontsize=24,
                  y=1.05)
        plt.xlabel("Learning Iterations",
                   fontsize=24)
        plt.ylabel(f"Loss {metric.capitalize()}",
                   fontsize=24)
        plt.tick_params(axis='y',
                        labelsize=20)
        plt.tick_params(axis='x',
                        labelsize=20)
        plt.grid(alpha=.7)
        plt.legend(prop={'size': 20})
        fig = plt.gcf()
        plt.draw()
        if plot_path:
            fig.savefig(plot_path, dpi=300)
        plt.show()

    def get_model(self):
        """return the xgboost model from the training method"""
        return self.xgb_model

    @staticmethod
    def plot_features_separate_figures(xgb_model: xgb,
                                       feature_names: Optional[list] = None,
                                       plot_path: pathlib.Path = None,
                                       display: Optional[bool] = False,
                                       my_palette: Optional[str] = 'dark:salmon_r',
                                       fig_title: Optional[int] = 24, big_title: Optional[int] = 26,
                                       legend_size: Optional[int] = 24,
                                       figsize: Optional[tuple] = (8, 24),
                                       dpi: Optional[int] = 300,
                                       render_plot: bool = False, ) -> dict:
        """
            Once an XGBoost src has been trained we can utilize XGBoost feature selection to select the best
            features.Although, XGBoost provides 4 different measures for the feature selection:
                \item weight: the number of times a feature is used to split the data across all trees.
                 \item gain: the average gain across all splits the feature is used in.
                 \item cover: the average coverage across all splits the feature is used in.
                 \item total gain: the total gain across all splits the feature is used in.
                 \item total cover: the total coverage across all splits the feature is used in.
             To analyze the selection, this function will plot the scores for each metrics and the corresponding
             predictor. The nice thing of the plot, is that we do not need to use the x-axis on each plot for the
             features but we use a unique color-encoded legend for all the features. If the predictor is zero
             it means the score was not significat but to have the single color-legend all plots must show a
             space for the predictor.

             The plot is constructred from a pd.Dtaframem, where the column Type assigns the unique color to each
             question. The amount of colors is Dynamic, it depends on the length of most significant question

            TO_DO: Add save path

            :param xgb_model: xgboost from xgboost library, not sklearn
            :param my_palette:
            :param fig_title:
            :param big_title:
            :param legend_size:
            :param figsize:
            :param dpi:
            :param render_plot: render the plot or not, for many feature the process is cpu intensive, not recommended
            to render
            :return: Nested dictionary, outside keys are the metrics, inner keys the features,and values the measure
        """
        sns.set_context("paper")
        sns.set_theme('paper')
        sns.set_style("whitegrid")

        def get_keys(dictionary: dict, only_inner: Optional[bool] = False):
            """Get nested keys from a nested dictionary"""
            result = []
            for key, value in dictionary.items():
                if type(value) is dict:
                    new_keys = get_keys(value)
                    if not only_inner:
                        result.append(key)
                    for innerkey in new_keys:
                        result.append(innerkey)
                else:
                    result.append(key)
            return result

        def detect_elbow(data):
            """
            Ordered series get the point of inflection or elbow.
            Usage:

                data = [1, 2, 3, 4, 3, 2, 1]
                elbow_index = detect_elbow(df[ticker])
                print('Elbow at index:', elbow_index)

            :param data:
            :return:
            """
            from scipy.spatial.distance import cdist
            # Coordinates of the first and last points
            first_point = [0, data[0]]
            last_point = [len(data) - 1, data[-1]]

            # Array of coordinates for all points
            coordinates = [[i, v] for i, v in enumerate(data)]

            # Array of distances from each point to the line from first to last
            distances = cdist([first_point, last_point], coordinates, 'euclidean').sum(axis=0)

            # Return the index of the point with max distance
            return np.argmax(distances)

        # define the different metrics, we will have one subplot per metric
        importance_types = {'weight': {}, 'gain': {}, 'cover': {}, 'total_gain': {}, 'total_cover': {}}
        # from the src, retrive the metrics with its given predictor
        for imp_ in importance_types.keys():
            imp_importance = xgb_model.get_score(importance_type=imp_, fmap='')
            # sort by value and assing in the list
            importance_types[imp_] = dict(sorted(imp_importance.items(), key=operator.itemgetter(1), reverse=True))

        if render_plot:
            all_questions = get_keys(importance_types, only_inner=True)
            set_questions = list(set(all_questions))
            # generate as many colors as are the unique features in the selection
            palette = sns.color_palette(my_palette, len(set_questions))
            palette_dict = {f'Type_{i}': palette[i] for i in range(0, len(palette))}
            # create the dataframe grid
            df = pd.DataFrame(0, columns=[*importance_types.keys()], index=set_questions)
            # assign an x value to each predictor
            df['x_value'] = [y + 1 for y in range(0, len([*palette_dict.keys()]))]
            # set a columns for a unique color assinged to a unique row (predictor)
            df['type_color'] = [*palette_dict.keys()]
            # make dict with the featu and the colors labels
            colors_pred_dict = df['type_color'].to_dict()
            colors_pred_dict = {val: key for key, val in colors_pred_dict.items()}
            # fill the grid
            for pred_ in set_questions:
                for measure_ in importance_types.keys():
                    df.loc[pred_, measure_] = importance_types[measure_][pred_]

            for ticker in importance_types.keys():
                # ticker = 'gain'
                df.sort_values(by=ticker,
                               ascending=False,
                               inplace=True)
                df['x_value'] = range(0, df.shape[0])
                elbow_index = detect_elbow(df[ticker])
                elbow_index_x_axis = df[ticker][elbow_index]
                plt.figure(figsize=figsize, dpi=dpi)
                # plt.suptitle("XGBoost Feature Selection Metrics", fontsize=big_title, y=.92)
                ax = plt.subplot(1, 1, 1)
                # filter df and plot ticker on the new subplot axis
                sns.barplot(data=df,
                            x=ticker,
                            y="x_value",
                            hue=ticker,
                            palette=my_palette,  # palette_dict,
                            dodge=False,
                            orient='h',
                            legend=False,
                            ax=ax)

                # twick the design of the plots
                ax.set_title(ticker.capitalize(),
                             fontsize=fig_title)
                ax.set_xlabel("")
                ax.set_ylabel("")
                # ax.get_legend().remove()
                plt.grid(alpha=.7)

                plt.yticks(df['x_value'],
                           df.index.to_list())
                plt.tick_params(axis='y',
                                labelsize=8)

                plt.tick_params(axis='x',
                                labelsize=12)

                # Draw a red vertical line at the elbow point
                plt.vlines(x=elbow_index_x_axis,
                           ymin=0,
                           ymax=max(df['x_value']),
                           colors='red')
                plt.ylim([max(df['x_value']), 0])
                plt.tight_layout()

                fig = plt.gcf()
                # plt.tight_layout()
                if display:
                    plt.show()
                    plt.draw()
                if plot_path:
                    new_filename = f'feature_impo_{ticker}'
                    new_path = plot_path.with_name(new_filename)
                    fig.savefig(new_path,
                                dpi=dpi)
                    print(f'XGBoost {ticker} feature selection saved in path {new_path}')
        return importance_types

    def save_my_model(self):
        model_xgb = self.get_model()
        model_xgb.save_model(self.path_model_save)
        #  Convert NumPy arrays to lists to store them as json object
        self.predictions = {key: value.tolist() for key, value in self.predictions.items()}
        with open(self.path_model_results.joinpath('predictions_splits.json'), 'w') as f:
            json.dump(self.predictions, f, indent=4)
        print(f'the model has been saved in the path {self.path_model_save}')

    def load_model(self):
        if self.path_model_save.exists():
            loaded_model = xgb.Booster()
            loaded_model.load_model(self.path_model_save)
            return loaded_model
        else:
            print(f'unable to locate model in {self.path_model_save}')


if __name__ == '__main__':
    pass
