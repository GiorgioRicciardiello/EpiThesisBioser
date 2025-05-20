"""
Module: XGBoost Custom Utilities

This module contains custom classes and utilities to extend XGBoost functionality, focusing on training,
evaluation, and hyperparameter optimization. These implementations support both classification and regression tasks,
with features tailored for advanced use cases like cross-validation, learning rate scheduling, and feature importance analysis.

Classes:
---------
1. LRScheduler:
   - A custom callback to dynamically adjust the learning rate during training based on validation performance.

2. XGBoostModel:
   - A wrapper class for training, evaluating, and fine-tuning XGBoost models.
   - Features include:
     - Binary and multiclass classification, regression support.
     - Specificity-focused metrics and evaluation.
     - Cross-validation (k-fold) with custom loss and evaluation metrics.
     - Learning rate decay and early stopping integration.
     - Custom plots (AUC/PRC, feature importance, loss curves).

3. XGBoostGridSearch:
   - Implements grid search for hyperparameter tuning across multiple configurations.
   - Integrates k-fold cross-validation and specificity-based metric evaluation.

Key Features:
-------------
- Learning Rate Scheduler: Adjusts the learning rate dynamically during training.
- Metrics Focused: Includes custom specificity metric and other standard metrics.
- Hyperparameter Tuning: Grid search, random search, and Optuna-based optimization.
- Feature Importance: Supports advanced feature importance analysis with separate visualizations.
- Cross-validation Support: k-fold cross-validation integrated into model evaluation.
- Model Saving and Loading: Includes methods for saving predictions, labels, and trained models.

Dependencies:
-------------
- xgboost
- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- tqdm
- optuna

Usage:
------
1. Instantiate `XGBoostModel` or `XGBoostGridSearch` with desired parameters and datasets.
2. Use `train_and_eval_model` for basic training or `train_and_eval_model_k_fold_cv` for cross-validation.
3. Apply hyperparameter tuning using grid search or Optuna-based methods.
4. Visualize results using built-in plotting utilities.

Example:
--------
# Initialize the dataset
ds = Datasets(train_X, train_y, valid_X, valid_y, test_X, test_y)

# Define model parameters
params = {
    'learning_rate': 0.01,
    'max_depth': 6,
    'objective': 'binary:logistic'
}

# Train and evaluate the model
xgb_model = XGBoostModel(n_boosting=500, ds=ds, params=params)
true_labels, predictions, metrics = xgb_model.train_and_eval_model()

# Plot results
xgb_model.plot_training_val_loss(metrics)

"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import optuna
from numpy import ndarray
from pandas import DataFrame
from typing import List, Optional, Dict, Union, Tuple, Any
import xgboost as xgb
from pydantic import BaseModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, \
    balanced_accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score

from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy.typing as npt
import pathlib
import seaborn as sns
from matplotlib import pyplot as plt
import operator
import json
import matplotlib.cm as cm
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from scipy.optimize import minimize_scalar
import itertools
from tqdm import tqdm


@dataclass
class Datasets:
    def __init__(self,
                 train_X: pd.DataFrame,
                 train_y: pd.Series,
                 test_X: Optional[pd.DataFrame] = None,
                 test_y: Optional[pd.Series] = None,
                 valid_X: Optional[pd.DataFrame] = None,
                 valid_y: Optional[pd.Series] = None):
        self.train_X = train_X
        self.valid_X = valid_X
        self.test_X = test_X
        self.train_y = train_y
        self.valid_y = valid_y
        self.test_y = test_y
        self._create_splits_dict()

    def _create_splits_dict(self):
        self.splits = {
            'train_X': self.train_X,
            'valid_X': self.valid_X,
            'test_X': self.test_X,
            'train_y': self.train_y,
            'valid_y': self.valid_y,
            'test_y': self.test_y
        }

    def plot_stratified_distribution(self,
                                     output_path: Union[pathlib.Path, None]=None,
                                     show_plot:Optional[bool]=True,
                                     save_plot:Optional[bool]=True,
                                     ):
        """
        Plot the stratified target (categorical/ordinal) as a bar plot. The  x axis contains the train, validation, and
        test split. Each x-ticks has the bar of the count of each class in the split
        :return:
        """
        splits_target = {key: item for key, item in self.splits.items() if 'y' in key and item is not None}
        splits_count = {}
        for lbl_, split_ in splits_target.items():
            splits_count[lbl_] = split_.value_counts().to_dict()
        # Sorting each inner dictionary by its keys
        splits_count = {outer_k: dict(sorted(outer_v.items())) for outer_k, outer_v in splits_count.items()}

        df_splits_count = pd.DataFrame(splits_count)
        df_melted = df_splits_count.reset_index().melt(id_vars='index',
                                                       var_name='split',
                                                       value_name='count')
        df_melted.rename(columns={'index': 'class'},
                         inplace=True)

        # Now we can create a seaborn barplot with splits on the x-axis and count on the y-axis
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melted,
                    x='split',
                    y='count',
                    hue='class')
        plt.title('Counts of Classes across Different Splits')
        plt.xlabel('Split')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.grid(0.7)
        if save_plot and output_path is not None:
            plt.savefig(output_path.joinpath('Distribution_Model.png'), dpi=300)
        if show_plot:
            plt.show()

    def plot_distribution(self):
        """
        Plot how the train, validation, and test sets are distributed
        :return:
        """
        # Calculate the percentage of each class in the datasets
        train_percentages = self.train_y.value_counts(normalize=True) * 100
        valid_percentages = self.valid_y.value_counts(normalize=True) * 100
        test_percentages = self.test_y.value_counts(normalize=True) * 100

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 7))
        index = np.arange(len(train_percentages))
        bar_width = 0.25

        bar1 = ax.bar(index, train_percentages, bar_width, label='Train')
        bar2 = ax.bar(index + bar_width, valid_percentages, bar_width, label='Validation')
        bar3 = ax.bar(index + 2 * bar_width, test_percentages, bar_width, label='Test')

        ax.set_xlabel('Class')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Percentage of patients in each class per data split - Stratified')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(train_percentages.index)
        ax.legend()

        # Adding the percentages on top of the bars
        for bar in bar1 + bar2 + bar3:
            height = bar.get_height()
            ax.annotate('%.2f%%' % height,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

    def get_shape(self):
        return {
            'train_X': self.train_X.shape,
            'train_y': self.train_y.shape,
            'test_X': self.test_X.shape if self.test_X is not None else None,
            'test_y': self.test_y.shape if self.test_y is not None else None,
            'valid_X': self.valid_X.shape if self.valid_X is not None else None,
            'valid_y': self.valid_y.shape if self.valid_y is not None else None,
        }

    def get_info(self):
        return {
            'train_X' : self.train_X.info(),
            'valid_X' : self.valid_X.info() if self.valid_X is not None else None,
            'test_X' : self.test_X.info(),
            'train_y' : "Series size : " + str(self.train_y.size),
            'valid_y' : "Series size : " + str(self.valid_y.size) if self.valid_X is not None else 'None',
            'test_y' : "Series size : " + str(self.test_y.size)
        }

    def get_describe(self):
        return {
            'train_X': self.train_X.describe(),
            'valid_X': self.valid_X.describe() if self.valid_X is not None else None,
            'test_X': self.test_X.describe(),
            'train_y': self.train_y.describe(),
            'valid_y': self.valid_y.describe() if self.valid_y is not None else None,
            'test_y': self.test_y.describe()
        }

    def plot_distribution_2(self, layout: Optional[str] = 'vertical_stack', palette: Optional[str] = 'Set2'):
        """
        Plot distribution of target column for different splits.

        :param layout: Layout of the subplots, either 'stacked' or 'side_by_side'.
        :param palette: Seaborn color palette to use for the plots.
        """
        # Determine the number of subplots based on the layout
        if layout == 'side_by_side':
            nrows, ncols = 1, 3
            figsize = (18, 6)  # Wider figure for side-by-side layout
        elif layout == 'vertical_stack':
            nrows, ncols = 3, 1
            figsize = (6, 12)  # Taller figure for stacked layout
        else:
            raise ValueError("Invalid layout. Choose 'vertical_stack' or 'side_by_side'.")

        # Create subplots
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

        # Plot for each split
        splits = [('Train', self.train_y), ('Validation', self.valid_y), ('Test', self.test_y)]
        for idx, (split, data) in enumerate(splits):
            if data is not None:
                sns.histplot(data, kde=True, ax=axes[idx], palette=palette)
                axes[idx].set_title(f'{split} ({data.shape[0]})')
            else:
                axes[idx].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                axes[idx].set_title(f'{split} (No Data)')
                axes[idx].set_visible(False)

        plt.tight_layout()
        plt.show()

    def count_nans(self):
        nan_count = {
            'train_X' : self.train_X.isna().sum(),
            'valid_X' : self.valid_X.isna().sum(),
            'test_X' : self.test_X.isna().sum(),
            'train_y' : self.train_y.isna().sum(),
            'valid_y' : self.valid_y.isna().sum(),
            'test_y' : self.test_y.isna().sum()
        }

        return nan_count

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
                new_lr = (self.decay_factor ** epoch) * current_lr
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
                 n_boosting: int = 10000,
                 ds: Optional[Datasets] = None,
                 learning_task: str = 'regression',
                 num_classes: Optional[int] = None,
                 path_model_save: pathlib.Path = None,
                 path_model_results: pathlib.Path = None,
                 params:Optional[Dict] = None,
                 features_to_include: Optional[List[str]] = None,
                 best_threshold:Optional[float] = None,
                 threshold_metrics_maximize_binary:Optional[str] = 'auc',
                 invert_labels_metrics:bool=False
                 ):
        """

        :param n_boosting:
        :param ds:
        :param learning_task:
        :param num_classes:
        :param path_model_save:
        :param path_model_results:
        :param params:
        :param features_to_include:
        :param best_threshold: used for binary classification in the metrics reports. If none, we find the best
        :param invert_labels: Situation where the positive class is 0 and the negative class is 1. But the metrics should
            see the positive as 1 and negative as 0.
        threshold for the predictions using the training set. If float we use the one defined at the input.
        """
        self.seed = 42
        self.invert_labels_metrics = invert_labels_metrics
        self.learning_task = learning_task
        learning_tasks = {
            'regression': {
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
            },
            'classification_binary': {
                'objective': 'binary:logistic',  # For binary classification
                'eval_metric': 'logloss',  # 'auc',  # Binary classification error rate
                'num_class': None,
            },

            'classification': {
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
        if params:
            self.params.update(params)

        if self.learning_task == 'classification':
            self.params['num_class'] = ds.train_y.nunique()

        self.n_boosting = n_boosting  # num_boost_round

        self.xgb_model = None
        if features_to_include is None:
            features_to_include = ds.train_X.columns.tolist()

        if ds:
            has_eval_set = hasattr(ds, 'valid_X') and ds.valid_X is not None

            if self.learning_task == 'classification_binary':
                ds.train_y = ds.train_y.astype(int)
                if has_eval_set:
                    ds.valid_y = ds.valid_y.astype(int)
                if not ds.test_y is None:
                    ds.test_y = ds.test_y.astype(int)

            self.dtrain = xgb.DMatrix(data=ds.train_X[features_to_include],
                                      label=ds.train_y,
                                      feature_names=features_to_include
                                      )
            if has_eval_set:
                self.deval = xgb.DMatrix(data=ds.valid_X[features_to_include],
                                         label=ds.valid_y,
                                         feature_names=features_to_include)
            else:
                self.dval = None
            if not ds.test_X is None:
                self.dtest = xgb.DMatrix(ds.test_X[features_to_include],
                                         label=ds.test_y)
        else:
            self.dtrain = None
            self.dval = None
            self.dtest = None
        assert all([feature in self.dtrain.feature_names for feature in features_to_include]), \
            "Some features in 'features_to_include' are not present in the dataset columns."
        self.path_model_save = path_model_save
        self.path_model_results = path_model_results
        self.predictions = {}
        self.best_threshold = best_threshold  # for when we have a binary classification
        self.threshold_metrics_maximize_binary = threshold_metrics_maximize_binary

    def train_and_eval_model(self,
                             generate_plots: Optional[bool] = True,
                             plot_show: Optional[bool] = True,
                             output_margin: Optional[bool] = False) -> tuple[
        dict[str, Union[Optional[ndarray], Any]], dict[str, Optional[ndarray]], Union[
            tuple[dict[Any, dict[str, Union[float, int]]], Optional[dict]], tuple[DataFrame, Optional[dict]]]]:
        """
        Fits an XGBoost model on a subset of features using specified hyperparameters.

        :param ds: Datasets containing training, validation, and testing sets.
        :param features_to_include: List of features to include in the model.
        :param generate_plots: Flag to indicate if plots should be generated.
        :param plot_show: Flag to indicate if plots should be displayed.
        :return: Predictions and evaluation metrics.
        """

        # Custom evaluation metric for specificity
        def specificity_metric(preds, dtrain):
            """
            Custom evaluation metric for specificity.
            Args:
                preds: Predicted probabilities.
                dtrain: xgb.DMatrix containing labels.
            Returns:
                A tuple ('specificity', specificity_value).
            """
            labels = dtrain.get_label()
            preds_binary = (preds > 0.5).astype(int)  # Convert probabilities to binary
            tn = ((labels == 0) & (preds_binary == 0)).sum()
            fp = ((labels == 0) & (preds_binary == 1)).sum()

            # Avoid division by zero
            if tn + fp == 0:
                specificity = 0.0
            else:
                specificity = tn / (tn + fp)

            return 'specificity', float(specificity)  # Return scalar value


        # self.dtrain.get_label().shape
        # self.dtrain.get_data().shape
        #
        # self.dtest.get_data().shape
        # self.dtest.get_label().shape

        if not hasattr(self, 'dtest'):
            raise ValueError(f'XGBoostModel must have a test set in the method: self.train_and_eval_model() ')

        # Check if the evaluation set is provided
        has_eval_set = hasattr(self, 'deval')

        # Dynamically build the watchlist
        watchlist = [(self.dtrain, 'train')]
        if has_eval_set:
            watchlist.append((self.deval, 'valid'))

        eval_results = {}
        self.xgb_model = xgb.train(
            params=self.params,
            dtrain=self.dtrain,
            evals=watchlist,
            evals_result=eval_results,
            num_boost_round=self.n_boosting,
            feval=specificity_metric,
            verbose_eval=200,
        )

        # Plot training and validation loss curves if evaluation set is available
        self.plot_training_val_loss(results=eval_results,
                                    plot_path=self.path_model_results.joinpath('train_val_curve.png'))

        # Predicting for train, validation, and test sets
        pred_dtrain = self.xgb_model.predict(self.dtrain, output_margin=output_margin)
        pred_dval = self.xgb_model.predict(self.deval, output_margin=output_margin) if has_eval_set else None
        pred_dtest = self.xgb_model.predict(self.dtest, output_margin=output_margin)

        self.predictions = {"train": pred_dtrain,
                            "valid": pred_dval if has_eval_set else None,
                            "test": pred_dtest}

        true_labels = {"train": self.dtrain.get_label(),
                       "valid": self.deval.get_label() if has_eval_set else None,
                       "test": self.dtest.get_label()}

        if self.invert_labels_metrics:
            self.predictions, true_labels = self._invert_labels_and_predictions_binary(predictions=self.predictions,
                                                       true_labels=true_labels,
                                                       threshold=self.best_threshold,
                                                       apply_threshold=False)

        metrics = self.evaluate_model(true_labels=true_labels,
                                      pred=self.predictions,
                                      output_dtype='frame',
                                      generate_plots=generate_plots,
                                      plot_show=plot_show)

        self.plot_auc_prc_curves(
            predictions=self.predictions,
            true_labels=true_labels,
            file_name=f'auc_pcr_curve',
            plot_show=plot_show,
            output_path=self.path_model_results)

        return true_labels, self.predictions, metrics

    @staticmethod
    def _invert_labels_and_predictions_binary(predictions, true_labels,
                                              threshold=0.5,
                                              apply_threshold=False):
        """
        Inverts the true labels and predictions for each dataset split (train, valid, test).
        Applies a threshold to predictions if they are probabilities.

        Parameters:
            predictions (dict): Dictionary containing predictions for "train", "valid", and "test".
                                Predictions can be probabilities (floats) or binary values (0 or 1).
            true_labels (dict): Dictionary containing true labels for "train", "valid", and "test".
                                True labels must be binary (0 or 1).

        Returns:
            dict: Updated predictions with binary values inverted.
            dict: Updated true labels with binary values inverted.

        Raises:
            AssertionError: If true labels are not binary.
        """
        inverted_predictions = {}
        inverted_true_labels = {}

        for split in ["train", "valid", "test"]:
            # Handle predictions
            if predictions.get(split) is not None:
                pred = predictions[split]
                if np.issubdtype(pred.dtype, np.floating) and apply_threshold:  # Check if predictions are floats
                    pred = (pred > threshold).astype(int)  # Apply threshold
                inverted_predictions[split] = 1 - pred  # Invert binary predictions

            # Handle true labels
            if true_labels.get(split) is not None:
                labels = true_labels[split]
                # Ensure labels are binary
                assert np.array_equal(np.unique(labels), [0, 1]), f"True labels for {split} must be binary (0 or 1)."
                inverted_true_labels[split] = 1 - labels  # Invert binary labels

        return inverted_predictions, inverted_true_labels


    def train_and_eval_model_k_fold_cv(self,
                                       stratify:Optional[bool] = True,
                                       k: int = 3,
                                       plot_show:Optional[bool]=True,
                                       full_fold:Optional[xgb.core.DMatrix]=None,
                                       verbose:Optional[int]=200) -> tuple[
        DataFrame, dict[str, Optional[ndarray]]]:
        """
          Performs K-fold cross-validation on the model.

          :param k: Number of folds for cross-validation.
          :param full_fold: full DMatrix to split the train and validation sets for the folds, else it selects the
            default train set initialized at the constructor of the class
          :return: A dictionary containing the average metrics for each fold.
          """

        # Define a custom loss function to approximate maximizing specificity
        def specificity_loss(preds, dtrain):
            """
            Custom loss function to approximate maximizing specificity.

            Args:
            - preds: Predicted probabilities.
            - dtrain: DMatrix with labels.

            Returns:
            - grad: Gradient of the loss function.
            - hess: Hessian of the loss function.
            """
            labels = dtrain.get_label()
            preds = 1 / (1 + np.exp(-preds))  # Convert logits to probabilities

            # Calculate gradients and hessians
            grad = -labels * (1 - preds) + (1 - labels) * preds * 2  # Penalize FP more
            hess = preds * (1 - preds) * (1 + (1 - labels))

            return grad, hess

        # Custom evaluation metric for specificity
        def specificity_eval_metric(preds, dtrain):
            """
            Custom evaluation metric to calculate specificity.

            Args:
            - preds: Predicted probabilities.
            - dtrain: DMatrix with labels.

            Returns:
            - name: Name of the metric.
            - result: Computed specificity.
            """
            labels = dtrain.get_label()
            preds = (preds > 0.5).astype(int)  # Threshold at 0.5
            tn = np.sum((labels == 0) & (preds == 0))
            fp = np.sum((labels == 0) & (preds == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return 'specificity', specificity

        # Initialize variables
        if not full_fold is None:
            data = full_fold.get_data()
            labels = full_fold.get_label()
        else:
            # Extract data and labels from the training set
            data = self.dtrain.get_data()
            labels = self.dtrain.get_label()

        # Initialize StratifiedKFold if strata is provided, otherwise use a basic shuffle
        if stratify:
            skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
            fold_indices = list(skf.split(labels, y=labels))

            fold_data = []
            # Print value counts for each fold
            for fold, (train_idx, val_idx) in enumerate(fold_indices, start=1):
                train_labels = labels[train_idx]
                val_labels = labels[val_idx]
                fold_summary = {
                    "Fold": fold,
                    "Train_Counts": pd.Series(train_labels).value_counts().to_dict(),
                    "Validation_Counts": pd.Series(val_labels).value_counts().to_dict()
                }
                print(f"Fold {fold}:"
                      f"\n\tTrain label counts:\n\t\t{pd.Series(train_labels).value_counts().to_dict()} "
                      f"\n\tValidation label counts:\n\t\t{pd.Series(val_labels).value_counts().to_dict()}\n")
                fold_data.append(fold_summary)
            df_folds = pd.DataFrame(fold_data)
            df_folds.to_csv(self.path_model_results.joinpath('folds_count_labels.csv'), index=False)

        else:
            num_samples = data.shape[0]
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            fold_size = num_samples // k
            fold_indices = [
                (np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]]),
                 indices[i * fold_size:(i + 1) * fold_size])
                for i in range(k)
            ]

        self.predictions_k_fol = {}
        df_agg_metrics = pd.DataFrame()
        for fold, (train_indices, val_indices) in enumerate(fold_indices):
            print(f"Starting Fold {fold + 1}/{k}")

            # Define train/validation splits
            train_data = data[train_indices]
            train_labels = labels[train_indices]
            val_data = data[val_indices]
            val_labels = labels[val_indices]

            dtrain_fold = xgb.DMatrix(train_data, label=train_labels)
            deval_fold = xgb.DMatrix(val_data, label=val_labels)

            # Train the model
            eval_results = {}
            self.xgb_model = xgb.train(
                params=self.params,
                dtrain=dtrain_fold,
                custom_metric=specificity_eval_metric,  # Add custom evaluation metric
                obj=specificity_loss,  # Use custom loss function
                evals=[(dtrain_fold, 'train'), (deval_fold, 'valid')],
                evals_result=eval_results,
                num_boost_round=self.n_boosting,
                verbose_eval=verbose,
            )
            if plot_show:
                self.plot_training_val_loss(results=eval_results,
                                            plot_path=None,
                                            plot_show=plot_show,
                                            title=f'Loss for Fold {fold+1}')

            # Evaluate on validation set
            pred_dval = self.xgb_model.predict(deval_fold)
            pred_dtrain = self.xgb_model.predict(dtrain_fold)



            predictions_k_fold = {"train": pred_dtrain,
                            "valid": pred_dval,
                            "test": None}

            true_labels = {"train": dtrain_fold.get_label(),
                           "valid": deval_fold.get_label(),
                           "test": None}

            if self.invert_labels_metrics:
                predictions_k_fold, true_labels = self._invert_labels_and_predictions_binary(predictions=predictions_k_fold,
                                                                                           true_labels=true_labels,
                                                                                           threshold=self.best_threshold,
                                                                                           apply_threshold=False)

            self.predictions_k_fol[f'fold_{k}'] = predictions_k_fold
            # Store metrics
            fold_metrics, predictions = self.evaluate_model(
                true_labels=true_labels,
                pred=self.predictions_k_fol[f'fold_{k}'],
                output_dtype='frame',
                generate_plots=plot_show,
                plot_show=plot_show
            )
            fold_metrics['fold'] = f'fold_{fold}'
            df_agg_metrics = pd.concat([df_agg_metrics, fold_metrics], axis=0)
            title = (f"Fold {fold+1} With {self.params.get('eval_metric').capitalize()} \n "
                     f"Train: {np.round(eval_results.get('train').get(self.params.get('eval_metric'))[-1], 3)};"
                     f"Valid: {np.round(eval_results.get('valid').get(self.params.get('eval_metric'))[-1], 3)}")

            self.plot_probabilities_by_label_splits(true_labels=true_labels,
                                                    predictions=self.predictions_k_fol[f'fold_{k}'],
                                                    title=title,
                                                    best_threshold=0.5,
                                                    plot_show=plot_show)
            if plot_show:
                self.plot_auc_prc_curves(
                    predictions=self.predictions_k_fol[f'fold_{k}'],
                    true_labels=true_labels,
                    file_name=f'fold_{fold+1}_pcr_curve',
                    plot_show=plot_show,
                    output_path=self.path_model_results)

            # metrics_all_folds.append(fold_metrics)

        df_agg_metrics.sort_values(by='Splits', inplace=True)
        df_agg_metrics['N'] = df_agg_metrics[['TP', 'TN', 'FP', 'FN']].sum(axis=1).astype(int)
        # Aggregate metrics across all folds
        print("Cross-validation complete. Aggregated metrics:")

        return df_agg_metrics,  self.predictions_k_fol[f'fold_{k}']

    def train_and_eval_model_k_fold_cv_hla_test(self,
                                       k: int = 3,
                                       plot_show:Optional[bool]=True,
                                       verbose:Optional[int]=200,) -> tuple[
        DataFrame, dict[str, Optional[ndarray]], pd.DataFrame]:
        """
          Performs K-fold cross-validation on the model.

          :param k: Number of folds for cross-validation.
          :param full_fold: full DMatrix to split the train and validation sets for the folds, else it selects the
            default train set initialized at the constructor of the class
          :return: A dictionary containing the average metrics for each fold.
          """
        from scipy.sparse import csr_matrix
        # Define a custom loss function to approximate maximizing specificity
        def specificity_loss(preds, dtrain):
            """
            Custom loss function to approximate maximizing specificity.

            Args:
            - preds: Predicted probabilities.
            - dtrain: DMatrix with labels.

            Returns:
            - grad: Gradient of the loss function.
            - hess: Hessian of the loss function.
            """
            labels = dtrain.get_label()
            preds = 1 / (1 + np.exp(-preds))  # Convert logits to probabilities

            # Calculate gradients and hessians
            grad = -labels * (1 - preds) + (1 - labels) * preds * 2  # Penalize FP more
            hess = preds * (1 - preds) * (1 + (1 - labels))

            return grad, hess

        # Custom evaluation metric for specificity
        def specificity_eval_metric(preds, dtrain):
            """
            Custom evaluation metric to calculate specificity.

            Args:
            - preds: Predicted probabilities.
            - dtrain: DMatrix with labels.

            Returns:
            - name: Name of the metric.
            - result: Computed specificity.
            """
            labels = dtrain.get_label()
            preds = (preds > 0.5).astype(int)  # Threshold at 0.5
            tn = np.sum((labels == 0) & (preds == 0))
            fp = np.sum((labels == 0) & (preds == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return 'specificity', specificity


        # Extract data and labels from the training set
        data = self.dtrain.get_data()
        labels = self.dtrain.get_label()
        # dmatrix to dataframe
        df_folds = pd.DataFrame.sparse.from_spmatrix(data,
                                                     columns=self.dtrain.feature_names)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        fold_indices = list(skf.split(labels, y=labels))


        self.predictions_k_fol = {}
        df_agg_metrics = pd.DataFrame()
        for fold, (train_indices, val_indices) in enumerate(fold_indices):
            print(f"Starting Fold {fold + 1}/{k}")

            # Define train/validation splits
            train_data = data[train_indices]
            train_labels = labels[train_indices]
            val_data = data[val_indices]
            val_labels = labels[val_indices]

            dtrain_fold = xgb.DMatrix(train_data,
                                      label=train_labels,
                                      feature_names=self.dtrain.feature_names)
            deval_fold = xgb.DMatrix(val_data,
                                     label=val_labels,
                                     feature_names=self.dtrain.feature_names)

            # Train the model
            eval_results = {}
            self.xgb_model = xgb.train(
                params=self.params,
                dtrain=dtrain_fold,
                custom_metric=specificity_eval_metric,  # Add custom evaluation metric
                obj=specificity_loss,  # Use custom loss function
                evals=[(dtrain_fold, 'train'), (deval_fold, 'valid')],
                evals_result=eval_results,
                num_boost_round=self.n_boosting,
                verbose_eval=verbose,
            )
            # Evaluate on validation set
            pred_dval = self.xgb_model.predict(deval_fold)
            pred_dtrain = self.xgb_model.predict(dtrain_fold)

            # Optional: View feature importance
            # feature_importance = self.xgb_model.get_score(importance_type='gain')
            # # Print feature importance in a readable format
            # sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            # print("Feature Importance:")
            # for feature, importance in sorted_importance:
            #     print(f"{feature}: {importance}")
            # xgb.plot_importance(self.xgb_model, importance_type='gain')
            # plt.show()
            # Optional end: feature importance

            predictions_k_fold = {"train": pred_dtrain,
                            "valid": pred_dval,
                            "test": None}

            true_labels = {"train": dtrain_fold.get_label(),
                           "valid": deval_fold.get_label(),
                           "test": None}

            if self.invert_labels_metrics:
                predictions_k_fold, true_labels = self._invert_labels_and_predictions_binary(predictions=predictions_k_fold,
                                                                                           true_labels=true_labels,
                                                                                           threshold=self.best_threshold,
                                                                                           apply_threshold=False)
            self.predictions_k_fol[f'fold_{k}'] = predictions_k_fold

            # Store metrics
            fold_metrics, predictions = self.evaluate_model(
                true_labels=true_labels,
                pred=self.predictions_k_fol[f'fold_{k}'],
                output_dtype='frame',
                generate_plots=plot_show,
                plot_show=plot_show
            )

            # Insert true and predicted values into the DataFrame
            df_folds.loc[train_indices, f'fold{fold + 1}_train_true'] = true_labels.get('train')
            df_folds.loc[train_indices, f'fold{fold + 1}_train_pred'] = predictions.get('train')
            df_folds.loc[train_indices, f'fold{fold + 1}_train_pred_prob'] = predictions_k_fold.get('train')
            df_folds.loc[val_indices, f'fold{fold + 1}_val_true'] = true_labels.get('valid')
            df_folds.loc[val_indices, f'fold{fold + 1}_val_pred'] = predictions.get('valid')
            df_folds.loc[val_indices, f'fold{fold + 1}_val_pred_prob'] = predictions_k_fold.get('valid')

            fold_metrics['fold'] = f'fold_{fold}'
            df_agg_metrics = pd.concat([df_agg_metrics, fold_metrics], axis=0)
            title = (f"Fold {fold+1} With {self.params.get('eval_metric').capitalize()} \n "
                     f"Train: {np.round(eval_results.get('train').get(self.params.get('eval_metric'))[-1], 3)};"
                     f"Valid: {np.round(eval_results.get('valid').get(self.params.get('eval_metric'))[-1], 3)}")

            self.plot_probabilities_by_label_splits(true_labels=true_labels,
                                                    predictions=self.predictions_k_fol[f'fold_{k}'],
                                                    title=title,
                                                    best_threshold=0.5,
                                                    plot_show=plot_show)
            if plot_show:
                self.plot_auc_prc_curves(
                    predictions=self.predictions_k_fol[f'fold_{k}'],
                    true_labels=true_labels,
                    file_name=f'fold_{fold+1}_pcr_curve',
                    plot_show=plot_show,
                    output_path=self.path_model_results)

            # metrics_all_folds.append(fold_metrics)

        df_agg_metrics.sort_values(by='Splits', inplace=True)
        df_agg_metrics['N'] = df_agg_metrics[['TP', 'TN', 'FP', 'FN']].sum(axis=1).astype(int)
        # Aggregate metrics across all folds
        print("Cross-validation complete. Aggregated metrics:")


        return df_agg_metrics,  self.predictions_k_fol[f'fold_{k}'], df_folds


    def train_and_eval_model_k_fold_cv_with_hyperparameter_tuning(self, k: int = 3, param_grid: dict = None) -> tuple[
        pd.DataFrame, dict]:
        """
        Perform K-Fold cross-validation with hyperparameter tuning.

        :param k: Number of folds for cross-validation.
        :param param_grid: Dictionary of hyperparameters for grid search.
        :return: DataFrame with aggregated metrics and best parameters.
        """
        from sklearn.model_selection import ParameterGrid
        from scipy.sparse import vstack

        if param_grid is None:
            param_grid = {"eta": [0.01, 0.1],
                          "subsample": [0.8, 1],
                          "scale_pos_weight": [15, 16, 17, 18, 19],
                          "reg_lambda": 0.0008,
                          "gamma": 0.001,
                          "reg_alpha":  0.0008,
                          }

        best_params = None
        best_metric = float('inf')  # Assuming lower is better, e.g., loss
        all_results = []

        if not self.dval is None:
            # combine the train and validation set so they can be fully used in the folds.
            # otherwise the validation fold will be unused
            data_train = self.dtrain.get_data()
            labels_train = self.dtrain.get_label()

            data_valid = self.dval.get_data()
            labels_valid = self.dval.get_label()
            combined_data = vstack((data_train, data_valid))  # Stack rows of data
            combined_labels = np.hstack((labels_train, labels_valid))  # Concatenate labels
        else:
            # no validation set so we are using the full train set + test set
            combined_data = self.dtrain.get_data()
            combined_labels = self.dtrain.get_label()

        if self.dtest is None:
            raise ValueError(f'A test set must be define to run the XGBoostModel class method: '
                             f'\ntrain_and_eval_model_k_fold_cv_with_hyperparameter_tuning')

        # Create a new DMatrix with the combined data and labels
        combined_dmatrix = xgb.DMatrix(combined_data, label=combined_labels)
        for params in ParameterGrid(param_grid):
            print(f"Testing hyperparameters: {params}")
            self.params.update(params)

            # Perform K-Fold Cross-Validation
            metrics, _ = self.train_and_eval_model_k_fold_cv(k=k, full_fold=combined_dmatrix)

            # Aggregate the validation metric (e.g., average loss or accuracy)
            avg_val_metric = metrics.loc[metrics['Splits'] == 'valid', self.params.get('eval_metric')].mean()
            print(f"Average validation {self.params.get('eval_metric')} for current params: {avg_val_metric}")

            if avg_val_metric < best_metric:
                best_metric = avg_val_metric
                best_params = params

            all_results.append({"params": params, "avg_val_metric": avg_val_metric})

        print(f"Best hyperparameters: {best_params}, Metric: {best_metric}")

        # Train final model with the best hyperparameters
        self.params.update(best_params)
        final_metrics, predictions = self.train_and_eval_model_k_fold_cv(k=k)
        return pd.DataFrame(all_results), {"best_params": best_params, "final_metrics": final_metrics,
                                           "predictions": predictions}

    def apply_threshold_to_predicted_probabilities(self,
                                                   threshold:float,
                                                   predictions: Dict[str, Union[ndarray, None]],
                                                   inplace:bool=False,
                                                   ) -> Union[dict, Union[ndarray, None]]:
        """
        Apply the threshold to the predictions to obtain binary classifications.

        :param threshold: The threshold to apply.
        :return: A dictionary with binary predictions for each split.
        """
        binary_predictions = {}
        for split, pred_probs in predictions.items():
            if pred_probs is not None:
                binary_predictions[split] = (pred_probs >= threshold).astype(int)
            else:
                binary_predictions[split] = None
        if inplace:
            self.predictions = binary_predictions
            return None
        return binary_predictions

    def predict_on_loaded_model(self,
                                xgb_model,
                                ds: Datasets = None,
                                features_to_include: Optional[list[str]] = None,
                                generate_plots: Optional[bool] = True,
                                plot_show: Optional[bool] = False
                                ):
        """
        Make predictions on a loaded model
        :param ds:
        :param features_to_include:
        :param plot_show: (bool) Flag to determine whether to display the plot or not.
        :param generate_plots: (bool) Flag to determine whether to generate the plot or not.

        :return: predictions and evaluation metrics
        """
        if features_to_include is None:
            features_to_include = ds.train_X.columns.tolist()

        assert all([feature in ds.train_X.columns for feature in features_to_include])

        dtrain = xgb.DMatrix(data=ds.train_X[features_to_include],
                             label=ds.train_y)

        deval = xgb.DMatrix(data=ds.valid_X[features_to_include],
                            label=ds.valid_y)

        dtest = xgb.DMatrix(ds.test_X[features_to_include],
                            label=ds.test_y)

        pred_dtrain = xgb_model.predict(data=dtrain, output_margin=False, )

        pred_dval = xgb_model.predict(data=deval, output_margin=False, )
        pred_dtest = xgb_model.predict(data=dtest, output_margin=False, )

        self.predictions = {"train": pred_dtrain,
                            "valid": pred_dval,
                            "test": pred_dtest}

        true_labels = {"train": ds.train_y,
                       "valid": ds.valid_y,
                       "test": ds.test_y}

        if generate_plots:
            self._plot_performance(predictions=self.predictions,
                                   true_labels=true_labels,
                                   plot_show=plot_show)

        return self.predictions, self.evaluate_model(true_labels=true_labels, pred=self.predictions, output_dtype='frame')

    def _plot_performance(self,
                          predictions: Dict[str, Any],
                          true_labels: Dict[str, Any],
                          plot_show: bool = False) -> None:
        """
        Plot performance of a model based on its learning task.

        :param predictions: (Dict[str, Any]): A dictionary containing the model's predictions for different datasets.
        :param true_labels (Dict[str, Any]): A dictionary containing the true labels for different datasets.
        :param plot_show (bool): Flag to determine whether to display the plot or not.

        Returns:
            None
        """
        self._clear_plots()
        if self.learning_task == 'regression':
            for subset in predictions.keys():
                plt.figure(figsize=(10, 6))
                plt.scatter(x=true_labels[subset],
                            y=predictions[subset],
                            alpha=0.3)  # Adjust the dataset dynamically
                plt.plot([true_labels[subset].min(), true_labels[subset].max()],  # Dynamically adjust plot line
                         [true_labels[subset].min(), true_labels[subset].max()],
                         'k--', lw=4)  # Ideal line
                plt.xlabel('Truth')
                plt.ylabel('Predicted')
                plt.title(f'True vs. Predicted Values for {subset.capitalize()} Data')
                plt.tight_layout()
                plt.savefig(self.path_model_results.joinpath(f'scatter_truth_vs_pred_{subset}'))
                if plot_show:
                    plt.show()

        if self.learning_task == 'classification' or self.learning_task == 'classification_binary':
            # # evalaute classifier
            # # Print classification report
            # print(classification_report(ds.test_y, test_predictions))
            #
            # # Calculate and display accuracy
            # accuracy = accuracy_score(ds.test_y, test_predictions)
            # print("Accuracy:", accuracy)

            # Generate confusion matrix
            # Iterate through subsets of predictions
            # for subset in predictions.keys():
            #     true_subset = true_labels[subset]
            #     pred_subset = predictions[subset]
            #
            #     # Check if both true labels and predictions are non-empty
            #     if true_subset is not None and pred_subset is not None:
            #         # Compute confusion matrix
            #         conf_matrix = confusion_matrix(true_subset, pred_subset)
            #
            #         # Plotting confusion matrix
            #         plt.figure(figsize=(10, 7))
            #         sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", cbar=False)
            #         plt.title(f'Confusion Matrix - {subset}')
            #         plt.xlabel('Predicted')
            #         plt.ylabel('Truth')
            #         plt.tight_layout()
            #
            #         # Save the confusion matrix if a path is provided
            #         if self.path_model_results:
            #             plt.savefig(self.path_model_results.joinpath(f'confusion_matrix_{subset}'))
            #
            #         # Show the plot if specified
            #         if plot_show:
            #             plt.show()
            #     else:
            #         print(f"Skipping confusion matrix for subset '{subset}' as true labels or predictions are empty.")

            # Filter out subsets with valid data
            valid_subsets = {
                subset: (true_labels.get(subset), predictions.get(subset))
                for subset in predictions.keys()
                if true_labels.get(subset) is not None and predictions.get(subset) is not None
                   and len(true_labels.get(subset)) > 0 and len(predictions.get(subset)) > 0
            }
            import math
            # Determine the number of plots needed
            num_subsets = len(valid_subsets)
            if num_subsets == 0:
                print("No valid subsets with true labels and predictions. Exiting.")
            else:
                # Create a figure with one row and up to three columns
                cols = num_subsets
                rows = math.ceil(num_subsets / cols)
                fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6))

                # Flatten axes array for easier iteration if there are multiple rows
                axes = axes.flatten() if rows > 1 else [axes]
                axes = axes[0]

                for i, (subset, (true_subset, pred_subset)) in enumerate(valid_subsets.items()):
                    # Compute confusion matrix
                    conf_matrix = confusion_matrix(true_subset, pred_subset)

                    # Plot confusion matrix
                    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues",cbar=False, ax=axes[i])
                    axes[i].set_title(f'Confusion Matrix - {subset}')
                    axes[i].set_xlabel('Predicted')
                    axes[i].set_ylabel('Truth')
                # Adjust layout
                plt.tight_layout()

                # Save the figure if a path is provided
                if self.path_model_results:
                    plt.savefig(self.path_model_results.joinpath('confusion_matrices_all.png'))

                # Show the plot
                if plot_show:
                    plt.show()


    def train_and_eval_model_lr_scheduler(self,
                                          ds: Datasets = None,
                                          features_to_include: Optional[list[str]] = None,
                                          generate_plots: Optional[bool] = True,
                                          plot_show: Optional[bool] = False
                                          ):
        """
        Fits AdaBoostRegressor on a subset of features using specified hyperparameters

        :param ds: Datasets for training, validation and testing
        :param features_to_include: List of features to include in the model
        :param plot_show: (bool) Flag to determine whether to display the plot or not.
        :param generate_plots: (bool) Flag to determine whether to generate the plot or not.
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
            'colsample_bytree': 0.6,
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

        true_labels = {"train": ds.train_y,
                       "valid": ds.valid_y,
                       "test": ds.test_y}
        if generate_plots:
            self._plot_performance(predictions=self.predictions,
                                   true_labels=true_labels,
                                   plot_show=plot_show)

        return self.predictions, self.evaluate_model(true_labels=true_labels, pred=self.predictions, output_dtype='frame')

    def _train_and_eval_model_best_params_lr_scheduler(self,
                                                       best_params: Dict,
                                                       ds: Datasets = None,
                                                       generate_plots: Optional[bool] = True,
                                                       plot_show: Optional[bool] = False,
                                                       lr_decay: Optional[float] = 0.09
                                                       ):
        """

        :params best_params: best parameters obtained from hyperparameter tuning
        :param ds: Datasets for training, validation and testing
        :param plot_show: (bool) Flag to determine whether to display the plot or not.
        :param generate_plots: (bool) Flag to determine whether to generate the plot or not.
        :return: predictions and evaluation metrics
        """
        self.dtrain = xgb.DMatrix(data=ds.train_X,
                                  label=ds.train_y)

        self.deval = xgb.DMatrix(data=ds.valid_X,
                                 label=ds.valid_y)

        self.dtest = xgb.DMatrix(ds.test_X,
                                 label=ds.test_y)

        lr_scheduler = LRScheduler(monitor=self.params.get('eval_metric'),
                                   delta=0.001,
                                   patience=50,
                                   decay_factor=lr_decay,
                                   min_lr=1e-2,
                                   n_boosting=self.n_boosting)

        watchlist = [(self.dtrain, 'train'), (self.deval, 'valid')]
        eval_results = {}
        self.xgb_model = xgb.train(
            params=best_params,
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

        true_labels = {"train": ds.train_y,
                       "valid": ds.valid_y,
                       "test": ds.test_y}
        if generate_plots:
            self._plot_performance(predictions=self.predictions,
                                   true_labels=true_labels,
                                   plot_show=plot_show)

        return self.predictions, self.evaluate_model(true_labels=true_labels,
                                                     pred=self.predictions,
                                                     output_dtype='frame')

    def _train_and_eval_model_best_params_early_stopping(self,
                                                         best_params: Dict,
                                                         ds: Datasets = None,
                                                         generate_plots: Optional[bool] = True,
                                                         plot_show: Optional[bool] = False,
                                                         early_stopping_rounds:int=200,
                                                         ):
        """
        :param best_params: best parameters obtained from hyperparameter tuning
        :param ds: Datasets for training, validation, and testing
        :param generate_plots: (bool) Flag to determine whether to generate the plot or not.
        :param plot_show: (bool) Flag to determine whether to display the plot or not.
        :return: predictions and evaluation metrics
        """
        self.dtrain = xgb.DMatrix(data=ds.train_X, label=ds.train_y)
        self.deval = xgb.DMatrix(data=ds.valid_X, label=ds.valid_y)
        self.dtest = xgb.DMatrix(ds.test_X, label=ds.test_y)

        watchlist = [(self.dtrain, 'train'), (self.deval, 'valid')]
        eval_results = {}
        # early_stopping = xgb.callback.EarlyStopping(
        #     rounds=early_stopping_rounds,
        #     save_best=True,
        #     maximize=False,
        # )

        self.xgb_model = xgb.train(
            params=best_params,
            dtrain=self.dtrain,
            evals=watchlist,
            evals_result=eval_results,
            num_boost_round=self.n_boosting,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=200,
        )

        # if 'best_score' in early_stopping:
        #     print(f"Best score: {early_stopping.best_score} on round {early_stopping.best_iteration}")

        self.plot_training_val_loss(results=eval_results,
                                    plot_path=self.path_model_results.joinpath('train_val_curve.png'))

        pred_dtrain = self.xgb_model.predict(self.dtrain, output_margin=False)
        pred_dval = self.xgb_model.predict(self.deval, output_margin=False)
        pred_dtest = self.xgb_model.predict(self.dtest, output_margin=False)

        self.predictions = {"train": pred_dtrain, "valid": pred_dval, "test": pred_dtest}

        true_labels = {"train": ds.train_y, "valid": ds.valid_y, "test": ds.test_y}

        if generate_plots:
            self._plot_performance(predictions=self.predictions,
                                   true_labels=true_labels,
                                   plot_show=plot_show)

        return self.predictions, self.evaluate_model(ds=ds, pred=self.predictions, output_dtype='frame')

    def _objective_decay_or_early_stop(self,
                                       trial: optuna.Trial,
                                       X: pd.DataFrame,
                                       y: pd.Series,
                                       kfold,
                                       early_stopping_rounds: Union[int, None] = 300,
                                       lr_decay: Optional[float] = 0.09,
                                       patience: int = 10) -> float:
        """
        Function inside the objective of optuna. Here we have the option if to use the learning rate scheduler or not.
        If we do, the learning rate will vary while we train a model configuration, else the learning rate is one of
        the hyperparameters to optimize in the model
        :param trial:
        :param X:
        :param y:
        :param kfold:
        :param early_stopping_rounds:
        :param lr_decay:
        :param patience:
        :return:
        """

        if lr_decay is not None:
            # learning ate decay implementation
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
        else:
            # Early stop implementation
            # Define parameters to optimize
            params = {
                # Add Optuna suggested parameters
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
                'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1, log=True),
                'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.05),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-10, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-10, 10.0, log=True),
                **self.params,
            }

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
                                  # evals_result={},
                                  early_stopping_rounds=early_stopping_rounds,
                                  verbose_eval=False)
                valid_pred = model.predict(dvalid)
                valid_mae = mean_absolute_error(y.iloc[valid_index], valid_pred)
                maes.append(valid_mae)
            mean_mae = np.mean(maes)
        return mean_mae

    @staticmethod
    def run_hparam_random_search_sklearn(ds: Optional[Datasets]):
        from sklearn.model_selection import RandomizedSearchCV
        from xgboost import XGBClassifier
        X = ds.train_X
        y = ds.train_y
        param_dist = {
            # "n_estimators": [50, 100, 200, 500],
            # "learning_rate": np.linspace(0.01, 0.3, 10),  # Learning rate
            "max_depth": [7],
            "subsample": np.linspace(0.5, 1, 6),  # Fraction of samples
            "colsample_bytree": np.linspace(0.5, 1, 6),  # Fraction of features
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.001, 0.1, 0.2, 0.3],  # Minimum loss reduction
            "reg_alpha": np.linspace(0.001, 0.006, 10),  # L1 regularization
            "reg_lambda": np.linspace(0.001, 0.006, 10),  # L2 regularization
            "scale_pos_weight": np.linspace(1, len(y[y == 0]) / len(y[y == 1]), 10),  # Balance positive class
        }

        random_search = RandomizedSearchCV(
            estimator=XGBClassifier(use_label_encoder=False,
                                    eval_metric="auc",
                                    n_estimators=1000,
                                    random_state=42),
            param_distributions=param_dist,
            n_iter=50,  # Number of combinations to try
            scoring="roc_auc",  # Use AUC as the scoring metric
            cv=2,  # 2-fold cross-validation
            verbose=1,
            random_state=42,
            n_jobs=-1  # Parallelize
        )

        # Fit random search to training data
        random_search.fit(X, y)

        # Get best parameters and best score
        best_params = random_search.best_params_
        best_score = random_search.best_score_

        print("Best Parameters:", best_params)
        print("Best AUC Score:", best_score)

        # Evaluate on the validation set
        best_model = random_search.best_estimator_
        val_preds = best_model.predict_proba(ds.valid_X)[:, 1]
        val_auc = roc_auc_score(ds.valid_y, val_preds)
        print(f"Validation AUC: {val_auc}")


    @staticmethod
    def run_hparam_random_search_xgb(ds: Optional[Datasets],
                                 ):
        import random
        # Prepare your DMatrix (required for XGBoost GPU support)
        dtrain = xgb.DMatrix(ds.train_X, label=ds.train_y)
        dvalid = xgb.DMatrix(ds.valid_X, label=ds.valid_y)

        # Define the search space for hyperparameters
        param_dist = {
            "objective": "binary:logistic",
            "tree_method": "gpu_hist",  # Enable GPU support
            "eval_metric": "auc",
            "max_depth": [7],  # Single value as per your setup
            "subsample": np.linspace(0.5, 1, 6).tolist(),  # Convert to list
            "colsample_bytree": np.linspace(0.5, 1, 6).tolist(),  # Convert to list
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0.001, 0.1, 0.2, 0.3],
            "reg_alpha": np.linspace(0.001, 0.006, 10).tolist(),
            "reg_lambda": np.linspace(0.001, 0.006, 10).tolist(),
            "scale_pos_weight": [1, len(ds.train_y[ds.train_y == 0]) / len(ds.train_y[ds.train_y == 1])]
        }

        # Number of random iterations to try
        n_iter = 50
        best_params = None
        best_auc = 0

        # Random search
        for i in range(n_iter):
            # Randomly sample parameters
            params = {
                "objective": param_dist["objective"],
                "tree_method": param_dist["tree_method"],
                "eval_metric": param_dist["eval_metric"],
                "max_depth": random.choice(param_dist["max_depth"]),
                "subsample": random.choice(param_dist["subsample"]),
                "colsample_bytree": random.choice(param_dist["colsample_bytree"]),
                "min_child_weight": random.choice(param_dist["min_child_weight"]),
                "gamma": random.choice(param_dist["gamma"]),
                "reg_alpha": random.choice(param_dist["reg_alpha"]),
                "reg_lambda": random.choice(param_dist["reg_lambda"]),
                "scale_pos_weight": random.choice(param_dist["scale_pos_weight"])
            }

            # Train using cross-validation
            results = xgb.cv(
                params,
                dtrain,
                num_boost_round=200,  # Adjust based on needs
                nfold=2,  # Cross-validation folds
                early_stopping_rounds=10,
                metrics="auc",
                seed=42,
                verbose_eval=False
            )

            # Get the best AUC for this iteration
            mean_auc = results["test-auc-mean"].max()
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = params

            print(f"Iteration {i + 1}/{n_iter} - AUC: {mean_auc:.4f}")

        # Print the best parameters and AUC
        print("Best Parameters:", best_params)
        print("Best AUC Score:", best_auc)

        # Train final model with the best parameters
        final_model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=200,
            evals=[(dvalid, "validation")],
            early_stopping_rounds=10,
            verbose_eval=True
        )

        # Validate the model
        val_preds = final_model.predict(dvalid)
        val_auc = roc_auc_score(ds.valid_y, val_preds)
        print(f"Validation AUC: {val_auc}")

    def run_hparam_search_optuna(self,
                                 ds: Optional[Datasets],
                                 n_splits: int = 10,
                                 early_stopping_rounds: Optional[int] = 200,
                                 n_trials: int = 10,
                                 lr_decay: float = None,
                                 plt_show:Optional[bool]=False) -> tuple[Any, DataFrame, dict[str, ndarray], Union[dict, DataFrame]]:
        """
        Run hyperparameter search using Optuna to find the best combination of hyperparameters for the XGBoost model.

        Args:
        - ds (Any): Dataset object containing training data.
        - n_splits (int, optional): Number of splits for cross-validation. Defaults to 10.
        - n_trials (int, optional): Set total number of trials for Optuna
        - early_stopping_rounds (int, optional): Number of rounds for early stopping during model training.
                                                 Defaults to 100.

        Returns:
            -dict,  dictionary of the best parameters
            - Dataframe, cv results as a dataframe
            - dict[str, array], predictions of the model with the best parameter on each split
            - Dataframe, metrics of the best model
        """
        X = ds.train_X
        y = ds.train_y
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        # Create a tqdm progress bar
        with tqdm(total=n_trials, desc="Optimizing", unit="trial") as progress_bar:
            # Define objective function
            def objective(trial):
                # Inside the objective function, you can use tqdm to track the progress
                progress_bar.update(1)  # Update tqdm progress bar
                return self._objective_decay_or_early_stop(trial=trial,
                                                           X=X,
                                                           y=y,
                                                           kfold=kfold,
                                                           early_stopping_rounds=early_stopping_rounds,
                                                           lr_decay=lr_decay)

            # Create an Optuna study
            study = optuna.create_study(direction='minimize')
            # Optimize hyperparameters
            study.optimize(objective, n_trials=n_trials)

        # Extract best parameters
        best_params = study.best_params
        print(f'Best params: {best_params}')

        # Update model with best parameters
        self._update_params(best_params=best_params)

        # train a xgbmodel with the best params, so we can return the model and the metrics
        if lr_decay:
            predictions_xgb, metrics_xgb = self._train_and_eval_model_best_params_lr_scheduler(best_params=self.params,
                                                                                               ds=ds,
                                                                                               lr_decay=lr_decay,
                                                                                               generate_plots=True,
                                                                                               plot_show=plt_show)
        else:
            predictions_xgb, metrics_xgb = self._train_and_eval_model_best_params_early_stopping(
                best_params=self.params,
                early_stopping_rounds=early_stopping_rounds,
                ds=ds,
                generate_plots=True,
                plot_show=plt_show)

        cv_results_df = pd.DataFrame.from_dict(study.trials_dataframe())

        # save the cv results
        cv_results_df.to_excel(self.path_model_results.joinpath('optuna_results_df.xlsx'))
        df_best_params = pd.DataFrame(best_params, index=[0])
        df_best_params.to_excel(self.path_model_results.joinpath('optuna_best_params.xlsx'))

        return best_params, cv_results_df, predictions_xgb, metrics_xgb

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

    @staticmethod
    def plot_probabilities_by_label(y_true_train: np.ndarray,
                                    y_pred_train: np.ndarray,
                                    best_threshold:float=0.5):
        """
        Plot the predicted probabilities hue-coded by the true labels.

        :param y_true_train: Ground truth binary labels (0 or 1).
        :param y_pred_train: Predicted probabilities (or scores) from the model.
        :param best_threshold: Best threshold, float
        """
        # Create a DataFrame for easier manipulation
        data = pd.DataFrame({
            'Predicted Probability': y_pred_train,
            'True Label': y_true_train
        })
        data.reset_index(inplace=True, drop=False, names='index')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=data,
                        x='index',
                        y='Predicted Probability',
                        hue='True Label',
                        palette='coolwarm',
                        s=50)
        plt.axhline(y=best_threshold, color='red', linestyle='--', label='Separation Threshold')
        plt.xlim(data['index'].min(), data['index'].max())  # Adjust x-axis to sample range
        plt.ylim(-0.05, 1.05)  # Predicted probabilities are in [0, 1]
        plt.legend(title='Legend', loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Probability')
        plt.title('Classification Predictions with Separation Line')
        plt.grid(True, alpha=0.7)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_probabilities_by_label_splits(predictions: dict,
                                           true_labels: dict,
                                           best_threshold: float = 0.5,
                                           title: Optional[str] = None,
                                           plot_show:Optional[bool]=True):
        """
        Plot the predicted probabilities hue-coded by the true labels for train, valid, and test splits.

        :param predictions: Dictionary containing predicted probabilities for "train", "valid", and "test".
        :param true_labels: Dictionary containing true labels for "train", "valid", and "test".
        :param best_threshold: Threshold value for the separation line.
        """
        splits = ['train', 'valid', 'test']
        available_splits = [split for split in splits if predictions.get(split) is not None]
        num_splits = len(available_splits)

        if num_splits == 0:
            print("No splits available for plotting.")
            return

        # Create subplots
        fig, axes = plt.subplots(1, num_splits, figsize=(6 * num_splits, 6), sharey=True)
        if num_splits == 1:  # Ensure axes is iterable for a single subplot
            axes = [axes]

        for i, split in enumerate(available_splits):
            y_pred = predictions[split]
            y_true = true_labels[split]

            # Prepare DataFrame for plotting
            data = pd.DataFrame({
                'Predicted Probability': y_pred,
                'True Label': y_true
            })
            data.reset_index(inplace=True, drop=False, names='index')
            true_counts = data['True Label'].value_counts()
            # Plot the scatter plot
            sns.scatterplot(ax=axes[i],
                            data=data,
                            x='index',
                            y='Predicted Probability',
                            hue='True Label',
                            palette='coolwarm',
                            s=50)
            # Add the separation threshold
            axes[i].axhline(y=best_threshold, color='red', linestyle='--', label='Separation Threshold')

            # Adjust plot settings
            axes[i].set_xlim(data['index'].min(), data['index'].max())
            axes[i].set_ylim(-0.05, 1.05)
            axes[i].set_title(f'{split.capitalize()} Split\n Counts:{true_counts}')
            axes[i].set_xlabel('Sample Index')
            axes[i].set_ylabel('Predicted Probability' if i == 0 else "")
            axes[i].legend(title='Legend', loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
            axes[i].grid(True, alpha=0.7)
        if title:
            fig.suptitle(title, fontsize=16, y=1)
        plt.tight_layout()
        if plot_show:
            plt.show()
        plt.clf()
        plt.cla()
        plt.close()

    def plot_training_val_loss(self, results: dict,
                               plot_path: Optional[pathlib.Path] = None,
                               plot_show: Optional[bool] = False,
                               title:Optional[str] = None):
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
        if title:
            fig.suptitle(title, fontsize=16, y=1)
        if plot_path:
            fig.savefig(plot_path, dpi=300)
        if plot_show:
            plt.show()
        self._clear_plots()

    def get_model(self):
        """return the xgboost model from the training method"""
        return self.xgb_model

    @staticmethod
    def plot_features_separate_figures(xgb_model: xgb,
                                       feature_names: Optional[list] = None,
                                       plot_path: pathlib.Path = None,
                                       display: Optional[bool] = False,
                                       my_palette: Optional[str] = 'dark:salmon_r',
                                       fig_title: Optional[int] = 24,
                                       big_title: Optional[int] = 26,
                                       legend_size: Optional[int] = 24,
                                       figsize: Optional[tuple] = (8, 24),
                                       dpi: Optional[int] = 300,
                                       orientation: Optional[str] = 'h',
                                       render_plot: bool = False) -> Union[Dict, pd.DataFrame]:
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

        # Function for retrieving nested dictionary keys
        def get_keys(dictionary: dict, only_inner: Optional[bool] = False):
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

        # Function for detecting the elbow point
        def detect_elbow(data):
            from scipy.spatial.distance import cdist
            first_point = [0, data[0]]
            last_point = [len(data) - 1, data[-1]]
            coordinates = [[i, v] for i, v in enumerate(data)]
            distances = cdist([first_point, last_point], coordinates, 'euclidean').sum(axis=0)
            return np.argmax(distances)

        # Retrieve feature importance metrics
        importance_types = {'weight': {}, 'gain': {}, 'cover': {}, 'total_gain': {}, 'total_cover': {}}
        for imp_ in importance_types.keys():
            imp_importance = xgb_model.get_score(importance_type=imp_, fmap='')
            importance_types[imp_] = dict(sorted(imp_importance.items(), key=operator.itemgetter(1), reverse=True))

        if render_plot:
            if orientation == 'v':
                if figsize[1] > figsize[0]:
                    figsize = (figsize[1], figsize[0])

            all_questions = get_keys(importance_types, only_inner=True)
            set_questions = list(set(all_questions))
            palette = sns.color_palette(my_palette, len(set_questions))
            df = pd.DataFrame(0, columns=[*importance_types.keys()], index=set_questions)
            df['type_color'] = [palette[i % len(palette)] for i in range(len(set_questions))]

            for pred_ in set_questions:
                for measure_ in importance_types.keys():
                    df.loc[pred_, measure_] = importance_types[measure_].get(pred_, 0)

            for ticker in importance_types.keys():
                df.sort_values(by=ticker, ascending=False, inplace=True)
                df['x_value'] = range(len(df))
                elbow_index = detect_elbow(df[ticker])
                elbow_value = df[ticker][elbow_index]

                plt.figure(figsize=figsize, dpi=dpi)
                ax = plt.subplot(1, 1, 1)

                # Orientation control
                if orientation == 'h':
                    sns.barplot(data=df, x=ticker, y=df.index, palette=my_palette, ax=ax, orient='h')
                    plt.axvline(x=elbow_value, color='red', linestyle='--')
                elif orientation == 'v':
                    sns.barplot(data=df, x=df.index, y=ticker, palette=my_palette, ax=ax, orient='v')
                    plt.axhline(y=elbow_value, color='red', linestyle='--')

                ax.set_title(ticker.capitalize(), fontsize=fig_title)
                ax.set_xlabel("Score")
                ax.set_ylabel("Features")
                plt.tight_layout()

                if display:
                    plt.show()

                if plot_path:
                    save_path = plot_path / f'feature_importance_{ticker}.png'
                    plt.savefig(save_path, dpi=dpi)
                    print(f'Feature importance plot saved at {save_path}')

        # importance to frame
        # Create DataFrame
        # df_importance_types = pd.DataFrame(importance_types)

        return importance_types


    def save_my_model(self):
        model_xgb = self.get_model()
        model_xgb.save_model(self.path_model_save)
        #  Convert NumPy arrays to lists to store them as json object
        self.predictions = {key: value.tolist() for key, value in self.predictions.items()}
        true_labels = {
            'train': self.dtrain.get_label().tolist(),
            'valid': self.deval.get_label().tolist(),
            'test': self.dtest.get_label().tolist(),
        }
        pred_and_lbls = {
            'predictions': self.predictions,
            'true_labels': true_labels,
        }

        with open(self.path_model_results.joinpath('SplitPredTrueLbls.json'), 'w') as f:
            json.dump(pred_and_lbls, f, indent=4)
        print(f'the model has been saved in the path {self.path_model_save}')

    def get_model_outputs(self, data_splits) -> dict:
        predictions = {key: np.array(value).tolist() for key, value in self.predictions.items()}
        true_labels = {key: data.get_label() for key, data in data_splits.items()}

        return {
            'predictions': predictions,
            'true_labels': true_labels
        }


    def load_model(self):
        if self.path_model_save.exists():
            loaded_model = xgb.Booster()
            loaded_model.load_model(self.path_model_save)
            return loaded_model
        else:
            print(f'unable to locate model in {self.path_model_save}')

    def _clear_plots(self):
        plt.clf()
        plt.cla()
        # plt.close()

    @staticmethod
    def plot_auc_prc_curves(predictions: dict,
                            true_labels: dict,
                            plot_show: Optional[bool] = False,
                            output_path: Optional[pathlib.Path] = None,
                            file_name: Optional[str] = None) -> plt.Figure:
        """
        Plot AUC (ROC) and PRC curves for train, validation, and test datasets in a grid layout.

        :param predictions: Dictionary containing predicted probabilities for train, valid, and test.
                            Format: {"train": ndarray, "valid": ndarray, "test": ndarray}
        :param true_labels: Dictionary containing true labels for train, valid, and test.
                            Format: {"train": ndarray, "valid": ndarray, "test": ndarray}
        :param plot_show: Whether to display the plot.
        :param output_path: Path to save the plot.
        :param file_name: Name of the output file.
        :return: A matplotlib figure.
        """
        if file_name is None:
            file_name = 'auc_prc_plot'

        # Filter splits where predictions are available
        splits = [split for split in predictions.keys() if predictions.get(split) is not None]

        # Initialize figure
        fig, axes = plt.subplots(nrows=2,
                                 ncols=len(splits),
                                 figsize=(6 * len(splits), 10),
                                 sharey='row')

        if len(splits) == 1:
            axes = np.expand_dims(axes, axis=1)  # Ensure 2D shape for consistent indexing

        for idx, split in enumerate(splits):
            y_true = true_labels.get(split)
            y_pred = predictions.get(split)

            if y_true is not None and y_pred is not None:
                # Calculate value counts
                value_counts = {int(unique): count for unique, count in zip(*np.unique(y_true, return_counts=True))}

                # Calculate metrics for PRC
                precision, recall, prc_thresholds = precision_recall_curve(y_true, y_pred)
                average_precision = average_precision_score(y_true, y_pred)
                auc_prc = auc(recall, precision)

                # Calculate metrics for AUC (ROC)
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
                auc_roc = auc(fpr, tpr)

                # Find the best threshold for ROC (nearest to (0.9, 0.9))
                roc_points = np.vstack((fpr, tpr)).T
                roc_distances = np.linalg.norm(roc_points - np.array([0.9, 0.9]), axis=1)
                roc_nearest_idx = np.argmin(roc_distances)
                roc_nearest_point = roc_points[roc_nearest_idx]


                # Plot ROC curve (AUC) in the first row
                ax = axes[0, idx]
                ax.plot(fpr, tpr,
                        label=f'AUC={auc_roc:.3f}',
                        linewidth=2)
                # ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
                ax.scatter(roc_nearest_point[0], roc_nearest_point[1],
                           color='blue',
                           s=100,
                           zorder=5,
                           label=f"Thresh={roc_thresholds[roc_nearest_idx]:.3}")
                ax.set_title(f"{split.capitalize()} ROC Curve\nLabels: {value_counts}")
                ax.set_xlabel('False Positive Rate')
                if idx == 0:
                    ax.set_ylabel('True Positive Rate')
                ax.legend(loc='lower right', fontsize=12)
                ax.grid(alpha=0.7)

                # Plot PRC curve in the second row
                # Find the best threshold for PRC (nearest to (0.9, 0.9))
                prc_points = np.vstack((recall, precision)).T
                prc_distances = np.linalg.norm(prc_points - np.array([0.9, 0.9]), axis=1)
                prc_nearest_idx = np.argmin(prc_distances)
                prc_nearest_point = prc_points[prc_nearest_idx]

                ax = axes[1, idx]
                ax.plot(recall, precision,
                        label=f'PRC AUC={auc_prc:.3f}, Avg. Prec={average_precision:.3f}',
                        linewidth=2)
                ax.scatter(prc_nearest_point[0], prc_nearest_point[1],
                           color='blue',
                           s=100,
                           zorder=5,
                           label=f"Thresh={float(prc_thresholds[prc_nearest_idx]):.3f}")
                ax.set_title(f"{split.capitalize()} PRC\nLabels: {value_counts}")
                ax.set_xlabel('Recall')
                if idx == 0:
                    ax.set_ylabel('Precision')
                ax.legend(loc='lower left', fontsize=12)
                ax.grid(alpha=0.7)

        plt.tight_layout()

        # Save or show the plot
        if output_path:
            plt.savefig(output_path.joinpath(f'{file_name}.png'), dpi=300)
        if plot_show:
            plt.show()

        return fig


    def evaluate_model(self,
                       true_labels: dict,
                       pred: dict,
                       output_dtype: Optional[str] = 'dict',
                       generate_plots: Optional[bool] = True,
                       plot_show: Optional[bool] = True) -> Union[
        tuple[dict[Any, dict[str, Union[float, int]]], Optional[dict]], tuple[DataFrame, Optional[dict]]]:
        """
        Based on the learning task, create a metrics dataframe for each available split.

        :param true_labels: True labels as a dict, keys are: train, val, test.
        :param pred: Prediction labels as a dict, keys are: train, val, test.
        :param output_dtype: Output type, either 'dict' or 'dataframe'.
        :param generate_plots: generates plot of the metrics
        :param plot_show: show the plots of the metrics
        :return: Metrics as a dictionary or DataFrame.
        """

        def compute_binary_classification_metrics(y_true, y_pred) -> dict[str, Union[float, int]]:
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "F1 Score": f1_score(y_true, y_pred, average='weighted'),
                "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred)
            }
            try:
                auc_score = roc_auc_score(y_true, y_pred)
            except Exception as e:
                print(f"An error occurred calculating AUC: {e}")
                auc_score = -1

            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            # Calculate sensitivity (recall for positive class) and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

            metrics.update({
                'AUC': auc_score,
                'TP': tp,
                'TN': tn,
                'FP': fp,
                'FN': fn,
                'Sensitivity': sensitivity,
                'Specificity': specificity
            })
            return metrics

        def compute_multiclass_classification_metrics(y_true, y_pred)-> dict[str, Union[float, int]]:
            metrics = {
                "Accuracy": accuracy_score(y_true, y_pred),
                "F1 Score": f1_score(y_true, y_pred, average='weighted'),
                "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred)
            }
            return metrics

        def compute_regression_metrics(y_true, y_pred)-> dict[str, Union[float, int]]:
            return {
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred)
            }

        if self.learning_task == 'classification_binary':
            if self.best_threshold is None:
                self.best_threshold = self.find_best_threshold_for_predictions(
                    y_true_train=true_labels.get('train'),
                    y_pred_train=pred.get('train'),
                    metric=self.threshold_metrics_maximize_binary
                )
            if plot_show:
                self.plot_probabilities_by_label_splits(true_labels=true_labels,
                                                        predictions=pred,
                                                        best_threshold=0.5,
                                                        plot_show=plot_show)
            pred = self.apply_threshold_to_predicted_probabilities(threshold=self.best_threshold,
                                                                   inplace=False,
                                                                   predictions=pred)

        metrics = {}
        splits = [key for key, val in pred.items() if not val is None]
        for split in splits:
            if split in pred:
                if self.learning_task == 'regression':
                    metrics[split] = compute_regression_metrics(y_true=true_labels[split],
                                                                y_pred=pred[split])
                elif self.learning_task == 'classification_binary':
                    metrics[split] = compute_binary_classification_metrics(y_true=true_labels[split],
                                                                           y_pred=pred[split])
                elif self.learning_task == 'classification':
                    metrics[split] = compute_multiclass_classification_metrics(y_true=true_labels[split],
                                                                               y_pred=pred[split])

        metrics_df = pd.DataFrame(metrics).T
        metrics_df.reset_index(inplace=True, drop=False, names='Splits')
        if self.path_model_results:
            metrics_df.to_excel(self.path_model_results.joinpath(r'metrics_on_splits.xlsx'),
                                index=True)
        if generate_plots:
            self._plot_performance(predictions=pred,
                                   true_labels=true_labels,
                                   plot_show=plot_show)

        if output_dtype == 'dict':
            return metrics, pred
        else:
            return metrics_df, pred

    @staticmethod
    def find_best_threshold_for_predictions(y_true_train: np.ndarray,
                                            y_pred_train: np.ndarray,
                                            metric: str = 'f1') -> float:
        """
        Find the best threshold for binary classification predictions based on a specific metric.
        Uses optimization for fine-tuned threshold selection.

        :param y_true_train: Ground truth binary labels.
        :param y_pred_train: Predicted probabilities (or scores).
        :param metric: Metric to optimize. Options: 'f1', 'accuracy', 'precision', 'recall', 'auc'.
        :return: Best threshold based on the metric.
        """

        def metric_for_threshold(threshold):
            y_pred_thresh = (y_pred_train >= threshold).astype(int)
            if metric == 'f1':
                return -f1_score(y_true_train, y_pred_thresh)
            elif metric == 'accuracy':
                return -accuracy_score(y_true_train, y_pred_thresh)
            elif metric == 'sensitivity':  # Sensitivity (Recall)
                return -recall_score(y_true_train, y_pred_thresh)
            elif metric == 'specificity':  # Specificity (True Negative Rate)
                tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_thresh).ravel()
                specificity = tn / (tn + fp)
                return -specificity
            elif metric == 'auc':
                return -roc_auc_score(y_true_train, y_pred_thresh)
            else:
                raise ValueError("Unsupported metric. Choose from 'f1', 'accuracy', 'sensitivity', 'specificity', 'auc'.")

        # Use scalar minimization for the threshold search
        result = minimize_scalar(metric_for_threshold, bounds=(0.0, 1.0), method='bounded')
        best_threshold = result.x
        best_metric_value = -result.fun

        print(f"Best threshold based on {metric}: {best_threshold:.4f} with score: {best_metric_value:.4f}")
        return best_threshold


    @staticmethod
    def find_best_threshold_for_predictions_old(y_true_train: np.ndarray,
                                                y_pred_train: np.ndarray,
                                                metric: str = 'f1') -> float:
        """
        Find the best threshold for the predictions after a softmax based on the specified metric.

        :param metric: Metric to optimize. Options: 'f1', 'accuracy', 'precision', 'recall', 'auc'.
        :return: Best threshold for the train set based on the specified metric.
        """

        def plot_probabilities_by_label(y_true_train: np.ndarray,
                                        y_pred_train: np.ndarray,
                                        best_threshold: float = 0.5):
            """
            Plot the predicted probabilities hue-coded by the true labels.

            :param y_true_train: Ground truth binary labels (0 or 1).
            :param y_pred_train: Predicted probabilities (or scores) from the model.
            :param best_threshold: Best threshold, float
            """
            # Create a DataFrame for easier manipulation
            data = pd.DataFrame({
                'Predicted Probability': y_pred_train,
                'True Label': y_true_train
            })
            data.reset_index(inplace=True, drop=False, names='index')
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=data,
                            x='index',
                            y='Predicted Probability',
                            hue='True Label',
                            palette='coolwarm',
                            s=50)
            plt.axhline(y=best_threshold, color='red', linestyle='--', label='Separation Threshold')
            plt.xlim(data['index'].min(), data['index'].max())  # Adjust x-axis to sample range
            plt.ylim(-0.05, 1.05)  # Predicted probabilities are in [0, 1]
            plt.legend(title='Legend', loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True)
            plt.xlabel('Sample Index')
            plt.ylabel('Predicted Probability')
            plt.title('Classification Predictions with Separation Line')
            plt.grid(True, alpha=0.7)
            plt.tight_layout()
            plt.show()

        plot_probabilities_by_label(y_true_train, y_pred_train)

        best_metric_value = -1
        best_threshold = 0.0
        thresholds = np.arange(0.0, 1.0, 0.1)  # Step size reduced to 0.001

        for threshold in thresholds:
            y_pred_thresh = (y_pred_train >= threshold).astype(int)
            if metric == 'f1':
                metric_value = f1_score(y_true_train, y_pred_thresh)
            elif metric == 'accuracy':
                metric_value = accuracy_score(y_true_train, y_pred_thresh)
            elif metric == 'sensitivity':  # Sensitivity (Recall)
                metric_value = recall_score(y_true_train, y_pred_thresh)
            elif metric == 'specificity':  # Specificity (True Negative Rate)
                tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_thresh).ravel()
                specificity = tn / (tn + fp)
                metric_value = specificity
            elif metric == 'auc':
                metric_value = -roc_auc_score(y_true_train, y_pred_thresh)
            else:
                raise ValueError(
                    "Unsupported metric. Choose from 'f1', 'accuracy', 'sensitivity', 'specificity', 'auc'.")
            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_threshold = threshold

        print(f"Best threshold based on {metric}: {best_threshold} with score: {best_metric_value}")
        return best_threshold


class XGBoostGridSearch:
    def __init__(self, base_params, param_grid, n_boosting, ds, result_path, k_folds=5):
        """


        :param base_params:
        :param param_grid:
        :param n_boosting:
        :param ds:
        :param result_path:
        :param k_folds:

        Usage:
            # Define the parameter grid
            param_grid = {
                'reg_lambda': [2, 5],
                'gamma': [3, 5],
                'reg_alpha': [5, 7],
                'min_child_weight': [3, 5, 7],
            }

            # Base parameters
            base_params = {
                                     # Good results
                                     'scale_pos_weight': 16,
                                     'reg_lambda': 0.0008,
                                     'max_depth': 7,  # 7
                                     'gamma': 0.001,
                                     'reg_alpha': 0.0008,
                                     # 'colsample_bytree': 0.5
                                     'min_child_weight': 5,
                                 }
            # Initialize the grid search
            grid_search = XGBoostGridSearch(
                base_params=base_params,
                param_grid=param_grid,
                n_boosting=400,
                ds=ds,
                result_path=result_path,
                k_folds=5
            )

            # Run the grid search
            best_params, best_specificity, results_df = grid_search.run_grid_search()

            # Print the best parameters and results
            print("Best Parameters:", best_params)
            print("Best Specificity:", best_specificity)
            print("Grid Search Results:")
            print(results_df)


        """
        self.base_params = base_params
        self.param_grid = param_grid
        self.n_boosting = n_boosting
        self.ds = ds
        self.result_path = result_path
        self.k_folds = k_folds

    def train_model(self, params):
        # Instantiate the model with the current parameter set
        myxgb = XGBoostModel(
            n_boosting=self.n_boosting,
            ds=self.ds,
            learning_task='classification_binary',
            path_model_results=self.result_path,
            params=params,
            best_threshold=0.5,
            threshold_metrics_maximize_binary='specificity',
            invert_labels_metrics=True
        )

        # Perform k-fold cross-validation
        df_agg_metrics, _ = myxgb.train_and_eval_model_k_fold_cv(k=self.k_folds,
                                                                 plot_show=False,
                                                                 verbose=None)

        # Compute average specificity across validation splits
        specificity_avg = df_agg_metrics.loc[df_agg_metrics['Splits'] == 'valid', 'Specificity'].mean()

        return specificity_avg

    def run_grid_search(self):
        # Generate all combinations of hyperparameters
        keys, values = zip(*self.param_grid.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        best_params = None
        best_specificity = 0

        results = []

        # Add tqdm for progress tracking
        with tqdm(total=len(param_combinations), desc="Grid Search Progress", unit="comb") as pbar:
            for param_set in param_combinations:
                # Combine base parameters with the current parameter set
                current_params = {**self.base_params, **param_set}

                # Train model and get specificity
                avg_specificity = self.train_model(current_params)
                results.append((current_params, avg_specificity))

                # Track the best parameters
                if avg_specificity > best_specificity:
                    best_specificity = avg_specificity
                    best_params = current_params

                pbar.update(1)  # Update tqdm progress bar

        # Convert results to DataFrame for easier analysis
        results_df = pd.DataFrame(results, columns=['Params', 'Specificity'])
        results_df = results_df.sort_values(by='Specificity', ascending=False).reset_index(drop=True)

        return best_params, best_specificity, results_df



def reduction_elastic_net_lasso(ds:Datasets,
                                model_task: str,
                                output_path: pathlib.Path,
                                figsize_cm:Optional[Tuple] = (15, 7),
                                figsize_roc: Optional[Tuple] = (8, 6),
                                grid_search:Optional[bool] = False,
                                penalty:str='elasticnet',
                                loss:str='log_loss',
                                sgdclass_class_weight:Dict=None
                                ) -> tuple[
    Union[Union[None, SGDClassifier, SGDRegressor], Any], dict[str, Any], Any, DataFrame, DataFrame]:
    """
    Perform lasso for classification or linear regression problem and select significant features that are higher than
    a certain threshold in absolute value.

    If train/val splits are received, the lasso is trained/tested on these splits. Otherwise, it opens the
    Bioserenity dataset with all the desired features.

    LassoClassification outputs the feature coefficients for each class separately. Therefore, to select them
    we select those that in at least one class satisfies the threshold in absolute value.

    :param model_task: str, if to regression or classification
    :param output_path: pathlib.path, path to save the csv
    :return:
            list[str] significant feature names given by lasso
    """
    sns.set_context(context='talk')
    if not model_task in ['regression', 'classification']:
        raise ValueError(f'model must be either classification or regression')

    train_x = ds.train_X
    train_y = ds.train_y

    if ds.valid_X is None:
        test_x = ds.test_X
        test_y = ds.test_y
    else:
        test_x = ds.valid_X
        test_y = ds.valid_y
    if penalty not in ['l1', 'l2', 'elasticnet']:
        raise ValueError(f'penalty must be either l1 or l2, got {penalty}')
    if loss not in ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']:
        raise ValueError(f'loss must be either hinge or log_loss, got {loss}')

    if model_task == 'classification':
        elastic_net_model = SGDClassifier(loss=loss,
                                          penalty=penalty,
                                          l1_ratio=0.5,
                                          alpha=0.001,
                                          max_iter=2000,
                                          tol=None,
                                          random_state=42,
                                          learning_rate='adaptive',
                                          eta0=0.01,
                                          warm_start=True,
                                          class_weight={0: 1, 1: 2}# Increase weight for class 1 to increase specificity
                                          )
    else:
        elastic_net_model = SGDRegressor(loss=loss,
                                         penalty=penalty,
                                         l1_ratio=0.5,
                                         alpha=0.001,
                                         max_iter=2000,
                                         tol=None,
                                         random_state=42,
                                         learning_rate='adaptive',
                                         eta0=0.01,
                                         warm_start=True)
    best_model = None
    if grid_search:
        # GridSearchCV
        if penalty == 'elasticnet':
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 2],  # Regularization strength
                'l1_ratio': [0.1, 0.5, 0.9],  # Balance between L1 and L2 regularization
            }
        else:
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 2],  # Regularization strength
            }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        if model_task == 'classification':
            gsearch = GridSearchCV(elastic_net_model,
                                       param_grid,
                                       cv=cv,
                                       scoring='accuracy',
                                       verbose=1,
                                       n_jobs=-1)
            gsearch.fit(train_x, train_y)
            # Predict on test set with the best model
            best_model = gsearch.best_estimator_
            # pred_test = best_model.predict(test_x)
            # pred_train = best_model.predict(train_x)

            # grid_search.best_params_
            # Out[3]: {'alpha': 0.001, 'l1_ratio': 0.5}
            #
            # Classification report
            # print(classification_report(test_y, pred_test))
            # report_df = pd.DataFrame(classification_report(test_y, pred_test, output_dict=True)).transpose()
            # report_df['split'] = 'val'
            # print(report_df)
            # report_df.to_excel(f'SGDClassifierRepVal.xlsx')
            # report_df = pd.DataFrame(classification_report(train_y, pred_train, output_dict=True)).transpose()
            # report_df['split'] = 'train'
            # report_df.to_excel(f'SGDClassifierTrain.xlsx')
        else:
            # neg_mean_squared_error : closer to the mean
            # neg_mean_absolute_error: closer to the median , useful significant outliers don’t want to influence model
            # best_params = None
            # best_score = -np.inf  # Negative because we use neg_mean_squared_error
            gsearch = GridSearchCV(elastic_net_model,
                                       param_grid,
                                       cv=5,
                                       scoring='neg_mean_squared_error',
                                       verbose=1,
                                       n_jobs=-1)
            gsearch.fit(train_x, train_y)
            # Predict on test set with the best model
            best_model = gsearch.best_estimator_
            # pred_test = best_model.predict(test_x)
            # pred_train = best_model.predict(train_x)
            print("Best parameters found: ", gsearch.best_params_)
            print("Best score achieved: ", gsearch.best_score_)
            # # Use KFold cross-validation
            # kf = KFold(n_splits=5, shuffle=True, random_state=42)
            #
            # # Iterate over the parameter grid
            # for alpha in param_grid['alpha']:
            #     # Create a Lasso model with the current alpha
            #
            #     # Perform cross-validation
            #     scores = cross_val_score(elastic_net_model,
            #                              train_x,
            #                              train_y,
            #                              cv=kf,
            #                              scoring='neg_mean_squared_error',
            #                              n_jobs=-1)
            #
            #     # Calculate the mean score
            #     mean_score = np.mean(scores)
            #
            #     # If the current score is better than the best score, update the best parameters and score
            #     if mean_score > best_score:
            #         best_score = mean_score
            #         best_params = {'alpha': alpha}
            # # Output the best parameters and score
            # print("Best parameters found: ", best_params)
            # print("Best cross-validation score (negative MSE): ", best_score)

    # Fit the model
    if best_model is not None:
        elastic_net_model = best_model
        print(f'Best {penalty.capitalize()} mode: {best_model}')
    else:
        elastic_net_model.fit(train_x, train_y)

    # Predict on training and validation set
    pred_train = elastic_net_model.predict(train_x)
    pred_test = elastic_net_model.predict(test_x)
    predictions = {
        'train': pred_train,
        'test': pred_test
    }
    if model_task == 'classification':
        labels = list(set(train_y))
        train_tpfnfptn = compute_confusion_matrix_elements(train_y, pred_train, labels)
        report = classification_report(train_y, pred_train, output_dict=True)
        report_train_df = pd.DataFrame(report).transpose()
        for metric_, values_ in train_tpfnfptn.items():
            report_train_df[metric_] = np.nan
            report_train_df.loc[report_train_df.index.isin(map(str, labels)), metric_] = values_
        report_train_df['split'] = 'train'

        labels = list(set(test_y))
        report = classification_report(test_y, pred_test, output_dict=True)
        report_val_df = pd.DataFrame(report).transpose()
        test_tpfnfptn = compute_confusion_matrix_elements(test_y, pred_test, labels)
        for metric_, values_ in test_tpfnfptn.items():
            report_val_df[metric_] = np.nan
            report_val_df.loc[report_val_df.index.isin(map(str, labels)), metric_] = values_
        report_val_df['split'] = 'val'

        report_df = pd.concat([report_train_df, report_val_df], axis=0)
        report_df.reset_index(drop=False, names='classes', inplace=True)

        print(report_df)
        # confusion matrix
        unique_values = test_y.value_counts().index.values
        columns_pred = [f'Predicted Class {class_}' for class_ in unique_values]
        columns_actual = [f'Actual Class {class_}' for class_ in unique_values]
        conf_matrix_train = confusion_matrix(y_true=train_y,
                                       y_pred=pred_train)
        conf_matrix_val = confusion_matrix(y_true=test_y,
                                             y_pred=pred_test)
        fig, axes = plt.subplots(nrows=1,
                                 ncols=2,
                                 figsize=figsize_cm)
        # Plot confusion matrix for train set
        sns.heatmap(conf_matrix_train,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    cbar=False,
                    ax=axes[0])
        axes[0].set_title('Confusion Matrix - Train Set')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        # Plot confusion matrix for validation set
        sns.heatmap(conf_matrix_val,
                    annot=True,
                    fmt='d',
                    cmap='Blues',
                    cbar=False,
                    ax=axes[1])
        axes[1].set_title('Confusion Matrix - Validation Set')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        plt.tight_layout()
        if not output_path is None:
            plt.savefig(output_path.joinpath('LassoConfMatrix.png'), dpi=300)
        plt.show()
        plt.close()
        # make them as dataframe to save them as a unique csv file
        conf_matrix_train = pd.DataFrame(conf_matrix_train,
                                      columns=columns_pred,
                                      index=columns_actual)
        conf_matrix_train['split'] = 'train'
        conf_matrix_val = pd.DataFrame(conf_matrix_val,
                                         columns=columns_pred,
                                         index=columns_actual)
        conf_matrix_val['split'] = 'val'
        conf_matrix = pd.concat([conf_matrix_train, conf_matrix_val], axis=0)
        if not output_path is None:
            report_df.to_csv(output_path.joinpath('LassoReportMetrics.csv'),
                             index=False)
            conf_matrix.to_csv(output_path.joinpath('LassoConfMatrix.csv'),
                                  index=True)

        # ROC AUC curve plots
        if set(train_y.unique()).issubset({0, 1}) and train_y.nunique() == 2:
            # take the positive class probabilities
            pred_train_proba = elastic_net_model.predict_proba(train_x)[:, 1]
            pred_test_proba = elastic_net_model.predict_proba(test_x)[:, 1]
            # Calculate ROC curve and AUC for the training set
            fpr_train, tpr_train, _ = roc_curve(train_y, pred_train_proba, pos_label=1)
            roc_auc_train = auc(fpr_train, tpr_train)
            # Calculate ROC curve and AUC for the validation set
            fpr_val, tpr_val, _ = roc_curve(test_y, pred_test_proba, pos_label=1)
            roc_auc_val = auc(fpr_val, tpr_val)
            # Plot the ROC curves
            plt.figure(figsize=figsize_roc)
            plt.plot(fpr_train, tpr_train,
                     color='blue',
                     lw=2,
                     label=f'Train ROC curve (area = {roc_auc_train:.2f})')
            plt.plot(fpr_val, tpr_val,
                     color='red',
                     lw=2,
                     label=f'Validation ROC curve (area = {roc_auc_val:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve'
                      f'\nSample Size Train: {train_y.value_counts().to_dict()}'
                      f'\nSample Size Val: {test_y.value_counts().to_dict()}')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.grid(alpha=0.7)
            if not output_path is None:
                plt.savefig(output_path.joinpath('LassoRocCurve.png'), dpi=300)
            plt.show()
            plt.close()
        # now we need to select the most important features with lasso
        feature_importance = elastic_net_model.coef_
        lasso_coeff_df = pd.DataFrame(np.nan,
                                      columns=[f'class_{idx}' for idx in range(0, len(feature_importance))],
                                      index=[*train_x.columns])
        for idx_coeff in range(0, len(feature_importance)):
            lasso_coeff_df.iloc[:, idx_coeff] = feature_importance[idx_coeff]
        lasso_coeff_df.sort_values(by=[*lasso_coeff_df.columns][0],
                                   ascending=False,
                                   inplace=True)
    else:
        def evaluate_regression(y_pred: npt.NDArray, y_true: npt.NDArray) -> dict:
            return {
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "R2": r2_score(y_true, y_pred),
            }

        report = evaluate_regression(y_pred=pred_test, y_true=test_y)
        report_df = pd.DataFrame.from_dict(report, orient='index', columns=['Values'])
        if not output_path is None:
            report_df.to_csv(output_path.joinpath('lasso_reduction_metrics_val_set.csv'))

        report = evaluate_regression(y_pred=pred_train, y_true=train_y)
        report_df = pd.DataFrame.from_dict(report, orient='index', columns=['Values'])
        if not output_path is None:
            report_df.to_csv(output_path.joinpath('lasso_reduction_metrics_train_set.csv'))

        lasso_coeff_df = pd.DataFrame({'Feature': [*train_x.columns],
                                       'Importance': elastic_net_model.coef_})
    if not output_path is None:
        lasso_coeff_df.to_csv(output_path.joinpath('lasso_reduction_coeff_list.csv'),
                              index=True)
    # get the rows where at least one of the columns is higher than 0.09 in absolute value
    filtered_df = lasso_coeff_df[(lasso_coeff_df.abs() > 0.09).any(axis=1)]
    if not output_path is None:
        filtered_df.to_csv(output_path.joinpath('lasso_reduction_coeff_list_filtered.csv'),
                           index=True)

    return elastic_net_model, predictions, lasso_coeff_df.index.to_list(), lasso_coeff_df, report_df


def compute_confusion_matrix_elements(y_true, y_pred, labels) -> dict[str, float]:
    """
    Compute the TP, FN, FP, and TN when mutliclass problem
    :param y_true:
    :param y_pred:
    :param labels:
    :return:
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp = cm.diagonal()
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = cm.sum() - (fp + fn + tp)
    return {'TP':tp, 'FN':fn, 'FP':fp, 'TN':tn}

def plot_lasso_coefficients(lasso_coeff_df: pd.DataFrame,
                            n_top: int = 5,
                            n_bottom: int = 5,
                            class_column: Optional[Union[str, None]] = 'class_0',
                            output_path: Optional[pathlib.Path] = None,
                            figsize: Optional[Tuple[int, int]] = (12, 8)) -> None:
    """
    Plot the top and bottom Lasso coefficients for specified class(es).

    Parameters:
    - lasso_coeff_df (pd.DataFrame): DataFrame containing Lasso coefficients, with feature names as the index.
    - n_top (int): Number of top features to display. Default is 5.
    - n_bottom (int): Number of bottom features to display. Default is 5.
    - class_column (str or None): The column name in lasso_coeff_df that contains the coefficients for a specific class.
                                  If None, plot for all classes.
    - output_path (pathlib.Path or None): Path to save the output plot. Default is None.
    - figsize (tuple): Figure size. Default is (12, 8).

    Returns:
    - None
    """
    if class_column is None:
        class_columns = lasso_coeff_df.columns
    else:
        class_columns = [class_column]

    num_classes = len(class_columns)
    fig, axes = plt.subplots(num_classes, 1, figsize=(figsize[0], figsize[1] * num_classes), sharex=True)

    if num_classes == 1:
        axes = [axes]

    for i, class_col in enumerate(class_columns):
        top_features = lasso_coeff_df[class_col].sort_values(ascending=False).head(n_top).reset_index()
        top_features['order'] = f'Top {n_top}'

        bottom_features = lasso_coeff_df[class_col].sort_values(ascending=True).head(n_bottom).reset_index()
        bottom_features['order'] = f'Bottom {n_bottom}'

        combined_features = pd.concat([top_features, bottom_features])
        combined_features.rename(columns={class_col: 'Coefficient', 'index': 'Feature'}, inplace=True)

        top_color = sns.color_palette("viridis", as_cmap=True)(0.7)
        bottom_color = sns.color_palette("rocket", as_cmap=True)(0.3)
        palette = {f'Top {n_top}': top_color, f'Bottom {n_bottom}': bottom_color}

        sns.barplot(data=combined_features,
                    x='Coefficient',
                    y='Feature',
                    hue='order',
                    dodge=False,
                    palette=palette,
                    edgecolor='w',
                    ax=axes[i])

        axes[i].set_title(f'Top {n_top} and Bottom {n_bottom} Lasso Coefficients for {class_col}', fontsize=16)
        axes[i].set_xlabel('Coefficient Value', fontsize=14)
        axes[i].set_ylabel('Feature', fontsize=14)
        axes[i].grid(axis='x', linestyle='--', alpha=0.7)
        axes[i].legend(title='Order')

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path.joinpath(f'LassoCoeff.png'), dpi=300)
    plt.show()


def plot_lasso_coefficients_all(lasso_coeff_df: pd.DataFrame,
                                output_path: Optional[pathlib.Path] = None,
                                figsize: Optional[Tuple[int, int]] = (12, 14),
                                orientation: str = 'vertical') -> None:
    """
    Plot all the class coefficients of all the classes in a single figure with a unique color for each class.

    Parameters:
    - lasso_coeff_df (pd.DataFrame): DataFrame containing Lasso coefficients, with feature names as the index.
    - output_path (pathlib.Path or None): Path to save the output plot. Default is None.
    - figsize (tuple): Figure size. Default is (12, 14).
    - orientation (str): Orientation of the plot, either 'vertical' or 'horizontal'. Default is 'vertical'.

    Returns:
    - None
    """
    # Filter out features with all zero coefficients
    lasso_coeff_df_filtered = lasso_coeff_df.loc[
        ~(lasso_coeff_df[lasso_coeff_df.columns] == 0).all(axis=1)]

    # Melt the DataFrame for easier plotting with seaborn
    lasso_coeff_df_filtered.reset_index(drop=False, inplace=True, names=['Features'])
    lasso_coeff_df_melted = lasso_coeff_df_filtered.melt(id_vars='Features',
                                                         var_name='Class',
                                                         value_name='Coefficient')

    # Define color mapping for each unique class using the Blues palette
    unique_classes = lasso_coeff_df_melted['Class'].unique()
    colormap = sns.color_palette("Blues", n_colors=len(unique_classes))
    color_mapping = {cls: colormap[i] for i, cls in enumerate(unique_classes)}

    # Set up plot orientation
    plt.figure(figsize=figsize)
    if orientation == 'horizontal':
        sns.barplot(x='Coefficient', y='Features', hue='Class', data=lasso_coeff_df_melted, palette=color_mapping)
        plt.axvline(0, color='red', linestyle='--', linewidth=1)  # Add a vertical red line at zero
        plt.xlabel('Coefficient Value', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.grid(True, axis='y', alpha=0.7)
    else:  # Vertical orientation
        sns.barplot(x='Features', y='Coefficient', hue='Class', data=lasso_coeff_df_melted, palette=color_mapping)
        plt.axhline(0, color='red', linestyle='--', linewidth=1)  # Add a horizontal red line at zero
        plt.ylabel('Coefficient Value', fontsize=14)
        plt.xlabel('Feature', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(True, axis='x', alpha=0.7)

    # Plot settings
    plt.title('Non-Zero Lasso Coefficients for All Classes', fontsize=16)
    plt.legend(title='Class')
    plt.tight_layout()

    # Save the figure if output path is provided
    if output_path:
        plt.savefig(output_path.joinpath('LassoCoeffAllClasses.png'), dpi=300)
    plt.show()

def compute_real_world_metrics(
        df: Union[pd.DataFrame, pathlib.Path],
        prevalence: float = 0.0003,
        output_path: pathlib.Path = None
) -> pd.DataFrame:
    """
    Using the metrics dataframe created with the elasticnet function, compute real world metrics using an estimated
    prevalence in the population.

    The input is expected to be of the form:
                classes  precision    recall  f1-score  ...    FN    FP     TN  split
        0             0   0.945187  0.976519  0.960598  ...  17.0  41.0  403.0  train
        1             1   0.959524  0.907658  0.932870  ...  41.0  17.0  707.0  train
        2      accuracy   0.950342  0.950342  0.950342  ...   NaN   NaN    NaN  train
        3     macro avg   0.952355  0.942088  0.946734  ...   NaN   NaN    NaN  train
        4  weighted avg   0.950637  0.950342  0.950058  ...   NaN   NaN    NaN  train
        5             0   0.952128  0.983516  0.967568  ...   3.0   9.0  102.0    val
        6             1   0.971429  0.918919  0.944444  ...   9.0   3.0  179.0    val
        7      accuracy   0.959044  0.959044  0.959044  ...   NaN   NaN    NaN    val
        8     macro avg   0.961778  0.951218  0.956006  ...   NaN   NaN    NaN    val
        9  weighted avg   0.959440  0.959044  0.958808  ...   NaN   NaN    NaN    val

    :param df: DataFrame or path to the metrics .csv file
    :param prevalence: Prevalence in the population of the given class
    :param output_path: Path to save the resulting metrics dataframe
    :return: The modified dataframe with PPV for each class and split
    """
    def calculate_specificity(row) -> float:
        if row['TN'] + row['FP'] == 0:
            return 0  # Avoid division by zero
        return row['TN'] / (row['TN'] + row['FP'])

    def calculate_ppv(row) -> float:
        prevalence = 0.003
        ppv = (row[specificity_col] * prevalence) / (
                row[specificity_col] * prevalence + (1 - specificity) * (1 - prevalence))
        return ppv

    if isinstance(df, pathlib.Path):
        df = pd.read_csv(df)

    split_col = [col for col in df if 'split' in col.lower()][0]
    specificity_col = [col for col in df if col.lower() in ['recall', 'specificity']]
    if len(specificity_col) == 0:
        df['specificity'] = df.apply(calculate_specificity, axis=1)
        specificity_col = 'specificity'

    # Create a list to store the PPV for each class
    ppv_list = []
    # Iterate over each split in the dataframe
    for split_name in df[split_col].unique():
        df_set = df[df[split_col] == split_name].copy()

        # Calculate PPV for each class in the split
        for _, row in df_set.iterrows():
            # Check if the row corresponds to a class and not an aggregate
            if pd.notna(row['TN']) and pd.notna(row['FP']):
                specificity = row['TN'] / (row['TN'] + row['FP'])
                try:
                    ppv = (row[specificity_col] * prevalence) / (
                                row[specificity_col] * prevalence + (1 - specificity) * (1 - prevalence))
                except ZeroDivisionError:
                    ppv = np.nan
                ppv_list.append(ppv)
            else:
                ppv_list.append(None)

    # Add the PPV to the dataframe
    df['PPV'] = ppv_list

    # Save the results dataframe if output_path is provided
    if output_path:
        df.to_csv(output_path.joinpath('real_world_metrics.csv'), index=False)

    return df


def identify_misclassified_observations(ds: Datasets,
                                        predictions: dict[str, np.ndarray],
                                        model_task: str,
                                        threshold: float = 0.5) -> pd.DataFrame:
    """
    Identify misclassified or outlier observations.

    For classification, identifies observations where predicted class differs from the true class.
    For regression, identifies observations where absolute prediction error exceeds a threshold.

    :param ds: Datasets object containing the true labels.
    :param predictions: Dictionary containing predictions for 'train' and 'test' sets.
    :param model_task: str, either 'regression' or 'classification'.
    :param threshold: float, threshold for identifying outliers in regression.
    :return: DataFrame containing misclassified or outlier observations.
    """
    if model_task not in ['regression', 'classification']:
        raise ValueError("model_task must be either 'regression' or 'classification'")

    results = []
    for split, y_pred in predictions.items():
        if y_pred is None:
            continue
        if split == 'train':
            y_true = ds.train_y
            x_data = ds.train_X
        elif split == 'test':
            y_true = ds.test_y  # ds.valid_y if ds.valid_X is not None else ds.test_y
            x_data = ds.test_X  # ds.valid_X if ds.valid_X is not None else ds.test_X
        else:
            y_true = ds.valid_y
            x_data = ds.valid_X

        if model_task == 'classification':
            misclassified_idx = np.where(y_pred != y_true)[0]
            errors = y_pred[misclassified_idx]  # Predicted labels for misclassified samples
            true_labels = y_true.iloc[misclassified_idx].values
            misclassified_data = x_data.iloc[misclassified_idx]
            misclassified_df = pd.DataFrame(misclassified_data)
            misclassified_df['True Label'] = true_labels
            misclassified_df['Predicted Label'] = errors

        else:  # regression
            residuals = y_pred - y_true
            outlier_idx = np.where(np.abs(residuals) > threshold)[0]
            true_values = y_true.iloc[outlier_idx].values
            predicted_values = y_pred[outlier_idx]
            misclassified_data = x_data.iloc[outlier_idx]
            misclassified_df = pd.DataFrame(misclassified_data)
            misclassified_df['True Value'] = true_values
            misclassified_df['Predicted Value'] = predicted_values
            misclassified_df['Residual'] = residuals[outlier_idx]

        misclassified_df['Split'] = split
        results.append(misclassified_df)

    misclassified_observations = pd.concat(results, axis=0) # .reset_index(drop=True)
    return misclassified_observations



if __name__ == '__main__':
    pass
