import pathlib
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor,XGBClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from typing import Optional, List, Dict, Tuple, Union
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from tqdm.auto import tqdm
import math
from sklearn.utils import resample
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_error, median_absolute_error,
    max_error, explained_variance_score,
    confusion_matrix, roc_curve, auc,
    accuracy_score, recall_score, precision_score, f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    cohen_kappa_score,
)
from typing import List, Tuple, Dict, Any
import scipy.stats as st
from matplotlib.gridspec import GridSpec
import pathlib
import re
import pathlib
from typing import  Optional, List, Union, Tuple
import json
import pickle
from scipy.special import gammaln

from fontTools.tfmLib import PASSTHROUGH

from config.config import config, metrics_psg, encoding, sections
import statsmodels.api as sm
from src.regression_class_fcnn import N_TRIALS

from tabulate import tabulate
from scipy.stats import gaussian_kde


def _resample_minority_classes(df: pd.DataFrame,
                               target_col: str,
                               stratify_col: str,
                               random_state: int = 42):
    """
    Resample the minority classes in a DataFrame to match the size of the majority class
    while preserving the original class distributions.

    This function performs upsampling of minority classes in the specified column until
    they have the same number of samples as the majority class. This ensures balanced
    classes and mitigates issues caused by imbalanced datasets during data analysis or
    model training.

    :param df: The input pandas DataFrame containing the data to be resampled.
    :type df: pd.DataFrame
    :param target_col: The name of the target column for classification (used for reference).
    :type target_col: str
    :param stratify_col: The column name used to define the classes for resampling.
    :type stratify_col: str
    :param random_state: Random seed for reproducibility of the resampling operation.
    :type random_state: int
    :return: A resampled pandas DataFrame with balanced class sizes across all values
             in the 'stratify_col'.
    :rtype: pd.DataFrame
    """
    classes = df[stratify_col].unique()
    majority_class = df[stratify_col].value_counts().idxmax()
    majority_df = df[df[stratify_col] == majority_class]
    resampled_parts = [majority_df]

    for cls in classes:
        if cls != majority_class:
            minority_df = df[df[stratify_col] == cls]
            upsampled = resample(minority_df,
                                 replace=True,
                                 n_samples=len(majority_df),
                                 random_state=random_state)
            resampled_parts.append(upsampled)

    df_resampled = pd.concat(resampled_parts)
    df_resampled = df_resampled.sample(frac=1, random_state=random_state) # .reset_index(drop=True)
    return df_resampled

# %% Feature contrains

def create_feature_constraints(
    sections: List[str],
    df: pd.DataFrame
) -> Tuple[List[List[str]], List[str]]:
    """
    Build XGBoost-style interaction_constraints from:
      1. A demographics group (["age","bmi","gender","race"])
      2. One group per prefix in `sections`.

    Only columns actually in `df` are used for groups.
    Returns both:
      - groups: List of feature-name lists (for constraints)
      - missing: List of items (dem names or prefixes) that had no columns in `df`.

    Parameters
    ----------
    sections : List[str]
        Prefixes defining each feature-group (e.g. ["ep", "mh_", ...]).
    df : pd.DataFrame
        DataFrame whose columns we’ll partition into groups.

    Returns
    -------
    Tuple[List[List[str]], List[str]]
    """
    from typing import Any, List
    def flatten(nested: List[Any]) -> List[Any]:
        """
        Recursively flatten a nested list.

        Parameters
        ----------
        nested : List[Any]
            A list which can contain other lists (to any depth).

        Returns
        -------
        List[Any]
            A single, flat list with all non-list elements from `nested`.
        """
        flat: List[Any] = []
        for item in nested:
            if isinstance(item, list):
                flat.extend(flatten(item))  # recurse into sublist
            else:
                flat.append(item)
        return flat

    cols = set(df.columns)

    # 1) Demographics
    dem_cols = ["dem_age", "dem_bmi", "dem_gender", "dem_race"]
    present_dem = [c for c in dem_cols if c in cols]
    missing_dem = [c for c in dem_cols if c not in cols]

    groups: List[List[str]] = []
    if present_dem:
        groups.append(present_dem)

    # 2) Prefix groups
    seen = set(present_dem)
    missing_sections: List[str] = []
    for pref in sections:
        matched = [c for c in cols if c.startswith(pref) and c not in seen]
        if matched:
            groups.append(matched)
            seen.update(matched)
        else:
            missing_sections.append(pref)

    # Combine all “requested but not found” items
    missing = missing_dem + missing_sections

    groups_flat = flatten(groups)

    missing_cols = [col for col in groups_flat if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns in groups_flat are not in df.columns: {missing_cols}")

    return groups, missing


# %% Model training

# def train_xgb_regressor_with_optuna_and_cv(
#     df: pd.DataFrame,
#     feature_cols: list,
#     target_col: str,
#     optimization: bool = True,
#     n_trials: int = 50,
#     cv_folds: int = 5,
#     test_size: float = 0.2,
#     random_state: int = 42,
#     use_gpu: bool = True,
#     stratify_col: str = None
# ) -> dict:
#     """
#     Trains an XGBoost regressor with optional Optuna tuning and k-fold CV,
#     collecting per-fold train & validation metrics (mean ± std) and feature importances.
#     Returns a dict with: best_params, cv_metrics, test_metrics, feature_importance_df, model, predictions.
#     """
#     # 0) Drop stratify_col from features if present
#     if stratify_col and stratify_col in feature_cols:
#         feature_cols = [c for c in feature_cols if c != stratify_col]
#
#     # 1) Train / test split
#     X = df[feature_cols]
#     y = df[target_col]
#     stratify = df[stratify_col] if stratify_col else None
#
#     split_args = dict(test_size=test_size, random_state=random_state)
#     if stratify_col:
#         split_args['stratify'] = stratify.values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, **split_args)
#     strat_train = stratify.loc[y_train.index].values if stratify_col else None
#
#     # 2) GPU helper
#     def gpu_params(p: dict) -> dict:
#         if use_gpu:
#             p.update(tree_method='hist', device='cuda')
#         return p
#
#     # 3) Optuna tuning
#     if optimization:
#         def objective(trial):
#             params = dict(
#                 num_boost_round = trial.suggest_int('num_boost_round', 50, 500),
#                 max_depth    = trial.suggest_int('max_depth', 2, 10),
#                 learning_rate= trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
#                 subsample    = trial.suggest_float('subsample', 0.5, 1.0),
#                 colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
#                 objective    = 'reg:squarederror',
#                 random_state = random_state
#             )
#             params = gpu_params(params)
#
#             splitter = (
#                 StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
#                 if stratify_col else
#                 KFold(n_splits=3, shuffle=True, random_state=random_state)
#             )
#             rmses = []
#             X_arr = X_train.values; y_arr = y_train.values
#             for tr_idx, val_idx in splitter.split(X_arr, strat_train):
#                 dtr = xgb.DMatrix(X_arr[tr_idx], label=y_arr[tr_idx], missing=np.nan)
#                 dval= xgb.DMatrix(X_arr[val_idx], label=y_arr[val_idx], missing=np.nan)
#                 bst = xgb.train(params, dtr, num_boost_round=params['num_boost_round'],
#                                 evals=[(dtr,'train'),(dval,'valid')],
#                                 early_stopping_rounds=10, verbose_eval=False)
#                 preds = bst.predict(dval)
#                 rmses.append(np.sqrt(mean_squared_error(y_arr[val_idx], preds)))
#             return np.mean(rmses)
#
#         study = optuna.create_study(direction='minimize')
#         study.optimize(objective, n_trials=n_trials)
#         best_params = gpu_params({
#             **study.best_params,
#             'objective': 'reg:squarederror',
#             'random_state': random_state
#         })
#     else:
#         # default fallback
#         best_params = gpu_params({
#             'num_boost_round': 266,
#             'max_depth': 4,
#             'learning_rate': 0.00527,
#             'subsample': 0.58,
#             'colsample_bytree': 0.997,
#             'objective': 'reg:squarederror',
#             'random_state': random_state
#         })
#
#     # 4) Helper to compute fold metrics
#     def compute_fold_metrics(y_true, y_pred, n_feats):
#         r2       = r2_score(y_true, y_pred)
#         mse      = mean_squared_error(y_true, y_pred)
#         rmse     = np.sqrt(mse)
#         mae      = mean_absolute_error(y_true, y_pred)
#         medae    = median_absolute_error(y_true, y_pred)
#         mape     = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
#         maxerr   = max_error(y_true, y_pred)
#         expl_var = explained_variance_score(y_true, y_pred)
#         r2_adj   = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_feats - 1)
#         return dict(r2=r2, rmse=rmse, mae=mae, medae=medae,
#                     mape=mape, maxerr=maxerr, expl_var=expl_var, r2_adj=r2_adj)
#
#     # 5) CV on train set
#     splitter = (
#         StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
#         if stratify_col else
#         KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
#     )
#     metric_names   = ['r2','rmse','mae','medae','mape','maxerr','expl_var','r2_adj']
#     metrics        = {s: {m: [] for m in metric_names} for s in ('train','valid')}
#     importance_types= ['weight','gain','cover']
#     imp_lists      = {t: [] for t in importance_types}
#     n_feats        = len(feature_cols)
#
#     Xtr_arr = X_train.values; ytr_arr = y_train.values
#
#     for tr_idx, val_idx in splitter.split(Xtr_arr, strat_train):
#         # build DMatrices
#         dtr = xgb.DMatrix(Xtr_arr[tr_idx],
#                           label=ytr_arr[tr_idx],
#                           missing=np.nan)
#         dval= xgb.DMatrix(Xtr_arr[val_idx],
#                           label=ytr_arr[val_idx],
#                           missing=np.nan)
#
#         bst = xgb.train(
#             best_params, dtr,
#             num_boost_round = best_params['num_boost_round'],
#             evals=[(dtr,'train'),(dval,'valid')],
#             early_stopping_rounds=10,
#             verbose_eval=False
#         )
#
#         # predictions
#         y_pred_tr  = bst.predict(dtr)
#         y_pred_val = bst.predict(dval)
#
#         # compute + store
#         tr_m = compute_fold_metrics(ytr_arr[tr_idx], y_pred_tr, n_feats)
#         vl_m = compute_fold_metrics(ytr_arr[val_idx], y_pred_val, n_feats)
#         for mn in metric_names:
#             metrics['train'][mn].append(tr_m[mn])
#             metrics['valid'][mn].append(vl_m[mn])
#
#         # feature importances
#         for t in importance_types:
#             scores = bst.get_score(importance_type=t)
#             arr = np.array([scores.get(f"f{i}", 0.0) for i in range(n_feats)])
#             imp_lists[t].append(arr)
#
#     # 6) Aggregate importances
#     imp_stats = {}
#     for t, arrs in imp_lists.items():
#         mat = np.vstack(arrs)
#         imp_stats[f"{t}_mean"] = mat.mean(axis=0)
#         imp_stats[f"{t}_std"]  = mat.std(axis=0)
#
#     # 7) Build flattened CV metrics (mean ± std)
#     cv_metrics = {}
#     for split in ('train','valid'):
#         for mn in metric_names:
#             cv_metrics[f"{split}_{mn}_mean"] = np.mean(metrics[split][mn]).astype('float16').item()
#             cv_metrics[f"{split}_{mn}_std"]  = np.std(metrics[split][mn]).astype('float16').item()
#
#     # 8) Final training on full train set & test evaluation
#     dtrain_full = xgb.DMatrix(Xtr_arr, label=ytr_arr, missing=np.nan)
#     dtest       = xgb.DMatrix(X_test.values, label=y_test.values, missing=np.nan)
#     final_bst   = xgb.train(
#         best_params, dtrain_full,
#         num_boost_round=best_params['num_boost_round'],
#         verbose_eval=False
#     )
#     y_pred_test = final_bst.predict(dtest)
#     test_metrics = {
#         'r2':   r2_score(y_test, y_pred_test),
#         'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
#     }
#
#     # 9) Package feature‐importance DF
#     fi_df = pd.DataFrame({
#         'feature': feature_cols,
#         **imp_stats
#     }).sort_values('gain_mean', ascending=False).reset_index(drop=True)
#
#     return {
#         'best_params': best_params,
#         'cv_metrics':  cv_metrics,
#         'test_metrics': test_metrics,
#         'feature_importance_df': fi_df,
#         'model': final_bst,
#         'predictions': {
#             'test': {'y_true': y_test, 'y_pred': y_pred_test}
#         }
#     }

def _invs_prob_weight(y:np.ndarray) -> np.ndarray:
    """Apply inverse probability weight to each sample."""
    kde = gaussian_kde(y)
    weights = 1.0 / kde(y)
    return weights / np.mean(weights)

# %% training updated
xgb_classification_objectives = [
    # Binary classification
    "binary:logistic",
    "binary:logitraw",

    # Multiclass classification
    "multi:softprob",
    "multi:softmax",

    # Ranking objectives (also applicable in certain classification settings)
    "rank:pairwise",
    "rank:ndcg",
    "rank:map"
]

def train_xgb_collect(
    data: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    in_params: Optional[Dict] = None,
    optimization: bool = True,
    n_trials: int = 50,
    cv_folds: Union[int, bool] = 5,
    test_size: float = 0.2,
    val_size: Optional[float] = None,
    random_state: int = 42,
    use_gpu: bool = True,
    stratify_col: str = None,
    model_path:pathlib.Path = None,
    save_cv_models: bool = False,
    show_loss_curve: bool = False,
    resample: Optional[bool] = False,
    invs_prob_weight: Optional[bool] = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, xgb.Booster]:
    """
    Train an XGBoost regressor with optional Optuna tuning, early stopping,
    and k-fold cross-validation, then evaluate on a held-out test set.

    Args:
        data (pd.DataFrame): Full dataset containing both features and target.
        feature_cols (list[str]): Names of columns to use as input features.
        target_col (str): Name of the column to predict.
        in_params (dict | None): Extra XGBoost parameters to include (e.g., objective, eval_metric).
        optimization (bool): If True, run Optuna hyperparameter tuning.
        n_trials (int): Number of Optuna trials when `optimization` is True.
        cv_folds (int | bool): Number of folds for CV; set to False or 1 to skip CV.
        test_size (float): Fraction of data reserved for final held-out test evaluation.
        val_size (float | None): Fraction of training data to reserve for validation during tuning.
        random_state (int): Seed for reproducible splits and model initialization.
        use_gpu (bool): If True, enable GPU acceleration (`tree_method='hist'`, `device='cuda'`).
        stratify_col (str | None): Column name to use for stratified splits; None for random splits.
        model_path (pathlib.Path | None): Directory where models, CSVs, and plots will be saved.
        save_cv_models (bool): If True, persist each fold’s model to `model_path`.
        show_loss_curve (bool): If True, plots & saves the train vs. validation loss curve.

    Returns:
        fi_df (pd.DataFrame): Feature-importance table (mean & std across folds).
        preds_df (pd.DataFrame): DataFrame of per-fold and test set predictions & true values.
        best_params (dict): Best hyperparameter configuration (from Optuna or defaults).
        final_model (xgb.Booster): Booster trained on the full training set.
    """
    def _gpu(p: Dict) -> Dict:
        if use_gpu:
            p.update(tree_method='hist', device='cuda')
        return p
    # define path to save the models
    if model_path:
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
        base = model_path.stem
        suffix = ".json"

    df = data.copy()
    print(f'Initial dimension of dataset: {df.shape}')
    # drop any nan columns from the target and the stratification column
    df = df[~data[target_col].isna()]
    if stratify_col:
        df = df[~data[stratify_col].isna()]
    print(f'After dropping target and stratification: {df.shape}')

    # 1) Filter out any features not present in df
    removed_features = [c for c in feature_cols if c not in df.columns]
    feature_cols = [c for c in feature_cols if c in df.columns]

    # remove stratify_col from features if needed
    if stratify_col and stratify_col in feature_cols:
        feature_cols = [c for c in feature_cols if not c in [stratify_col, target_col]]



    # 2) Train/test split
    X, y = df[feature_cols], df[target_col]
    stratify = df[stratify_col] if stratify_col else None
    split_args = {'test_size': test_size, 'random_state': random_state}
    if stratify_col:
        split_args['stratify'] = stratify.values

    # get the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_args)

    if resample:
        df_to_resample = (
            X_train
            .copy()
            .assign(**{target_col: y_train})
            .merge(df[[stratify_col]], left_index=True, right_index=True, how='left')
        )
        # df_to_resample = df_to_resample.reset_index(drop=False, inplace=False, names='index_source')
        assert X_train.shape[0] == df_to_resample.shape[0]
        # apply a re-sampling method
        df_to_resample = _resample_minority_classes(df=df_to_resample,
                                        target_col=target_col,
                                        stratify_col=stratify_col,
                                        random_state=random_state)
        # X_train and y_train have duplicate indexes
        X_train, y_train = df_to_resample[feature_cols], df_to_resample[target_col]
        print(f'Resampled Minority Classes: '
              f'\t{df[stratify_col].value_counts()}'
              f'\rX train dim: {X_train.shape[0]}')

    # 3) hyperparameter tuning for optimziation from the train set
    if optimization and val_size is not None:
        tune_args  = {'test_size': val_size, 'random_state': random_state}
        if stratify_col:
            tune_args['stratify'] = df.loc[y_train.index, stratify_col].values
        # remove a tuning split from the training sample
        X_train, X_tune, y_train, y_tune = train_test_split(
            X_train, y_train, **tune_args
        )
        # separate the training sample into a train and test set
        tune_args_opt = {'test_size': 0.10, 'random_state': random_state}
        X_tune_train, X_tune_test, y_tune_train, y_tune_train_test = train_test_split(
            X_tune, y_tune, **tune_args_opt
        )

    # get a list of features
    col_features = list(X_train.columns)

    # objective baseline
    obj = in_params.get('objective', 'reg:squarederror') if in_params else 'reg:squarederror'
    eval_metric = in_params.get('eval_metric', 'rmse') if in_params else 'rmse'
    # tune with Optuna
    if optimization:

        class ThresholdEarlyStopping:
            """
            Optuna callback to early stop hyperparameter optimization when there is no
            *significant* improvement in the objective value for a given number of trials.

            This is useful when your optimization process converges early and continues
            evaluating similar parameter sets with marginal gains. For example, if your
            validation RMSE is stuck around 1.0800±0.0005 for many trials, you can save
            time by stopping after `patience` trials without an improvement greater than `min_delta`.

            Parameters:
                patience (int): Number of consecutive trials without significant improvement
                                allowed before stopping the study.
                min_delta (float): Minimum change in best value to be considered an improvement.

            Example usage:
                stopper = ThresholdEarlyStopping(patience=20, min_delta=1e-4)
                study.optimize(objective, n_trials=100, callbacks=[stopper])
            """

            def __init__(self, patience: int = 20, min_delta: float = 1e-4):
                self.patience = patience
                self.min_delta = min_delta
                self.best_score = None
                self.counter = 0

            def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
                current_best = study.best_value

                if self.best_score is None:
                    self.best_score = current_best
                    return

                if self.best_score - current_best > self.min_delta:
                    self.best_score = current_best
                    self.counter = 0
                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    print(f"Early stopping: No significant improvement greater than "
                          f"{self.min_delta}) in the last {self.patience} trials.")
                    study.stop()

        def objective(trial):
            p = {
                # "num_boost_round": trial.suggest_int("num_boost_round", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "lambda": trial.suggest_float("lambda", 1.1, 10.0),
                "gamma": trial.suggest_float("gamma",0, 10.0),
                "alpha": trial.suggest_float("alpha", 1.1, 10.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "objective": obj,
                "eval_metric": eval_metric ,
                "random_state": random_state,
            }

            # Include num_class only for multi-class objectives
            if in_params and in_params.get('num_class'):
                p["num_class"] = in_params["num_class"]

            p = _gpu(p)
            dtr = xgb.DMatrix(X_tune_train,
                              label=y_tune_train,
                              missing=np.nan,
                              weight=_invs_prob_weight(y_tune_train) if invs_prob_weight else None)
            dval= xgb.DMatrix(X_tune_test,
                              label=y_tune_train_test,
                              missing=np.nan,
                              weight=_invs_prob_weight(y_tune_train_test) if invs_prob_weight else None
                              )
            # num_boost_round_p =  p["num_boost_round"]
            bst = xgb.train(
                {key: val for key, val in p.items() if key != 'num_boost_round'},
                dtr,
                num_boost_round=2000,
                evals=[(dtr,"train"),(dval,"valid")],
                early_stopping_rounds=100,
                verbose_eval=False,
            )
            pred = bst.predict(dval)
            # === Score: Quadratic Weighted Kappa ===
            def _poisson_nll(y_true, y_pred):
                eps = 1e-9  # avoid log(0)
                y_pred = np.clip(y_pred, eps, None)
                return np.mean(y_pred - y_true * np.log(y_pred) + gammaln(y_true + 1))

            def compute_score(pred, y_true, p):
                if p["objective"] == "multi:softprob":
                    pred_labels = np.argmax(pred, axis=1)
                    return -cohen_kappa_score(y_true, pred_labels, weights="quadratic")
                elif p["objective"] == "multi:softmax":
                    return -cohen_kappa_score(y_true, pred.astype(int), weights="quadratic")
                elif p["objective"] in ["binary:logistic", "binary:logitraw"]:
                    pred_labels = (pred > 0.5).astype(int)
                    return log_loss(y_true, pred_labels)
                elif p["objective"].startswith("reg:"):
                    return np.sqrt(mean_squared_error(y_true, pred))
                elif p["objective"] == "reg:poisson":
                    return _poisson_nll(y_true, pred)
                else:
                    raise ValueError(f"No scoring rule defined for objective '{p['objective']}'")

            return compute_score(pred, y_tune_train_test, p)

        threshold_early_stopper = ThresholdEarlyStopping(patience=20, min_delta=1e-4)


        study = optuna.create_study(direction="minimize")
        study.optimize(objective,
                       n_trials=n_trials,
                       callbacks=[threshold_early_stopper])
        best = study.best_params
        best_params = {
            **best,
            "objective": obj,
            "eval_metric": eval_metric,
            "random_state": random_state
        }
        best_params = _gpu(best_params)
    else:
        best_params = _gpu({
            'num_boost_round': 1000,
            'max_depth': 4,
            'learning_rate': 0.00527,
            # 'subsample':    0.58,
            'objective': obj,
            'eval_metric': eval_metric,
            'random_state': random_state
        })

    if in_params:
        # only inject those in_params keys that Optuna did *not* tune
        for k, v in in_params.items():
            if k not in best_params:
                best_params[k] = v
        # or we can write in one line:  best_params = {**in_params, **best_params}

        best_params.update(in_params)
        if 'interaction_constraints' in in_params:
            best_params['interaction_constraints'] = _interactions_constraint_to_indexes(features=col_features,
                                               interaction_constraints=best_params['interaction_constraints'])

    # ─── Diagnostic printout ────────────────────────────────────────────────
    print("=== XGBoost parameters ===")
    for k, v in best_params.items():
        print(f"  {k:20s} : {v}")
    print("\n=== Dataset splits ===")
    # full training set (for CV)
    print(f"  X_train   : {X_train.shape}   y_train   : {y_train.shape}")
    # tuning splits (only if you created them)
    if optimization and val_size is not None:
        print(f"  X_tune_train     : {X_tune_train.shape}   y_tune_train        : {y_tune_train.shape}")
        print(f"  X_tune_test      : {X_tune_test.shape}   y_tune_train_test   : {y_tune_train_test.shape}")
    # held‐out test set
    print(f"  X_test    : {X_test.shape}   y_test    : {y_test.shape}")
    print("──────────────────────────────────────────────────────────────────────\n")

    if 'num_boost_round' in best_params:
        num_boost_round = best_params['num_boost_round']
        del best_params['num_boost_round']
    else:
        num_boost_round = 3000

    if 'early_stopping_rounds' in best_params:
        early_stopping_rounds = best_params['early_stopping_rounds']
        del best_params['early_stopping_rounds']
    else:
        early_stopping_rounds = 500

    # initialize preds container
    n_feats = len(feature_cols)
    importance_types = ['weight', 'gain', 'cover']

    preds_data = {}
    if cv_folds:
        # prepare CV
        splitter = (
            # RepeatedStratifiedKFold(n_splits=cv_folds,
            #                 # shuffle=True,
            #                 random_state=random_state)
            StratifiedKFold(n_splits=cv_folds,
            shuffle=True,
            random_state=random_state)
            if stratify_col else
            KFold(n_splits=cv_folds,
                  shuffle=True,
                  random_state=random_state)
        )
        # choose correct split iterator
        if stratify_col:
            strat_vals = df.loc[y_train.index, stratify_col].values
            cv_iterator = splitter.split(X_train, strat_vals)
        else:
            cv_iterator = splitter.split(X_train)

        for fold, (tr_ix, val_ix) in enumerate(
                tqdm(cv_iterator, total=cv_folds, desc="CV folds"), start=1):

            X_tr, y_tr = X_train.iloc[tr_ix], y_train.iloc[tr_ix]
            X_val, y_val = X_train.iloc[val_ix],   y_train.iloc[val_ix]

            dtr = xgb.DMatrix(X_tr,
                              label=y_tr,
                              missing=np.nan,
                              feature_names=feature_cols)
            dval= xgb.DMatrix(X_val,
                              label=y_val,
                              missing=np.nan,
                              feature_names=feature_cols)
            bst = xgb.train(
                best_params, dtr,
                num_boost_round=num_boost_round,
                evals=[(dtr,'train'),(dval,'valid')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=200,

            )
            # indices
            preds_data[f'train_fold_{fold}_index'] = list(y_tr.index)
            preds_data[f'train_fold_{fold}_true']  = y_tr.to_list()
            preds_data[f'train_fold_{fold}_pred']  = list(bst.predict(dtr))
            preds_data[f'val_fold_{fold}_index']   = list(y_val.index)
            preds_data[f'val_fold_{fold}_true']   = y_val.to_list()
            preds_data[f'val_fold_{fold}_pred']   = list(bst.predict(dval))
            # save model
            if model_path and save_cv_models:
                name = f"{base}_cv{fold}{suffix}"
                bst.save_model(str(model_path.parent / name))

    # Train final model on all of the training data
    dtrain_all = xgb.DMatrix(
        X_train,
        label=y_train,
        missing=np.nan,
        feature_names=feature_cols,
        weight=_invs_prob_weight(y_train) if invs_prob_weight else None
    )

    if optimization and val_size is not None:
        # use the optuna training set as validation set for the model
        dtr_optuna = xgb.DMatrix(X_tune_train,
                                 label=y_tune_train,
                                 missing=np.nan,
                                 weight=_invs_prob_weight(y_tune_train) if invs_prob_weight else None)
        evals = [(dtrain_all, "train"), (dtr_optuna, "valid")]
    else:
        evals = [(dtrain_all, "train")]

    evals_result = {}
    final_bst = xgb.train(
        best_params,
        dtrain_all,
        num_boost_round=num_boost_round,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=200,
    )
    best_iteration = final_bst.best_iteration
    best_score = np.round(final_bst.best_score, 3)
    print(f"Best iteration: {best_iteration} with score: {best_score}")

    # Predict on test data only
    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        missing=np.nan,
        feature_names=feature_cols,
        weight=_invs_prob_weight(y_test) if invs_prob_weight else None
    )
    test_preds = final_bst.predict(dtest)

    # 3) Record test‐set results
    preds_data.update({"test_index": list(y_test.index),
                       "test_true": y_test.tolist(),
                       "test_pred": test_preds.tolist(),
                       'test_strata': df.loc[y_test.index, stratify_col].tolist()} if stratify_col else None)
    # 3.b) Record train‐set results
    train_preds = final_bst.predict(dtrain_all)
    preds_data.update({"train_index": list(y_train.index),
                       "train_true": y_train.tolist(),
                       "train_pred": train_preds.tolist(),
                       'train_strata': df.loc[y_train.index, stratify_col].tolist()} if stratify_col else None)
    # 3.c) Record val‐set results
    if optimization and val_size is not None:
        val_preds = final_bst.predict(dtr_optuna)
        preds_data.update({"val_index": list(y_tune_train.index),
                           "val_true": y_tune_train.tolist(),
                           "val_pred": val_preds.tolist(),
                           "val_strata": df.loc[y_tune_train.index, stratify_col].tolist()} if stratify_col else None)

    # pad to same length and save as frame
    max_length = max(len(lst) for lst in preds_data.values())
    for key, lst in preds_data.items():
        if len(lst) < max_length:
            # extend with NaNs (or None if you prefer)
            lst.extend([np.nan] * (max_length - len(lst)))

    df_preds = pd.DataFrame(preds_data)

    # feature importance
    imp_store = {}
    for t in importance_types:
        scores = final_bst.get_score(importance_type=t)
        imp_store[t] = scores
    df_imp = pd.DataFrame(imp_store)
    df_imp.reset_index(inplace=True, names=['feature'], drop=False)
    df_imp.sort_values(by='gain', inplace=True, ascending=False)

    # loss curves
    # ─── Plot training-vs-validation curve ───
    metric = best_params.get("eval_metric", "rmse")
    epochs = len(evals_result["train"][metric])
    plt.figure(figsize=(9, 7))
    plt.plot(range(epochs), evals_result["train"][metric], label="Train")
    plt.scatter(best_iteration,
                best_score,
                color='red',
                s=100,
                label=f"Early stop ({best_iteration:.0f}; {best_score:.2f})")
    plt.plot(range(epochs), evals_result["valid"][metric], label="Validation")
    plt.xlabel("Boosting Round")
    plt.ylabel(metric.upper())
    plt.title(f"XGBoost {metric.upper()} \nTrain vs. Validation")
    plt.legend()
    plt.grid(True)


    # ensure output directory exists
    if model_path and not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)


    # save all the variables
    if model_path:
        final_bst.save_model(str(model_path.joinpath(f'final_bst{suffix}')))
        df_imp.to_csv(str(model_path.joinpath(f'df_imp.csv')),  index=False)
        df_preds.to_csv(str(model_path.joinpath(f'df_preds.csv')),  index=False)
        df_best_params = pd.DataFrame(best_params, index=[0])
        df_best_params.to_csv(str(model_path.joinpath(f'df_best_params.csv')),  index=False)
        plt.savefig(str(model_path.joinpath("train_valid_curve.png")),
                    bbox_inches="tight",
                    dpi=300)

    if show_loss_curve:
        plt.show()
    plt.close()

    return df_imp, df_preds, best_params, final_bst


def train_xgb_collect_wrapper(df_data: pd.DataFrame,
                        features: List[str],
                        target_col: str,
                          model_type: str,
                        in_params:Dict[str,Any],
                        optimization: bool = True,
                        val_size:float = 0.3,
                        test_size:float = 0.1,
                        n_trials:int = 15,
                        cv_folds:bool = False,
                        use_gpu:bool = True,
                        stratify_col:str = 'osa_four',
                        model_path:Optional[pathlib.Path]= None,
                        show_loss_curve:bool = True,
                        resample:bool = False,
                        random_state:int = 42,
                        label_dict:Optional[Dict[int,str]]=None
                        ) -> None:

    def apply_expm1_to_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform for log1p (i.e., apply expm1) to selected columns.

        Parameters:
        - df: DataFrame containing the columns to transform.
        - columns: List of column names to apply the transformation to.

        Returns:
        - Modified DataFrame with expm1 applied to specified columns.
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = np.expm1(df.loc[mask, col])
        return df

    if model_type not in ['regressor', 'classifier']:
        raise ValueError(f"Model type {model_type} not supported. Please select from ['regressor', 'classifier']")

    formal_target_names = {
        'osa_four_numeric': 'OSA Multiclass',
        'osa_binary_numeric': 'OSA Five',
        'binary_fifteenth_numeric': 'OSA Fifteen',
        "ahi_log1p": 'Log1(AHI)'
    }
    name_target_col = formal_target_names.get(target_col, target_col)
    # check features with no stratification inside
    features = [fet for fet in features if not fet in [stratify_col, target_col]]
    fi_df, preds_df, best_params, final_bst = train_xgb_collect(
        data=df_data,
        # in_params={'interaction_constraints': interaction_constraints},
        in_params=in_params,
        feature_cols=features,
        target_col=target_col,
        optimization=optimization,
        val_size=val_size,
        test_size=test_size,
        n_trials=n_trials,
        cv_folds=cv_folds,
        use_gpu=use_gpu,
        stratify_col=stratify_col,
        model_path=model_path,
        show_loss_curve=show_loss_curve,
        resample=resample,
        random_state=random_state,
    )
    plot_xgb_feature_imp(fi_df=fi_df,
                         top_n=20,
                         ncol=2,
                         output_path=model_path)

    if model_type == 'regressor':
        # === Apply inverse log1p transformation if needed ===
        if 'log1p' in target_col:
            for split in ['train', 'test', 'val']:
                cols_to_transform = [f'{split}_true', f'{split}_pred']
                preds_df = apply_expm1_to_columns(preds_df, cols_to_transform)

        df_metrics, df_metrics_summary = compute_regression_metrics(preds_df=preds_df,
                                                                    n_feats=len(features),
                                                                    output_dir=model_path)

        for split in ['train', 'test', 'val']:
            # split = 'train' #  'test', 'train'
            y_true = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_true']
            y_pred = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_pred']
            hue = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_strata']
            # hue = df_data.loc[y_true.index, stratify_col]

            plot_true_vs_predicted(
                # y_true=y_true.values,
                y_true=y_true,
                # y_pred=y_pred.values,
                y_pred=y_pred,
                title=f'True vs. Pred. - {name_target_col}\n {(split.capitalize())}',
                textstr='',
                hue=hue.values,
                output_path=model_path.joinpath(f'{target_col}_true_vs_pred_{split}.png'),
            )

            plot_true_vs_pred_with_percentiles(
                # y_true=y_true.values,
                y_true=y_true,
                # y_pred=y_pred.values,
                y_pred=y_pred,
                hue=hue.values,
                title=f'Mean Response {split.capitalize()} - {target_col} \nPredictions vs Targets',
                output_path=model_path.joinpath(f'{target_col}_true_vs_pred_percentiles_{split}.png'),
            )

            df_eval = pd.DataFrame({'True': y_true, 'Pred': y_pred, 'hue': hue})
            plot_true_pred_histograms_stacked(df_eval=df_eval,
                                              output_path=model_path.joinpath(
                                                  f'{target_col}_true_vs_hist_{split}.png'))

    else:
        if label_dict and df_data[target_col].nunique() != len(label_dict):
            label_dict = None
        df_metrics = compute_classifier_metrics(preds_df=preds_df,
                                                label_dict=label_dict,
                                                output_dir=model_path)

        print(tabulate(
            df_metrics,
            headers='keys',
            tablefmt='psql',
            showindex=False
        ))

        for split in ['train', 'val', 'test']:
            # Extract y_true and y_pred (which is a list of probabilities)
            y_true = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_true'].values
            y_pred_raw = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_pred'].values

            # Convert list-of-probabilities to:
            # - predicted class index
            # - probability of positive class for AUC (binary only)
            # Handle different prediction formats
            if isinstance(y_pred_raw[0], (list, np.ndarray)):
                # Multiclass softprob → convert to class index
                y_pred = np.array([np.argmax(p) for p in y_pred_raw])
                y_prob = np.array([p[1] if len(p) > 1 else p[0] for p in y_pred_raw])  # Prob. for class 1 (binary)
            elif np.issubdtype(y_pred_raw.dtype, np.floating) and np.all((y_pred_raw >= 0) & (y_pred_raw <= 1)):
                # Binary prob → threshold to class
                y_pred = (y_pred_raw > 0.5).astype(int)
                y_prob = y_pred_raw
            else:
                # Hard label output
                y_pred = y_pred_raw
                y_prob = None

            plot_confusion_and_auc(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                label_dict=label_dict,
                output_path=model_path.joinpath(f'{split}_confusion_and_auc.png'),
                title_prefix=f"{name_target_col} - ",
                split_name=split
            )
    return df_metrics


# %% Model train with pre-defined train, val, and test folds
def train_xgb_collect_defined_folds(
        df_train:pd.DataFrame,
        df_val:pd.DataFrame,
        df_test:pd.DataFrame,
        target:str,
        features:List[str],
        val_size:float=.20,
        in_params: Optional[Dict] = None,
        optimization: bool = True,
        n_trials: int = 50,
        cv_folds: Union[int, bool] = 5,
        random_state: int = 42,
        use_gpu: bool = True,
        stratify_col: str = None,
        model_path: pathlib.Path = None,
        save_cv_models: bool = False,
        show_loss_curve: bool = False,
        invs_prob_weight: Optional[bool] = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, xgb.Booster]:
    """
    Train an XGBoost regressor with optional Optuna tuning, early stopping,
    and k-fold cross-validation, then evaluate on a held-out test set.

    Args:
        data (pd.DataFrame): Full dataset containing both features and target.
        feature_cols (list[str]): Names of columns to use as input features.
        target (str): Name of the column to predict.
        in_params (dict | None): Extra XGBoost parameters to include (e.g., objective, eval_metric).
        optimization (bool): If True, run Optuna hyperparameter tuning.
        n_trials (int): Number of Optuna trials when `optimization` is True.
        cv_folds (int | bool): Number of folds for CV; set to False or 1 to skip CV.
        test_size (float): Fraction of data reserved for final held-out test evaluation.
        val_size (float | None): Fraction of training data to reserve for validation during tuning.
        random_state (int): Seed for reproducible splits and model initialization.
        use_gpu (bool): If True, enable GPU acceleration (`tree_method='hist'`, `device='cuda'`).
        stratify_col (str | None): Column name to use for stratified splits; None for random splits.
        model_path (pathlib.Path | None): Directory where models, CSVs, and plots will be saved.
        save_cv_models (bool): If True, persist each fold’s model to `model_path`.
        show_loss_curve (bool): If True, plots & saves the train vs. validation loss curve.

    Returns:
        fi_df (pd.DataFrame): Feature-importance table (mean & std across folds).
        preds_df (pd.DataFrame): DataFrame of per-fold and test set predictions & true values.
        best_params (dict): Best hyperparameter configuration (from Optuna or defaults).
        final_model (xgb.Booster): Booster trained on the full training set.
    """

    def _gpu(p: Dict) -> Dict:
        if use_gpu:
            p.update(tree_method='hist', device='cuda')
        return p

    # define path to save the models
    if model_path:
        if not model_path.exists():
            model_path.mkdir(parents=True, exist_ok=True)
        base = model_path.stem
        suffix = ".json"

    if in_params:
        if in_params.get('objective') == 'reg:gamma':
            epsilon = 0.001
            df_train[target] += epsilon
            df_val[target] += epsilon
            df_test[target] += epsilon
        if in_params.get('objective').split(':')[0] in ['multi', 'binary']:
            pred_prob = True

    # 3) hyperparameter tuning for optimziation from the val set
    if optimization and df_val is not None and val_size is not None:
        tune_args = {'test_size': val_size, 'random_state': random_state}
        if stratify_col:
            tune_args['stratify'] = df_val[stratify_col].values
        # remove a tuning split from the training sample
        X_tune_train, X_tune_test, y_tune_train, y_tune_train_test = train_test_split(
            df_val[features], df_val[target], **tune_args
        )


    # objective baseline
    obj = in_params.get('objective', 'reg:squarederror') if in_params else 'reg:squarederror'
    eval_metric = in_params.get('eval_metric', 'rmse') if in_params else "rmse"
    # tune with Optuna
    if optimization:

        class ThresholdEarlyStopping:
            """
            Optuna callback to early stop hyperparameter optimization when there is no
            *significant* improvement in the objective value for a given number of trials.

            This is useful when your optimization process converges early and continues
            evaluating similar parameter sets with marginal gains. For example, if your
            validation RMSE is stuck around 1.0800±0.0005 for many trials, you can save
            time by stopping after `patience` trials without an improvement greater than `min_delta`.

            Parameters:
                patience (int): Number of consecutive trials without significant improvement
                                allowed before stopping the study.
                min_delta (float): Minimum change in best value to be considered an improvement.

            Example usage:
                stopper = ThresholdEarlyStopping(patience=20, min_delta=1e-4)
                study.optimize(objective, n_trials=100, callbacks=[stopper])
            """

            def __init__(self, patience: int = 20, min_delta: float = 1e-4):
                self.patience = patience
                self.min_delta = min_delta
                self.best_score = None
                self.counter = 0

            def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
                current_best = study.best_value

                if self.best_score is None:
                    self.best_score = current_best
                    return

                if self.best_score - current_best > self.min_delta:
                    self.best_score = current_best
                    self.counter = 0
                else:
                    self.counter += 1

                if self.counter >= self.patience:
                    print(f"Early stopping: No significant improvement greater than "
                          f"{self.min_delta}) in the last {self.patience} trials.")
                    study.stop()

        def objective(trial):
            # === Suggest hyperparameters ===
            p = {
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "lambda": trial.suggest_float("lambda", 1.1, 10.0),
                "gamma": trial.suggest_float("gamma", 0, 10.0),
                "alpha": trial.suggest_float("alpha", 1.1, 10.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "objective": obj,  # e.g., "multi:softprob"
                "eval_metric": eval_metric,  # e.g., "mlogloss"
                "random_state": random_state,
            }

            # Include num_class only for multi-class objectives
            if in_params and in_params.get('num_class'):
                p["num_class"] = in_params["num_class"]

            p = _gpu(p)  # Use GPU if available

            # === Prepare data ===
            dtr = xgb.DMatrix(
                X_tune_train,
                label=y_tune_train,
                missing=np.nan,
                weight=_invs_prob_weight(y_tune_train) if invs_prob_weight else None
            )
            dval = xgb.DMatrix(
                X_tune_test,
                label=y_tune_train_test,
                missing=np.nan,
                weight=_invs_prob_weight(y_tune_train_test) if invs_prob_weight else None
            )

            # === Train model ===
            bst = xgb.train(
                {k: v for k, v in p.items() if k != 'num_boost_round'},
                dtr,
                num_boost_round=2000,
                evals=[(dtr, "train"), (dval, "valid")],
                early_stopping_rounds=100,
                verbose_eval=False,
            )

            pred = bst.predict(dval)

            # === Score: Quadratic Weighted Kappa ===
            def _poisson_nll(y_true, y_pred):
                eps = 1e-9  # avoid log(0)
                y_pred = np.clip(y_pred, eps, None)
                return np.mean(y_pred - y_true * np.log(y_pred) + gammaln(y_true + 1))

            def compute_score(pred, y_true, p):
                if p["objective"] == "multi:softprob":
                    pred_labels = np.argmax(pred, axis=1)
                    return -cohen_kappa_score(y_true, pred_labels, weights="quadratic")
                elif p["objective"] == "multi:softmax":
                    return -cohen_kappa_score(y_true, pred.astype(int), weights="quadratic")
                elif p["objective"] in ["binary:logistic", "binary:logitraw"]:
                    pred_labels = (pred > 0.5).astype(int)
                    return log_loss(y_true, pred_labels)
                elif p["objective"].startswith("reg:"):
                    return np.sqrt(mean_squared_error(y_true, pred))
                elif p["objective"] == "reg:poisson":
                    return _poisson_nll(y_true, pred)
                else:
                    raise ValueError(f"No scoring rule defined for objective '{p['objective']}'")

            return compute_score(pred, y_tune_train_test, p)

        threshold_early_stopper = ThresholdEarlyStopping(patience=20, min_delta=1e-4)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective,
                       n_trials=n_trials,
                       callbacks=[threshold_early_stopper])
        best = study.best_params
        best_params = {
            **best,
            "objective": obj,
            "eval_metric": eval_metric,
            "random_state": random_state
        }
        best_params = _gpu(best_params)
    else:
        best_params = _gpu({
            'num_boost_round': 1000,
            'max_depth': 4,
            'learning_rate': 0.00527,
            # 'subsample':    0.58,
            'objective': obj,
            'eval_metric': eval_metric,
            'random_state': random_state
        })

    if in_params:
        # only inject those in_params keys that Optuna did *not* tune
        for k, v in in_params.items():
            if k not in best_params:
                best_params[k] = v
        # or we can write in one line:  best_params = {**in_params, **best_params}

        best_params.update(in_params)
        if 'interaction_constraints' in in_params:
            best_params['interaction_constraints'] = _interactions_constraint_to_indexes(features=features,
                                                                                         interaction_constraints=
                                                                                         best_params['interaction_constraints'])

    # get the train and test set
    X_train = df_train[features]
    y_train = df_train[target]

    X_test = df_test[features]
    y_test = df_test[target]

    # ─── Diagnostic printout ────────────────────────────────────────────────
    print("=== XGBoost parameters ===")
    for k, v in best_params.items():
        print(f"  {k:20s} : {v}")
    print("\n=== Dataset splits ===")
    # full training set (for CV)
    print(f"  X_train   : {X_train.shape}   y_train   : {y_train.shape}")
    # tuning splits (only if you created them)
    if optimization and val_size is not None:
        print(f"  X_tune_train     : {X_tune_train.shape}   y_tune_train        : {y_tune_train.shape}")
        print(f"  X_tune_test      : {X_tune_test.shape}   y_tune_train_test   : {y_tune_train_test.shape}")
    # held‐out test set
    print(f"  X_test    : {X_test.shape}   y_test    : {y_test.shape}")
    print("──────────────────────────────────────────────────────────────────────\n")

    if 'num_boost_round' in best_params:
        num_boost_round = best_params['num_boost_round']
        del best_params['num_boost_round']
    else:
        num_boost_round = 1000

    if 'early_stopping_rounds' in best_params:
        early_stopping_rounds = best_params['early_stopping_rounds']
        del best_params['early_stopping_rounds']
    else:
        early_stopping_rounds = 200

    # initialize preds container
    n_feats = len(features)
    importance_types = ['weight', 'gain', 'cover']

    preds_data = {}
    if cv_folds:
        df_cv_fold = pd.concat([df_train, df_val], axis=0)
        assert df_cv_fold.shape[0] == df_train.shape[0] + df_val.shape[0]

        splitter = (
            StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            if stratify_col else
            KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        )

        # choose correct split iterator
        if stratify_col:
            strat_vals = df_cv_fold[stratify_col].values
            cv_iterator = splitter.split(df_cv_fold, strat_vals)
        else:
            cv_iterator = splitter.split(df_cv_fold)

        for fold, (tr_ix, val_ix) in enumerate(
                tqdm(cv_iterator, total=cv_folds, desc="CV folds"), start=1):

            X_tr = df_cv_fold.iloc[tr_ix][features]
            y_tr = df_cv_fold.iloc[tr_ix][target]
            X_val = df_cv_fold.iloc[val_ix][features]
            y_val = df_cv_fold.iloc[val_ix][target]

            dtr = xgb.DMatrix(X_tr,
                              label=y_tr,
                              missing=np.nan,
                              feature_names=features)
            dval = xgb.DMatrix(X_val,
                               label=y_val,
                               missing=np.nan,
                               feature_names=features)

            bst = xgb.train(
                best_params, dtr,
                num_boost_round=num_boost_round,
                evals=[(dtr, 'train'), (dval, 'valid')],
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=200,
            )

            # store predictions
            preds_data[f'train_fold_{fold}_index'] = list(df_cv_fold.iloc[tr_ix].index)
            preds_data[f'train_fold_{fold}_true'] = y_tr.to_list()
            preds_data[f'train_fold_{fold}_pred'] = list(bst.predict(dtr))

            preds_data[f'val_fold_{fold}_index'] = list(df_cv_fold.iloc[val_ix].index)
            preds_data[f'val_fold_{fold}_true'] = y_val.to_list()
            preds_data[f'val_fold_{fold}_pred'] = list(bst.predict(dval))

            # save model
            if model_path and save_cv_models:
                name = f"{base}_cv{fold}{suffix}"
                bst.save_model(str(model_path.parent / name))

    # Train final model on all of the training data
    dtrain_all = xgb.DMatrix(
        X_train,
        label=y_train,
        missing=np.nan,
        feature_names=features,
        weight=_invs_prob_weight(y_train) if invs_prob_weight else None
    )

    if optimization and val_size is not None:
        # use the optuna training set as validation set for the model
        dtr_optuna = xgb.DMatrix(X_tune_train,
                                 label=y_tune_train,
                                 missing=np.nan,
                                 weight=_invs_prob_weight(y_tune_train) if invs_prob_weight else None)
        evals = [(dtrain_all, "train"), (dtr_optuna, "valid")]
    else:
        evals = [(dtrain_all, "train")]

    evals_result = {}
    final_bst = xgb.train(
        best_params,
        dtrain_all,
        num_boost_round=num_boost_round,
        evals=evals,
        evals_result=evals_result,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=200,
    )
    best_iteration = final_bst.best_iteration
    best_score = np.round(final_bst.best_score, 3)
    print(f"Best iteration: {best_iteration} with score: {best_score}")

    # Predict on test data only
    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        missing=np.nan,
        feature_names=features,
        weight=_invs_prob_weight(y_test) if invs_prob_weight else None
    )
    test_preds = final_bst.predict(dtest)

    # 3) Record test‐set results
    preds_data.update({"test_index": list(y_test.index),
                       "test_true": y_test.tolist(),
                       "test_pred": test_preds.tolist(),
                       'test_strata': df_test[stratify_col].tolist()} if stratify_col else None)
    # 3.b) Record train‐set results
    train_preds = final_bst.predict(dtrain_all)
    preds_data.update({"train_index": list(y_train.index),
                       "train_true": y_train.tolist(),
                       "train_pred": train_preds.tolist(),
                       'train_strata': df_train[stratify_col].tolist()} if stratify_col else None)
    # 3.c) Record val‐set results
    if optimization and val_size is not None:
        val_preds = final_bst.predict(dtr_optuna)
        preds_data.update({"val_index": list(y_tune_train.index),
                           "val_true": y_tune_train.tolist(),
                           "val_pred": val_preds.tolist(),
                           "val_strata": df_val[stratify_col].tolist()} if stratify_col else None)

    # pad to same length and save as frame
    max_length = max(len(lst) for lst in preds_data.values())
    for key, lst in preds_data.items():
        if len(lst) < max_length:
            # extend with NaNs (or None if you prefer)
            lst.extend([np.nan] * (max_length - len(lst)))

    df_preds = pd.DataFrame(preds_data)

    # feature importance
    imp_store = {}
    for t in importance_types:
        scores = final_bst.get_score(importance_type=t)
        imp_store[t] = scores
    df_imp = pd.DataFrame(imp_store)
    df_imp.reset_index(inplace=True, names=['feature'], drop=False)
    df_imp.sort_values(by='gain', inplace=True, ascending=False)

    # loss curves
    # ─── Plot training-vs-validation curve ───
    metric = best_params.get("eval_metric", "rmse")
    epochs = len(evals_result["train"][metric])
    plt.figure(figsize=(9, 7))
    plt.plot(range(epochs), evals_result["train"][metric], label="Train")
    plt.scatter(best_iteration,
                best_score,
                color='red',
                s=100,
                label=f"Early stop ({best_iteration:.0f}; {best_score:.2f})")
    plt.plot(range(epochs), evals_result["valid"][metric], label="Validation")
    plt.xlabel("Boosting Round")
    plt.ylabel(metric.upper())
    plt.title(f"XGBoost {metric.upper()} \nTrain vs. Validation")
    plt.legend()
    plt.grid(True)

    # ensure output directory exists
    if model_path and not model_path.exists():
        model_path.mkdir(parents=True, exist_ok=True)

    # save all the variables
    if model_path:
        final_bst.save_model(str(model_path.joinpath(f'final_bst{suffix}')))
        df_imp.to_csv(str(model_path.joinpath(f'df_imp.csv')), index=False)
        df_preds.to_csv(str(model_path.joinpath(f'df_preds.csv')), index=False)
        df_best_params = pd.DataFrame([best_params])
        df_best_params.to_csv(str(model_path.joinpath(f'df_best_params.csv')), index=False)
        plt.savefig(str(model_path.joinpath("train_valid_curve.png")),
                    bbox_inches="tight",
                    dpi=300)

    if show_loss_curve:
        plt.show()
    plt.close()

    return df_imp, df_preds, best_params, final_bst


def train_xgb_collect_defined_folds_wrapper(df_train: pd.DataFrame,
                                            df_val: pd.DataFrame,
                                            df_test: pd.DataFrame,
                                            target_col: str,
                                            features: List[str],
                                            model_params: Dict,
                                            stratify_col: str = None,
                                            model_path: pathlib.Path = None,
                                            use_gpu: bool = True,
                                            n_trials: int = 3,
                                            val_size: float = 0.2,
                                            show_loss_curve: bool = True,
                                            label_dict:Optional[Dict[int, str]] = None,
                                            model_type: str = None) -> None:
    """
    Wrapper to train an XGBoost model using the `train_xgb_collect_defined_folds` function.

    Args:
        df_train (pd.DataFrame): Training dataset.
        df_val (pd.DataFrame): Validation dataset.
        df_test (pd.DataFrame): Test dataset.
        target_col (str): Target column name.
        features (List[str]): List of feature column names.
        model_params (Dict): Dictionary of model parameters.
        model_path (pathlib.Path): Path to save the trained model.
        use_gpu (bool, optional): Whether to use GPU. Defaults to True.
        n_trials (int, optional): Number of hyperparameter tuning trials. Defaults to 3.
        val_size (float, optional): Validation size for splitting. Defaults to 0.2.
        show_loss_curve (bool, optional): Show training loss curve. Defaults to True.
    """

    def apply_expm1_to_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform for log1p (i.e., apply expm1) to selected columns.

        Parameters:
        - df: DataFrame containing the columns to transform.
        - columns: List of column names to apply the transformation to.

        Returns:
        - Modified DataFrame with expm1 applied to specified columns.
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = np.expm1(df.loc[mask, col])
        return df

    if model_type not in ['regressor', 'classifier']:
        raise ValueError(f"Model type {model_type} not supported. Please select from ['regressor', 'classifier']")

    formal_target_names = {
        'osa_four_numeric': 'OSA Multiclass',
        'osa_binary_numeric': 'OSA Five',
        'binary_fifteenth_numeric': 'OSA Fifteen',
        "ahi_log1p": 'Log1(AHI)'
    }
    name_target_col  = formal_target_names.get(target_col, target_col)
    # check features with no stratification inside
    features = [fet for fet in features if not fet in [stratify_col, target_col]]
    fi_df, preds_df, best_params, final_bst = train_xgb_collect_defined_folds(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        features=features,
        target=target_col,
        val_size=val_size,
        n_trials=n_trials,
        in_params=model_params,
        random_state=42,
        cv_folds=False,
        use_gpu=use_gpu,
        stratify_col=stratify_col,
        model_path=model_path,
        show_loss_curve=show_loss_curve,
        invs_prob_weight=True
    )

    plot_xgb_feature_imp(fi_df=fi_df,
                         top_n=20,
                         ncol=2,
                         output_path=model_path)


    if model_type == 'regressor':
        # === Apply inverse log1p transformation if needed ===
        if 'log1p' in target_col:
            for split in ['train', 'test', 'val']:
                cols_to_transform = [f'{split}_true', f'{split}_pred']
                preds_df = apply_expm1_to_columns(preds_df, cols_to_transform)


        df_metrics, df_metrics_summary = compute_regression_metrics(preds_df=preds_df,
                                                                    n_feats=len(features),
                                                                    output_dir=model_path)

        for split in ['train','test', 'val']:
            # split = 'train' #  'test', 'train'
            y_true = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_true']
            y_pred = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_pred']
            hue = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_strata']
            # hue = df_data.loc[y_true.index, stratify_col]

            plot_true_vs_predicted(
                # y_true=y_true.values,
                y_true=y_true,
                # y_pred=y_pred.values,
                y_pred=y_pred,
                title=f'True vs. Pred. - {name_target_col}\n {(split.capitalize())}',
                textstr='',
                hue=hue.values,
                output_path=model_path.joinpath(f'{target_col}_true_vs_pred_{split}.png'),
            )

            plot_true_vs_pred_with_percentiles(
                # y_true=y_true.values,
                y_true=y_true,
                # y_pred=y_pred.values,
                y_pred=y_pred,
                hue=hue.values,
                title=f'Mean Response {split.capitalize()} - {target_col} \nPredictions vs Targets',
                output_path=model_path.joinpath(f'{target_col}_true_vs_pred_percentiles_{split}.png'),
            )

            df_eval = pd.DataFrame({'True': y_true, 'Pred': y_pred, 'hue': hue})
            plot_true_pred_histograms_stacked(df_eval=df_eval,
                                              output_path=model_path.joinpath(
                                                  f'{target_col}_true_vs_hist_{split}.png'))

    else:
        if df_train[target_col].nunique() != len(label_dict):
            label_dict = None
        df_metrics = compute_classifier_metrics(preds_df=preds_df,
                                                label_dict=label_dict,
                                                output_dir=model_path)

        print(tabulate(
            df_metrics,
            headers='keys',
            tablefmt='psql',
            showindex=False
        ))

        for split in ['train', 'val', 'test']:
            # Extract y_true and y_pred (which is a list of probabilities)
            y_true = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_true'].values
            y_pred_raw = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_pred'].values

            # Convert list-of-probabilities to:
            # - predicted class index
            # - probability of positive class for AUC (binary only)
            # Handle different prediction formats
            if isinstance(y_pred_raw[0], (list, np.ndarray)):
                # Multiclass softprob → convert to class index
                y_pred = np.array([np.argmax(p) for p in y_pred_raw])
                y_prob = np.array([p[1] if len(p) > 1 else p[0] for p in y_pred_raw])  # Prob. for class 1 (binary)
            elif np.issubdtype(y_pred_raw.dtype, np.floating) and np.all((y_pred_raw >= 0) & (y_pred_raw <= 1)):
                # Binary prob → threshold to class
                y_pred = (y_pred_raw > 0.5).astype(int)
                y_prob = y_pred_raw
            else:
                # Hard label output
                y_pred = y_pred_raw
                y_prob = None


            plot_confusion_and_auc(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                label_dict=label_dict,
                output_path=model_path.joinpath(f'{split}_confusion_and_auc.png'),
                title_prefix=f"{name_target_col} - ",
                split_name=split
            )
    return df_metrics


# %% Stacking XGBoost models
def train_regressors(data: pd.DataFrame,
                    target: str,
                    tgt_lbl:str,
                    features: List[str],
                    stratify_col: str,
                    n_trials:int=45,
                    num_boost_round:int=100,
                    dir_path: Optional[pathlib.Path] = None, ) -> Dict[str, Any]:
    """
    Trains regression models using XGBoost framework and computes performance metrics.

    This function preprocesses the input data, trains regression models using XGBoost with
    specific parameters, calculates regression metrics, and generates plots for feature
    importance and predicted vs. true values. The resulting model is saved to the specified
    directory, along with visualizations and metric summaries. The function specifically
    supports gamma distribution regression targets.

    :param data: A pandas DataFrame containing the input data. Must include both feature
        columns and the target column.
    :type data: pd.DataFrame
    :param target: The name of the target column to be predicted in the input DataFrame.
    :type target: str
    :param features: A list of feature column names to be used as predictors.
    :type features: List[str]
    :param stratify_col: The name of the column used for stratification.
    :type stratify_col: str
    :param dir_path: The path to the directory where model outputs will be saved.
    :type dir_path: pathlib.Path
    :return: A dictionary where the key is the target column name and the value is the
        trained XGBoost model.
    :rtype: Dict[str, Any]
    """
    def apply_expm1_to_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform for log1p (i.e., apply expm1) to selected columns.

        Parameters:
        - df: DataFrame containing the columns to transform.
        - columns: List of column names to apply the transformation to.

        Returns:
        - Modified DataFrame with expm1 applied to specified columns.
        """
        df = df.copy()
        for col in columns:
            if col in df.columns:
                mask = df[col].notna()
                df.loc[mask, col] = np.expm1(df.loc[mask, col])
        return df

    # clean_target = re.sub(r'[<>:"/\\|?*]', '', target)

    dir_current_target = dir_path.joinpath(f'{target}')
    dir_current_target.mkdir(parents=True, exist_ok=True)

    col_model =  features + [target] + [stratify_col]
    assert set(col_model).issubset(data.columns), "Some columns in col_model are missing from df_data.columns"

    df_model = data.loc[~data[target].isna(), col_model].copy()
    # for gamma dist
    epsilon = 0.001
    df_model[target] = df_model[target] + epsilon

    # X = df_model[features].copy()
    # y = df_model[target].copy()

    # output_dir_current = path_output.joinpath(f'{target}')
    fi_df, preds_df, best_params, final_bst = train_xgb_collect(
        data=df_model,
        # in_params={'interaction_constraints': interaction_constraints},
        in_params={
            'objective': 'reg:gamma',  # 'reg:squarederror',
            'eval_metric': "rmse",

            # 'objective': 'reg:tweedie',
            # 'tweedie_variance_power': 1.3,
            # 'eval_metric': "tweedie-nloglik@1.3",
            # 'gamma': 0.1,
            'max_bin': 512,
            'num_parallel_tree': 10,
            'n_e'
            'early_stopping_rounds': 100,
            'num_boost_round': num_boost_round
            # 'updater': 'coord_descent',
            # 'feature_selector': 'greedy'
        },
        feature_cols=features,
        target_col=target,
        optimization=True,
        val_size=0.3,
        test_size=0.1,
        n_trials=n_trials,
        cv_folds=False,
        use_gpu=True,
        stratify_col=stratify_col,
        model_path=dir_current_target,
        show_loss_curve=True,
        resample=False,
        invs_prob_weight=True
    )
    # === Apply inverse log1p transformation if needed ===
    if 'log1p' in target:
        for split in ['train', 'test', 'val']:
            cols_to_transform = [f'{split}_true', f'{split}_pred']
            preds_df = apply_expm1_to_columns(preds_df, cols_to_transform)

        # Apply to all *_true and *_pred columns for each split
        for split in ['train', 'test', 'val']:
            apply_expm1_to_columns(preds_df, [f'{split}_true', f'{split}_pred'])

    df_metrics, df_metrics_summary = compute_regression_metrics(preds_df=preds_df,
                                                                n_feats=len(features),
                                                                output_dir=dir_current_target)

    plot_xgb_feature_imp(fi_df=fi_df,
                         top_n=20,
                         ncol=2,
                         output_path=dir_current_target)

    for split in ['train', 'test', 'val']:
        # split = 'train' #  'test', 'train'
        y_true = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_true']
        y_pred = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_pred']
        hue = preds_df.loc[preds_df[f'{split}_true'].notna(), f'{split}_strata']
        # hue = df_data.loc[y_true.index, stratify_col]

        plot_true_vs_predicted(
            # y_true=y_true.values,
            y_true=y_true,
            # y_pred=y_pred.values,
            y_pred=y_pred,
            title=f'True vs. Pred. - {tgt_lbl}',
            textstr='',
            hue=hue.values,
            output_path=dir_current_target.joinpath(f'{target}_true_vs_pred_{split}.png'),
        )

        plot_true_vs_pred_with_percentiles(
            # y_true=y_true.values,
            y_true=y_true,
            # y_pred=y_pred.values,
            y_pred=y_pred,
            hue=hue.values,
            title=f'Mean Response {split.capitalize()} - {tgt_lbl} \nPredictions vs Targets',
            output_path=dir_current_target.joinpath(f'{target}_true_vs_pred_percentiles_{split}.png'),
        )
        df_eval = pd.DataFrame({'True':y_true, 'Pred': y_pred, 'hue': hue })
        plot_true_pred_histograms_stacked(df_eval=df_eval,
                                          output_path=dir_current_target.joinpath(f'{target}_true_vs_hist_{split}.png'))

    return {'model': final_bst,'preds_df': preds_df}



def stack_regressors_and_data(xgb_regressors: Dict[str, Any],
                              df_raw: pd.DataFrame,
                              include_columns: List[str] = None
                              ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stack predictions from multiple regressors into train, val, and test sets,
    indexed by their original data indexes, and optionally include columns from raw data.

    Parameters:
        xgb_regressors (dict): Dictionary with model outputs and predictions.
        df_raw (pd.DataFrame): Original dataset (used for reference or extra features).
        include_columns (list): Optional list of column names from df_raw to include.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Stacked train, val, and test DataFrames.
    """

    # Internal helper to extract and stack predictions by index
    def _extract_and_stack_preds(preds_df: pd.DataFrame,
                                 column_name: str,
                                 index_name: str,
                                 target: str,
                                 df_stack: pd.DataFrame) -> pd.DataFrame:
        """
        Stacks predictions from multiple models and associated raw data for further analysis or processing.
        The stacking is done by index so we stack the same predictions together
        :param xgb_regressors: Dictionary of XGBoost regressors keyed by their names or identifiers.
        :param df_raw: The raw input DataFrame containing data to be included alongside model predictions.
        :param include_columns: List of column names to include from the raw DataFrame in the stacked output.
        :return: A tuple containing three DataFrames:
                 - The first DataFrame contains the stacked predictions from all regressors.
                 - The second DataFrame includes a subset of raw data specified by `include_columns`.
                 - The third DataFrame is unprocessed, raw data.
        """
        stack = preds_df[[column_name, index_name]].dropna()
        stack.index = stack[index_name].astype(int)
        stack.index.name = None
        stack = stack.drop(index_name, axis=1)
        stack = stack.rename(columns={column_name: f'{target}'})

        df_stack = (
            pd.merge(df_stack, stack, how='outer', left_index=True, right_index=True)
            if not df_stack.empty else stack
        )
        return df_stack

    # Initialize empty DataFrames for stacking
    df_train_stack = pd.DataFrame()
    df_val_stack = pd.DataFrame()
    df_test_stack = pd.DataFrame()

    # Loop over each model's predictions
    for target, meta in xgb_regressors.items():
        # target = [*xgb_regressors.keys()][0]
        # meta = xgb_regressors.get(target)
        preds_df = meta.get('preds_df') if meta else None
        if preds_df is None:
            continue

        # Stack predictions for train, val, and test
        df_train_stack = _extract_and_stack_preds(preds_df=preds_df,
                                                  column_name='train_pred',
                                                  index_name='train_index',
                                                  target=target,
                                                  df_stack=df_train_stack)
        df_val_stack = _extract_and_stack_preds(preds_df=preds_df,
                                                column_name='val_pred',
                                                index_name='val_index',
                                                target=target,
                                                df_stack=df_val_stack)
        df_test_stack = _extract_and_stack_preds(preds_df=preds_df,
                                                 column_name='test_pred',
                                                 index_name='test_index',
                                                 target=target,
                                                 df_stack=df_test_stack)
    # check they all have different idnexes
    assert df_train_stack.index.isin(df_val_stack.index).sum() == 0, "Train and val indexes overlap!"
    assert df_train_stack.index.isin(df_test_stack.index).sum() == 0, "Train and test indexes overlap!"
    assert df_val_stack.index.isin(df_test_stack.index).sum() == 0, "Val and test indexes overlap!"

    # Optionally include selected columns from df_raw
    if include_columns:
        extra_cols = df_raw[include_columns]

        # Merge extra columns based on index
        df_train_stack = pd.merge(df_train_stack, extra_cols, how='left', left_index=True, right_index=True)
        df_val_stack = pd.merge(df_val_stack, extra_cols, how='left', left_index=True, right_index=True)
        df_test_stack = pd.merge(df_test_stack, extra_cols, how='left', left_index=True, right_index=True)

    return df_train_stack, df_val_stack, df_test_stack


def compare_regression_results(results_dict):
    """
    Create comparison DataFrame from collected results.

    Parameters:
    - results_dict: Output from collect_regression_results

    Returns:
    - DataFrame with model comparison metrics
    """
    comparison_data = []

    for target, result in results_dict.items():
        if 'error' in result:
            row = {'target': target, 'error': result['error']}
            comparison_data.append(row)
            continue

        try:
            row = {
                'target': target,
                'n_samples': result['n_samples'],
                # Cross-validation metrics
                'cv_r2_mean': result['cv_metrics']['r2_mean'],
                'cv_r2_std': result['cv_metrics']['r2_std'],
                'cv_rmse_mean': result['cv_metrics']['rmse_mean'],
                'cv_rmse_std': result['cv_metrics']['rmse_std'],
                # Test metrics
                'test_r2': result['test_metrics']['r2'],
                'test_rmse': result['test_metrics']['rmse'],
                # Error metrics
                'mae_mean': result['cv_metrics']['mae_mean'],
                'mae_std': result['cv_metrics']['mae_std'],
                'maxerr_mean': result['cv_metrics']['maxerr_mean']
            }
            comparison_data.append(row)

        except KeyError as e:
            print(f"Missing metric {str(e)} for {target}")
            comparison_data.append({'target': target, 'error': f"Missing metric {str(e)}"})

    # Create and sort DataFrame
    df = pd.DataFrame(comparison_data)
    if not df.empty:
        df = df.sort_values('test_rmse', ascending=True)

    return df



def extract_summary_metrics(df_summary: pd.DataFrame, grp: str) -> dict:
    """Return a dict with r2_adj_mean/std and rmse_mean/std for a given group."""
    row = df_summary.loc[df_summary['group'] == grp].iloc[0]
    return {
        'r2_adj_mean': row['r2_adj_mean'],
        'r2_adj_std':  row['r2_adj_std'],
        'rmse_mean':   row['rmse_mean'],
        'rmse_std':    row['rmse_std']
    }



def plot_metrics_by_target(df: pd.DataFrame,
                           output_path: pathlib.Path = None):
    """
    Plots R², Adj. R², and RMSE per target variable from the model comparison DataFrame.

    Parameters:
    - df: DataFrame with columns ['group', 'r2_mean', 'r2_adj_mean', 'rmse_mean', ...] and 'target'.
    - output_path: Directory to save the plots (optional).
    """
    metric_map = {
        'r2_mean': 'R²',
        'r2_adj_mean': 'Adj. R²',
        'rmse_mean': 'RMSE'
    }

    std_map = {
        'r2_mean': 'r2_std',
        'r2_adj_mean': 'r2_adj_std',
        'rmse_mean': 'rmse_std'
    }

    targets = df['target'].unique()
    groups = ['train', 'val', 'test']
    n_targets = len(targets)

    fig = plt.figure(figsize=(12, 3.2 * n_targets))
    gs = GridSpec(n_targets, 3, figure=fig)

    all_handles = []
    all_labels = []

    for row_idx, target in enumerate(targets):
        df_target = df[df['target'] == target]

        for col_idx, metric_key in enumerate(metric_map.keys()):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            for i, group in enumerate(groups):
                group_data = df_target[df_target['group'] == group]
                if not group_data.empty:
                    y = group_data[metric_key].values[0]
                    yerr = group_data[std_map[metric_key]].values[0] if std_map[metric_key] in group_data else None

                    if group == 'train':
                        h = ax.errorbar(i, y, yerr=yerr, fmt='o', capsize=4, color='blue', label='Train')
                    elif group == 'val':
                        h = ax.errorbar(i, y, yerr=yerr, fmt='o', capsize=4, color='orange', label='Validation')
                    else:
                        h = ax.scatter(i, y, color='black', marker='X', s=60, label='Test')

                    if row_idx == 0 and col_idx == 0:  # only collect handles once
                        handle, label = ax.get_legend_handles_labels()
                        all_handles.extend(handle)
                        all_labels.extend(label)

            ax.set_title(f"{metric_map[metric_key]} – {target}", fontsize=10)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)

    # Add single legend at the bottom center
    unique_legend = dict(zip(all_labels, all_handles))
    fig.legend(unique_legend.values(), unique_legend.keys(),
               loc='lower center', ncol=len(unique_legend), fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


# %%
def _interactions_constraint_to_indexes(features:List[str],
                                       interaction_constraints:List[List[str]]=None) -> str:
    """
    Interactions must be given as indexes numbers and not labels. It must be a string of a nested least, where each
    nested list is a group of features that are allowed to interact.
    :param features:
    :param interaction_constraints:
    :return:
    """
    features = {lbl: idx for idx, lbl in enumerate(features)}
    converted_constraints = []
    for group in interaction_constraints:
        indices = []
        for feature in group:
            if feature not in features.keys():
                raise ValueError(f"Feature '{feature}' not found in training data feature names: {features.values()}")
            indices.append(features.get(feature))
        converted_constraints.append(indices)
    return f'{converted_constraints}'

# %% Model evaluation
def compute_regression_metrics(preds_df: pd.DataFrame,
                               n_feats: int,
                               output_dir:pathlib.Path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute regression metrics for each true/pred column pair in preds_df.
    Also returns a summary DataFrame with mean & std per group (train/val/test) and
    generates a plot of these summary statistics for train vs. val metrics.

    Returns:
    - metrics_df: DataFrame with metrics per split
    - summary_df: DataFrame with metrics_{mean,std} per group ('train','val','test')
    """
    # 1) Compute per-split metrics
    records = []
    true_cols = [c for c in preds_df.columns if c.endswith('_true') and '_index' not in c]
    for true_col in true_cols:
        pred_col = true_col.replace('_true', '_pred')
        group = true_col.split('_')[0]  # 'train', 'val', or 'test'
        y_true = preds_df[true_col].dropna().values
        y_pred = preds_df[pred_col].dropna().values
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        medae = median_absolute_error(y_true, y_pred)
        maxerr = max_error(y_true, y_pred)
        expl_var = explained_variance_score(y_true, y_pred)
        r2_adj = 1 - (1 - r2) * (len(y_true) - 1) / (len(y_true) - n_feats - 1)
        records.append({
            'group': group,
            'split': true_col,
            'rmse': rmse,
            # 'mae': mae,
            'medae': medae,
            # 'maxerr': maxerr,
            # 'r2': r2,
            'explained_variance': expl_var,
            'r2_adj': r2_adj
        })
    metrics_df = pd.DataFrame(records)

    # 2) Aggregate summary by group
    agg_dict = {col: ['mean', 'std'] for col in metrics_df.columns if col not in ('group', 'split')}
    summary = metrics_df.groupby('group').agg(agg_dict)
    # flatten MultiIndex columns
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary_df = summary.reset_index()
    if output_dir:
        metrics_df.to_csv(output_dir.joinpath('metrics_df.csv'), index=False)
        summary_df.to_csv(output_dir.joinpath('summary_df.csv'), index=False)

    return metrics_df, summary_df


def plot_true_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    textstr: str,
    hue: Optional[np.ndarray] = None,
    output_path:pathlib.Path = None,
):

    df = pd.DataFrame({'True': y_true, 'Pred': y_pred})
    if hue is not None:
        hue_order = ['Normal', 'Mild', 'Moderate', 'Severe']
        df['Hue'] = pd.Categorical(hue, categories=hue_order, ordered=True)

    # compute ±1 STD on the residuals
    residuals = df['True'] - df['Pred']
    std_dev = residuals.std()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=df, x='True', y='Pred',
        hue='Hue' if hue is not None else None,
        palette='Reds',
        alpha=0.7,
        s=60,
        ax=ax
    )

    # determine data‐driven limits
    t_min, t_max = df['True'].min(), df['True'].max()
    p_min, p_max = df['Pred'].min(), df['Pred'].max()

    # only span the line & band over the overlap of true & pred ranges
    line_min = max(t_min, p_min)
    line_max = min(t_max, p_max)

    # 1:1 line
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        '--', color='gray', linewidth=2,
        label='Perfect'
    )

    # ±1 STD band
    ax.fill_between(
        [line_min, line_max],
        [line_min - std_dev, line_max - std_dev],
        [line_min + std_dev, line_max + std_dev],
        color='orange', alpha=0.2,
        label='±1 STD'
    )

    # axis limits tied to the **actual** data
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(p_min, p_max)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)

    # move legend out of the way
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.3)

    # metrics textbox
    ax.text(
        0.02, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    sns.despine()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()


def plot_true_vs_pred_with_percentiles(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        hue: Optional[np.ndarray] = None,
        palette: Optional[str] = "Reds",
        title: str = "Mean Response Predictions vs Targets",
        output_path: Optional[pathlib.Path] = None,
        bins: int = 20,
        percentile_bounds: Tuple[int, int] = (5, 95),
):
    def safe_bin_column(df, col='True', bins=20, min_count=4):
        # Compute bin edges across the column
        bin_edges = np.linspace(df[col].min(), df[col].max(), bins + 1)
        df['Bin'] = pd.cut(df[col], bins=bin_edges, include_lowest=True)

        # Remove bins with fewer than min_count samples
        bin_counts = df['Bin'].value_counts()
        valid_bins = bin_counts[bin_counts >= min_count].index

        # Filter DataFrame and handle empty case
        filtered_df = df[df['Bin'].isin(valid_bins)]
        if filtered_df.empty:
            print(f"Warning: No bins have at least {min_count} samples. Returning empty DataFrame.")
            return filtered_df, bin_edges

        return filtered_df, bin_edges

    # Main code
    df = pd.DataFrame({'True': y_true, 'Pred': y_pred})
    if hue is not None:
        hue_order = ['Normal', 'Mild', 'Moderate', 'Severe']
        df['Hue'] = pd.Categorical(hue, categories=hue_order, ordered=True)

    # Auto-adjust bins for small datasets
    if bins is None:
        bins = max(5, min(20, len(df) // 30))  # Reasonable bin count for small data

    # Apply safe binning
    df, bin_edges = safe_bin_column(df, col='True', bins=bins, min_count=4)

    # Check if DataFrame is empty after binning
    if df.empty:
        print("No valid bins found. Cannot compute bin statistics.")
        bin_stats = pd.DataFrame()  # Return empty DataFrame or handle as needed
    else:
        # Group and aggregate safely
        bin_stats = (
            df
            .groupby('Bin', observed=True)  # Only include bins with data
            .agg(
                true_mean=('True', 'mean'),
                pred_mean=('Pred', 'mean'),
                pred_p05=('Pred', lambda x: np.percentile(x, percentile_bounds[0])),
                pred_p95=('Pred', lambda x: np.percentile(x, percentile_bounds[1])),
            )
            .reset_index()
        )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter plot with hue
    if hue is not None:
        custom_palette = {
            'Normal': '#1f77b4',  # blue
            'Mild': '#ff7f0e',  # orange
            'Moderate': '#2ca02c',  # green
            'Severe': '#d62728',  # red
        }
        scatter = sns.scatterplot(
            data=df,
            x='True',
            y='Pred',
            hue='Hue',
            hue_order=hue_order,
            palette=custom_palette,
            edgecolor=None,
            s=20,
            alpha=0.3,
            ax=ax,
        )
    else:
        ax.scatter(df['True'], df['Pred'], color='blue', s=10, alpha=0.3, label='True vs Pred')

    # 5th–95th percentile bounds
    ax.plot(
        bin_stats['true_mean'], bin_stats['pred_p05'],
        linestyle='--', color='black', linewidth=1,
        label=f'{percentile_bounds[0]}th–{percentile_bounds[1]}th quantile'
    )
    ax.plot(
        bin_stats['true_mean'], bin_stats['pred_p95'],
        linestyle='--', color='black', linewidth=1
    )

    # Mean line
    ax.plot(
        bin_stats['true_mean'], bin_stats['pred_mean'],
        'o-', color='black', linewidth=2, markersize=5, label='Mean true vs prediction'
    )

    ax.set_title(title)
    ax.set_xlabel("True AHI")
    ax.set_ylabel("Predicted AHI")
    ax.grid(True, linestyle='--', alpha=0.3)

    # Full legend: move below plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
        frameon=True,
        fontsize=10
    )

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()



def plot_true_pred_histograms_stacked(df_eval: pd.DataFrame,
                                      bins: int = 40,
                                      output_path:pathlib.Path = None):
    """
    Generates and displays stacked histograms comparing the distributions of true and predicted
    values across different categories of a data set. The histograms are visually enhanced with
    statistical markers such as mean, median, and standard deviations to allow for insightful
    analysis. The function specifically supports stacking based on a 'hue' column in the data
    frame for categorical differentiation.

    :param df_eval: Input data frame containing the columns 'True', 'Pred', and 'hue'. The 'True'
        and 'Pred' columns must contain numerical values, while the 'hue' column
        should have categorical labels to segment the data.
    :type df_eval: pandas.DataFrame
    :param bins: Number of bins to use in the histograms. Default is 40.
    :type bins: int
    :return: None
    """
    # x_max = df_eval[['True', 'Pred']].max().max()

    df_eval = df_eval.sort_values(by='True', ascending=True)

    if df_eval['hue'].nunique() == 4:
        hue_order = ['Normal', 'Mild', 'Moderate', 'Severe']
    else:
        hue_order = ['No OSA', 'OSA']

    palette = sns.color_palette('Set1', n_colors=len(hue_order))

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=False)
    for ax, col in zip(axes, ['True', 'Pred']):
        sns.histplot(
            data=df_eval,
            x=col,
            hue='hue',
            hue_order=hue_order,
            palette=palette,
            bins=bins,
            multiple='stack',
            stat='probability',
            alpha=0.7,
            edgecolor=None,
            ax=ax,
            legend=False
        )

        # Stats & verticals
        mean = df_eval[col].mean()
        median = df_eval[col].median()
        std = df_eval[col].std()
        x_max = df_eval[col].max()

        # title
        counts = df_eval['hue'].value_counts().to_dict()
        # Format as string: "Normal: 6591 | Mild: 6591 | Moderate: 6591 | Severe: 6591"
        title_suffix = ", ".join([f"{k}: {v}" for k, v in sorted(counts.items())])


        ax.axvline(mean, color='black', linestyle='--', linewidth=1.5, label='Mean')
        ax.axvline(median, color='lightblue', linestyle='-.', linewidth=1.5, label='Median')
        ax.axvline(mean + 3 * std, color='orange', linestyle=':', linewidth=1.2, label='+3 STD')
        ax.axvline(mean - 3 * std, color='gray', linestyle=':', linewidth=1.2)
        ax.axvline(mean + 4 * std, color='red', linestyle=':', linewidth=1.2, label='+4 STD')
        ax.axvline(mean - 4 * std, color='darkred', linestyle=':', linewidth=1.2)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title(f"{col} \n {title_suffix}", fontsize=20)
        ax.set_ylabel("Proportion")
        ax.set_xlim([0, x_max * 1.05])

        # Only bottom plot gets x-label
        # if col == 'Pred':
        #     ax.set_xlabel("AHI")
        # else:
        #     ax.set_xlabel("")

    # axes[-1].set_xlabel(col)
    # Shared hue legend
    hue_handles = [Patch(facecolor=palette[i], label=label) for i, label in enumerate(hue_order)]

    # 2. Get stat handles from bottom ax
    stat_handles, stat_labels = axes[1].get_legend_handles_labels()

    # 3. Combine
    all_handles = hue_handles + stat_handles
    all_labels = hue_order + stat_labels

    # 4. Add combined legend
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.01),
        ncol=4,
        frameon=True
    )


    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

# %% Metrics classifier

def compute_classifier_metrics(preds_df: pd.DataFrame,
                               label_dict: dict = None,
                               output_dir: Optional[pathlib.Path] = None) -> pd.DataFrame:
    """
    Computes classification metrics for each true/pred column pair in preds_df.
    Handles both binary and multiclass classification and supports label formatting.

    Parameters:
    - preds_df: DataFrame with columns like 'train_true', 'train_pred', ...
    - label_dict: Optional dictionary to map numeric labels to readable names (e.g., {0: "No", 1: "Yes"})
    - output_dir: Optional path to save CSVs

    Returns:
    - metrics_df: DataFrame with detailed metrics per split
    - summary_df: DataFrame with metrics_{mean,std} per group
    """
    records = []
    true_cols = [c for c in preds_df.columns if c.endswith('_true') and '_index' not in c]

    for true_col in true_cols:
        pred_col = true_col.replace('_true', '_pred')
        group = true_col.split('_')[0]

        y_true = preds_df[true_col].dropna().values
        y_pred = preds_df[pred_col].dropna().values

        # Convert predictions to class indices if needed
        if isinstance(y_pred[0], (list, np.ndarray)):
            # Multi-class softprob output
            y_pred = np.array([np.argmax(p) for p in y_pred])
        elif np.issubdtype(y_pred.dtype, np.floating) and np.all((y_pred >= 0) & (y_pred <= 1)):
            # Binary probability output — apply threshold
            y_pred = (y_pred > 0.5).astype(int)


        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        is_binary = len(unique_classes) == 2
        average = 'binary' if is_binary else 'macro'

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average=average, zero_division=0)
        rec = recall_score(y_true, y_pred, average=average, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average, zero_division=0)

        # Probabilities (optional)
        prob_col = pred_col.replace('_pred', '_prob')
        has_probs = prob_col in preds_df.columns
        auroc = logloss = np.nan

        if has_probs and is_binary:
            y_prob = preds_df[prob_col].dropna().values
            if len(np.unique(y_true)) > 1:
                auroc = roc_auc_score(y_true, y_prob)
            logloss = log_loss(y_true, y_prob, labels=[0, 1])
        elif has_probs:
            y_prob = np.vstack(preds_df[prob_col].dropna().values)
            if y_prob.shape[1] == len(unique_classes):
                logloss = log_loss(y_true, y_prob, labels=unique_classes)

        # Global metrics row
        records.append({
            'group': group,
            'split': true_col,
            'class_label': 'global',
            'accuracy': round(acc, 3),
            'precision': round(prec, 3),
            'recall': round(rec, 3),
            'f1_score': round(f1, 3),
            'auroc': round(auroc, 3) if has_probs and is_binary else np.nan,
            'log_loss': round(logloss, 3) if has_probs else np.nan,
            'tp': np.nan,
            'fp': np.nan,
            'fn': np.nan,
            'tn': np.nan,
        })

        # Per-class metrics (1-vs-rest)
        for cls in unique_classes:
            binary_true = (y_true == cls).astype(int)
            binary_pred = (y_pred == cls).astype(int)

            cls_prec = precision_score(binary_true, binary_pred, zero_division=0)
            cls_rec = recall_score(binary_true, binary_pred, zero_division=0)
            cls_f1 = f1_score(binary_true, binary_pred, zero_division=0)

            # Compute confusion matrix manually
            tn, fp, fn, tp = confusion_matrix(binary_true, binary_pred, labels=[0, 1]).ravel()

            records.append({
                'group': group,
                'split': true_col,
                'class_label': label_dict.get(cls, cls) if label_dict else cls,
                'accuracy': np.nan,
                'precision': round(cls_prec, 3),
                'recall': round(cls_rec, 3),
                'f1_score': round(cls_f1, 3),
                'auroc': np.nan,
                'log_loss': np.nan,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
            })

    metrics_df = pd.DataFrame(records)
    metrics_df.dropna(how='all', axis=1, inplace=True)


    # Summary
    # agg_dict = {col: ['mean', 'std'] for col in metrics_df.columns if col not in ('group', 'split')}
    # summary = metrics_df.groupby('group').agg(agg_dict)
    # summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    # summary_df = summary.reset_index()

    # Optional output
    if output_dir:
        metrics_df.to_csv(output_dir.joinpath('classifier_metrics_df.csv'), index=False)
        # summary_df.to_csv(output_dir.joinpath('classifier_summary_df.csv'), index=False)

    # Optional label formatting
    if label_dict:
        preds_df = preds_df.replace(label_dict)

    return metrics_df #  summary_df




def plot_confusion_and_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray = None,
    label_dict: Optional[dict] = None,
    output_path: Optional[pathlib.Path] = None,
    title_prefix: str = "",
    split_name: str = "",
    figsize: Tuple[int, int] = (10, 4)
):
    """
    Plot confusion matrix with both counts and percentages.
    Plot ROC curve and overlay diagnostic metrics if binary classification and probabilities are available.
    """
    # Convert predicted probs to class labels if needed
    if isinstance(y_pred[0], (list, np.ndarray)):
        y_pred = np.array([np.argmax(p) for p in y_pred])

    # Extract probability of positive class for binary AUC
    if y_prob is not None:
        if isinstance(y_prob[0], (list, np.ndarray)):
            y_prob = np.array([p[1] if len(p) > 1 else p[0] for p in y_prob])
        else:
            y_prob = np.array(y_prob)

    # Setup label names
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    label_names = [label_dict.get(lbl, lbl) for lbl in unique_labels] if label_dict else unique_labels

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    # Annotate with both count and %
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_percent[i, j]:.1f}%)"

    # Determine if we should include ROC/AUC
    is_binary = len(unique_labels) == 2 and y_prob is not None

    fig, axes = plt.subplots(1, 2 if is_binary else 1, figsize=figsize)

    ax_cm = axes[0] if is_binary else axes
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names,
                cbar=False, ax=ax_cm)

    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    ax_cm.set_title(f"{title_prefix} ({split_name})")

    # ROC + diagnostic metrics (only for binary)
    if is_binary:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)

        # Calculate diagnostics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)  # same as sensitivity
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        f1 = f1_score(y_true, y_pred)

        ax = axes[1]
        ax.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_title(f"{title_prefix} ({split_name})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(True, alpha=0.7)

        # Place the line legend top left
        # ax.legend(
        #     loc='upper left',  # reference corner
        #     bbox_to_anchor=(0.5, -0.03),  # x=0.6, y=0.95 in axis coordinates
        #     fontsize=9
        # )
        # Place the metrics box separately (bottom right)
        textstr = (
            r"$\bf{AUC}$" + f"        = {auc_score:.2f}\n"
            f"Accuracy    = {accuracy:.2f}\n"
            f"Sensitivity = {recall:.2f}\n"
            f"Specificity = {specificity:.2f}\n"
            f"Precision   = {precision:.2f}\n"
            f"F1 Score    = {f1:.2f}"
        )
        ax.text(0.95, 0.05,
                textstr,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # metrics_label = (
        #     f"AUC = {auc_score:.3f}\n"
        #     f"Accuracy = {accuracy:.2f}, Sensitivity = {recall:.2f},\n"
        #     f"Specificity = {specificity:.2f}, F1 = {f1:.2f}"
        # )

        # # Single legend call
        # ax.plot(fpr, tpr)
        # ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        # ax.legend([metrics_label], loc="lower right")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()



# %% XGBoost feature importance
def plot_xgb_feature_imp(fi_df,
                         ncol: int = 2,
                         height_prop: float = 0.8,
                         width: int = 14,
                         show:bool=True,
                         top_n: Optional[int] = None,
                         output_path=None):
    """
    Plots weight, gain, and cover importances in one figure, but only the top_n
    features for each importance type. Legend is limited to exactly those plotted.
    """

    def create_unique_alias(name: str, existing_aliases: set, max_length: int = 10,
                            exclude_words: list = ['have']) -> str:
        """
        Generate a unique alias for a feature name, avoiding duplicates.

        Args:
            name (str): Original feature name (e.g., 'user_age_group').
            existing_aliases (set): Set of already used aliases to ensure uniqueness.
            max_length (int): Maximum length of the alias.
            exclude_words (list): Words to exclude from alias generation.

        Returns:
            str: A unique alias.
        """
        # Split the name by underscores and filter out excluded words
        parts = [p for p in name.split('_') if p.lower() not in exclude_words]

        if not parts:
            alias = 'F'  # Fallback for empty parts after filtering
        else:
            # Take first character of each part, or full first part if only one part
            if len(parts) == 1:
                alias = parts[0][:max_length].capitalize()
            else:
                alias = ''.join(p[0].upper() for p in parts)[:max_length]

        # Handle duplicates by appending a number
        base_alias = alias
        counter = 1
        while alias in existing_aliases:
            alias = f"{base_alias}{counter}"[:max_length]
            counter += 1

        existing_aliases.add(alias)
        return alias

    def create_alias(name):
        parts = name.split('_')
        if len(parts) > 1:
            parts = parts[1:]
        return ''.join(p[0].upper() for p in parts if p.lower() != 'have')


    # --- ensure aliases exist ---
    if 'alias' not in fi_df.columns:
        existing_aliases = set()  # Track used aliases
        fi_df['alias'] = fi_df['feature'].apply(
            lambda x: create_unique_alias(x, existing_aliases, max_length=10, exclude_words=['have'])
        )
        # fi_df['alias'] = fi_df['feature'].apply(create_alias)

    importance_types = ['weight', 'gain', 'cover']
    all_features = fi_df['feature'].tolist()
    alias_map = dict(zip(fi_df['feature'], fi_df['alias']))

    # Pre-compute a colormap for all features (we’ll only use a subset)
    n_feats = len(all_features)
    cmap = plt.get_cmap('tab20', max(n_feats, 20))
    feature_colors = {f: cmap(i) for i, f in enumerate(all_features)}

    # We'll collect which features actually get plotted
    plotted_feats = set()

    # Figure height should reflect how many bars we're drawing per subplot
    if not top_n:
        top_n = fi_df.shape[0]
    plot_n = min(n_feats, top_n)
    fig, axes = plt.subplots(
        1, len(importance_types),
        figsize=(width, max(6, height_prop * plot_n)),
        constrained_layout=False
    )

    # --- plot each importance, but only top_n rows ---
    for ax, imp in zip(axes, importance_types):
        mean_col, std_col = f"{imp}_mean", f"{imp}_std"
        if not mean_col in fi_df.columns:
            mean_col = imp
            std_col = None
        sorted_df = fi_df.sort_values(mean_col, ascending=False).reset_index(drop=True)
        sorted_df = sorted_df.head(top_n)  # <-- limit to top_n
        plotted_feats.update(sorted_df['feature'])

        y = list(range(len(sorted_df)))
        colors = [feature_colors[f] for f in sorted_df['feature']]
        ax.barh(
            y, sorted_df[mean_col],
            xerr=sorted_df[std_col] if std_col else None,
            color=colors, edgecolor='black', capsize=3
        )
        ax.grid(axis='x', alpha=0.7)
        ax.set_title(f"{imp.capitalize()} Importance")
        ax.set_yticks(y)
        ax.set_yticklabels(sorted_df['alias'])
        ax.invert_yaxis()

    # --- build a legend only for the plotted features, sorted by alias ---
    legend_pairs = sorted(
        ((alias_map[f], f) for f in plotted_feats),
        key=lambda x: x[0]
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=feature_colors[f])
        for a, f in legend_pairs
    ]
    labels = [f"({a}) {f}" for a, f in legend_pairs]

    if not ncol:
        ncol = max(1, min(4, len(legend_pairs) // 5 + 1))

    # Create a new figure just for the legend with adaptive height and width
    n_labels = len(labels)
    ncol = max(1, min(4, n_labels // 5 + 1))
    nrows = math.ceil(n_labels / ncol)
    col_width = 6.0  # e.g. 2" per column
    row_height = 0.4  # e.g. 0.4" per row
    extra_height = 0.6
    fig_w = ncol * col_width
    fig_h = nrows * row_height + extra_height


    leg_fig = plt.figure(figsize=(fig_w, fig_h))
    leg_ax = leg_fig.add_subplot(111)
    leg_ax.axis('off')  # no axes, just legend

    # Draw the legend in the center of this new figure
    leg = leg_ax.legend(
        handles, labels,
        title="Feature (Alias)",
        ncol=ncol,
        frameon=False,
        fontsize='small',
        loc='center'
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path.joinpath('feature_imp.png'), bbox_inches="tight", dpi=300)
        leg_fig.savefig(
            output_path.joinpath('feature_imp_legend.png'),
            bbox_inches='tight',
            dpi=300
        )
    if show:
        plt.show()
    plt.close(leg_fig)
    plt.close(fig)

def plot_xgb_hue_by(fi_df,
                    hue_by: str,
                    importance_types: list = None,
                    top_n: int = 20,
                    width: int = 14,
                    height_prop: float = 0.8,
                    palette: str = 'tab20',
                    output_path: str = None):
    """
    Plots XGBoost feature importances, colouring bars by the values in `hue_by`.
    Only the top_n features (by each importance) are shown.

    Parameters
    ----------
    fi_df : pd.DataFrame
        Must contain columns:
          - 'feature', 'alias',
          - '<imp>_mean', '<imp>_std' for imp in importance_types
          - plus the `hue_by` column to colour by.
    hue_by : str
        Name of the column in fi_df to use for bar colours (e.g. 'alias').
    importance_types : list of str, optional
        Which importances to plot. Defaults to ['weight','gain','cover'].
    top_n : int, optional
        How many features to show per subplot (default 20).
    width : int, optional
        Figure width (default 14).
    height_prop : float, optional
        Multiplier for bar-count → figure height (default 0.8).
    palette : str, optional
        A Matplotlib colormap name (default 'tab20').
    output_path : str, optional
        If given, saves to `{output_path}.png`.
    """
    # ensure alias exists
    if 'alias' not in fi_df.columns:
        def _mk_alias(n):
            parts = n.split('_')
            if len(parts)>1: parts = parts[1:]
            return ''.join(p[0].upper() for p in parts if p.lower()!='have')
        fi_df = fi_df.copy()
        fi_df['alias'] = fi_df['feature'].map(_mk_alias)

    if importance_types is None:
        importance_types = ['weight','gain','cover']

    # build palette for all unique hue values
    hues = fi_df[hue_by].astype(str)
    uniq = list(dict.fromkeys(hues))           # preserve order
    cmap = plt.get_cmap(palette, max(len(uniq),1))
    hue_colors = {h: cmap(i) for i,h in enumerate(uniq)}

    # figure sizing based on top_n
    plot_n = min(len(fi_df), top_n)
    fig, axes = plt.subplots(
        1, len(importance_types),
        figsize=(width, max(6, height_prop * plot_n)),
        constrained_layout=False
    )

    # track which hue categories actually got plotted
    plotted_hues = set()

    for ax, imp in zip(axes, importance_types):
        mcol, scol = f"{imp}_mean", f"{imp}_std"
        sub = (fi_df
               .sort_values(mcol, ascending=False)
               .head(top_n)
               .reset_index(drop=True))
        plotted_hues.update(sub[hue_by].astype(str))

        ys = list(range(len(sub)))
        cols = [hue_colors[str(x)] for x in sub[hue_by]]
        ax.barh(ys, sub[mcol], xerr=sub[scol],
                color=cols, edgecolor='black', capsize=3)
        ax.grid(axis='x', alpha=0.7)
        ax.set_title(f"{imp.capitalize()} Importance")
        ax.set_yticks(ys)
        ax.set_yticklabels(sub['alias'])
        ax.invert_yaxis()

    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    # legend for only the hues we plotted
    legend_list = sorted(plotted_hues)
    handles = [plt.Rectangle((0,0),1,1, color=hue_colors[h])
               for h in legend_list]
    labels  = legend_list

    ncol = max(1, min(4, len(legend_list)//5 + 1))
    fig.legend(
        handles, labels,
        title=hue_by.title(),
        loc="lower center",
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        fontsize='small'
    )

    if output_path:
        fig.savefig(output_path + ".png", bbox_inches="tight", dpi=300)
    plt.show()


# %% Fusion methods
from sklearn.linear_model import LinearRegression

def train_late_fusion(
    df: pd.DataFrame,
    target_col: str,
    sections: list[str],
    dem_cols: list[str] = ["age", "bmi", "gender", "race"],
    fusion_strategies: list[str] = ["mean", "mse_weighted", "r2_weighted", "stacking"],
    optimization: bool = True,
    n_trials: int = 50,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    use_gpu: bool = True,
    stratify_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict], LinearRegression | None]:
    """
    Perform late-fusion of block-wise XGBoost regressors optimized via Optuna.

    Returns:
      - ensemble_df: DataFrame indexed by test set, with columns:
          * 'true'
          * one column per fusion strategy in fusion_strategies
      - block_results: dict mapping block_name -> {
            'model': trained xgb.Booster,
            'best_params': dict,
            'mse': float,
            'r2': float,
            'preds': np.ndarray
        }
      - meta_model: fitted LinearRegression (only if 'stacking' in fusion_strategies)
    """
    # 1. Identify feature blocks
    all_feats = [c for c in df.columns if c != target_col]
    blocks = {}
    dem_group = [c for c in dem_cols if c in all_feats]
    if dem_group:
        blocks["demographics"] = dem_group
    for pref in sections:
        group = [c for c in all_feats if c.startswith(pref) and c not in dem_group]
        if group:
            blocks[pref] = group

    # 2. Train/Test split
    stratify_vals = df[stratify_col] if stratify_col else None
    split_args = dict(test_size=test_size, random_state=random_state)
    if stratify_col:
        split_args["stratify"] = stratify_vals.values
    X = df[all_feats]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_args)

    # 3. Train each block model
    block_results = {}
    for name, cols in blocks.items():
        # 3a. Optuna tuning
        def objective(trial):
            params = {
                "num_boost_round": trial.suggest_int("num_boost_round", 50, 500),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "objective": "reg:squarederror",
                "random_state": random_state,
            }
            if use_gpu:
                params.update(tree_method="hist", device="cuda")

            splitter = (
                StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
                if stratify_col
                else KFold(n_splits=3, shuffle=True, random_state=random_state)
            )

            rmses = []
            X_arr = X_train[cols].values
            y_arr = y_train.values
            for tr_idx, val_idx in splitter.split(X_arr, stratify_vals.loc[y_train.index].values if stratify_col else y_arr):
                dtr = xgb.DMatrix(X_arr[tr_idx], label=y_arr[tr_idx])
                dval = xgb.DMatrix(X_arr[val_idx], label=y_arr[val_idx])
                bst = xgb.train(
                    params,
                    dtr,
                    num_boost_round=params["num_boost_round"],
                    evals=[(dtr, "train"), (dval, "valid")],
                    early_stopping_rounds=10,
                    verbose_eval=False,
                )

                preds = bst.predict(dval)
                rmses.append(np.sqrt(mean_squared_error(y_arr[val_idx], preds)))
            return float(np.mean(rmses))

        if optimization:
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_params.update(objective="reg:squarederror", random_state=random_state)
            if use_gpu:
                best_params.update(tree_method="hist", device="cuda")
        else:
            best_params = {
                "num_boost_round": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "objective": "reg:squarederror",
                "random_state": random_state,
            }
            if use_gpu:
                best_params.update(tree_method="hist", device="cuda")

        # 3b. Final train on full training set
        dtrain = xgb.DMatrix(X_train[cols], label=y_train.values)
        bst_final = xgb.train(best_params, dtrain, num_boost_round=best_params["num_boost_round"])

        # 3c. Predict on test set
        dtest = xgb.DMatrix(X_test[cols])
        y_pred = bst_final.predict(dtest)

        # 3d. Metrics
        mse = float(mean_squared_error(y_test, y_pred))
        r2  = float(r2_score(y_test, y_pred))

        block_results[name] = {
            "model": bst_final,
            "best_params": best_params,
            "mse": mse,
            "r2":  r2,
            "preds": y_pred,
        }

    # 4. Assemble test predictions
    test_index = X_test.index
    preds_matrix = np.column_stack([block_results[b]["preds"] for b in block_results])
    true_vals = y_test.values

    # 5. Compute fusion strategies
    ensemble = {"true": true_vals}
    if "mean" in fusion_strategies:
        ensemble["mean"] = preds_matrix.mean(axis=1)
    if "mse_weighted" in fusion_strategies:
        inv_mse = np.array([1.0 / block_results[b]["mse"] for b in block_results])
        weights = inv_mse / inv_mse.sum()
        ensemble["mse_weighted"] = preds_matrix.dot(weights)
    if "r2_weighted" in fusion_strategies:
        scores = np.array([block_results[b]["r2"] for b in block_results])
        positive = np.clip(scores, a_min=0, a_max=None)
        weights = positive / positive.sum() if positive.sum() > 0 else np.ones_like(positive) / len(positive)
        ensemble["r2_weighted"] = preds_matrix.dot(weights)

    meta_model = None
    if "stacking" in fusion_strategies:
        oof_preds = np.column_stack([
            block_results[b]["model"].predict(xgb.DMatrix(X_train[blocks[b]]))
            for b in blocks
        ])
        meta_model = LinearRegression().fit(oof_preds, y_train.values)
        ensemble["stacking"] = meta_model.predict(preds_matrix)

    # 6. Return DataFrame and models
    ensemble_df = pd.DataFrame(ensemble, index=test_index)
    return ensemble_df, block_results, meta_model


# %% Model loading
def _load_xgb_model(model_path: str) -> xgb.Booster:
    """
    Load a single XGBoost model from disk.

    Parameters:
    - model_path: path to the .model file

    Returns:
    - Loaded xgb.Booster object
    """
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def _load_cv_models(model_dir: str, pattern: str = "cv_fold_{}.model") -> List[xgb.Booster]:
    """
    Load all cross-validation fold models from a directory.

    Parameters:
    - model_dir: directory where CV models are saved
    - pattern: filename pattern with one placeholder for fold number

    Returns:
    - List of xgb.Booster objects in fold order
    """
    boosters = []
    i = 1
    while True:
        path = os.path.join(model_dir, pattern.format(i))
        if not os.path.exists(path):
            break
        bst = xgb.Booster(); bst.load_model(path)
        boosters.append(bst)
        i += 1
    return boosters


def load_all_models(model_dir: str) -> Tuple[List[xgb.Booster], xgb.Booster]:
    """
    Load CV fold models and the final model from a directory.

    Parameters:
    - model_dir: directory containing 'cv_fold_#.model' and 'final.model'

    Returns:
    - Tuple of (list of CV boosters, final booster)
    """
    cv_boosters = _load_cv_models(model_dir)
    final_path = os.path.join(model_dir, "final.model")
    final_booster = _load_xgb_model(final_path)
    return cv_boosters, final_booster

