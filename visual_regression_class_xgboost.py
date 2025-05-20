import pathlib
from typing import  Optional, List, Union, Tuple
import numpy as np
import json
import pickle
from config.config import config, metrics_psg, encoding, sections
import pandas  as pd
import statsmodels.api as sm
from library.ml_tabular_data.my_simple_xgb import (train_xgb_collect_defined_folds_wrapper,
                                                   train_regressors,
                                                   stack_regressors_and_data,
                                                   train_xgb_collect_wrapper)
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import os

def compare_model_metrics(root_dir: pathlib.Path,
                          plot: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collects all classifier and regressor metrics from the specified results directory.
    Optionally generates comparison plots.

    Parameters:
    - root_dir: pathlib.Path to the main results folder (e.g., "regres_classif_xgboost")
    - plot: whether to generate comparative subplots

    Returns:
    - df_classifiers: long-format DataFrame with classifier metrics
    - df_regressors: DataFrame with regressor metrics
    """
    classifier_dfs = []
    regressor_dfs = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = pathlib.Path(subdir) / file

            # CLASSIFIER
            if file == "classifier_metrics_df.csv":
                df = pd.read_csv(file_path)
                df["model"] = file_path.parent.name
                df = df[df["class_label"] == "global"]  # keep global summary rows only
                classifier_dfs.append(df)

            # REGRESSOR
            elif file == "metrics_df.csv":
                df = pd.read_csv(file_path)
                df["model"] = file_path.parent.name
                regressor_dfs.append(df)

    df_classifiers = pd.concat(classifier_dfs, ignore_index=True) if classifier_dfs else pd.DataFrame()
    df_regressors = pd.concat(regressor_dfs, ignore_index=True) if regressor_dfs else pd.DataFrame()
    metrics_reg = ['rmse', 'medae', 'explained_variance', 'r2_adj']
    df_regressors[metrics_reg] = df_regressors[metrics_reg].round(3)

    if plot:
        # ── Classifier Plot ──
        if not df_classifiers.empty:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            for ax, metric in zip(axes.flat, metrics):
                sns.barplot(data=df_classifiers[df_classifiers['split'] == 'test_true'],
                            x='model', y=metric, ax=ax)
                ax.set_title(f"Classifier Test {metric.capitalize()}")
                ax.set_ylabel(metric.capitalize())
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

        # ── Regressor Plot ──
        if not df_regressors.empty:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            metrics = ['rmse', 'medae', 'explained_variance', 'r2_adj']
            for ax, metric in zip(axes.flat, metrics):
                sns.barplot(data=df_regressors[df_regressors['split'] == 'test_true'],
                            x='model', y=metric, ax=ax)
                ax.set_title(f"Regressor Test {metric.replace('_', ' ').title()}")
                ax.set_ylabel(metric)
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()
            plt.show()

    return df_classifiers, df_regressors

# %%
if __name__ == '__main__':
    # %% Input data

    # %% output directory
    path_dir = config.get('results')['dir'].joinpath('regres_classif_xgboost')
    # %%
    df_classifiers, df_regressors = compare_model_metrics(root_dir=path_dir, plot=True)















