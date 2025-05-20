"""
====================================================================
Train and Stack XGBoost Regressors and Classifiers for OSA Severity
====================================================================

This script performs the following tasks:
1. Loads preprocessed PSG data (optionally PCA-reduced).
2. Trains multiple XGBoost regressors on respiratory event indices.
3. Stores and plots regression results, including error metrics.
4. Builds stacked feature representations from OOF predictions.
5. Trains multi-class and binary classifiers (e.g., AHI-based OSA severity).
6. Optionally trains a final XGBoost regressor for the composite AHI index.

Modules:
- XGBoost training and Optuna tuning (regression/classification)
- Visualizations for predictions, feature importances, histograms
- Stacking logic for downstream model input preparation
- Group-level metric comparison and summary plots

Inputs:
- CSV data (PCA or feature-expanded)
- Config file with paths and experiment metadata

Outputs:
- Trained model artifacts (pickles, plots, metrics)
- Stacked datasets for classification
- Model comparison tables

Dependencies:
- numpy, pandas, matplotlib, seaborn
- xgboost, optuna, scikit-learn
- statsmodels, scipy
- Custom modules from `library.ml_tabular_data`

Usage:
- Run this script directly (`__main__`) to generate models and outputs.
- Configuration is controlled via the `config` object from `config/config.py`.

Author: Giorgio Ricciardiello
"""

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

# %%
if __name__ == '__main__':
    # %% Input data
    PCA_reduced_dim = True
    N_BOOSTING_ROUNDS = 1500
    N_TRIALS = 45
    log1p = True
    stratify_col = 'osa_four'
    target_ahi_regressor = 'ahi_log1p'

    # %% Data to read
    if PCA_reduced_dim:
        df_data = pd.read_csv(config.get('data')['pp_data']['pca_reduced'], low_memory=False)
    else:
        df_data = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)

    print(f'Dataset dimensions {df_data.shape}')
    # %% output directory
    path_dir = config.get('results')['dir'].joinpath('regres_classif_xgboost')
    path_dir.mkdir(parents=True, exist_ok=True)
    path_regressors = path_dir.joinpath('xgb_regressors.pkl')
    # %% define the output
    t_col = 'resp-position-total'
    df_data['sleep_hours'] = df_data[t_col] / 60.0

    section_features = [sect for sect in sections if not sect in ['presleep_', 'resp']]
    # 1. Define your sections and columns
    features = [col for col in df_data.columns
                    if any(col.startswith(alias) for alias in section_features)]
    features.append('sleep_hours')
    features = list(set(features))

    resp_measures = ['ahi',
                    'resp-oa-total',
                     'resp-ca-total',
                     'resp-ma-total',
                     'resp-hi_hypopneas_only-total',
                     'resp-ri_rera_only-total']

    # resp_measures_idx = metrics_psg.get('resp_events')['indices']
    df_data.loc[df_data['sleep_hours'] < 2, 'sleep_hours'] = 2
    df_data = df_data.dropna(subset=resp_measures)

    # Compute index values
    resp_measures_idx = []
    for resp in resp_measures:
        if resp == 'ahi':
            # ahi is already an index so we do not need to compute it
            resp_measures_idx.append(resp)
            continue
        col_idx = f'{resp}_idx'
        df_data[col_idx] = df_data[resp] / df_data['sleep_hours']
        resp_measures_idx.append(col_idx)
    #
    # apply the log1p to all the respiratory index measures
    if log1p:
        updated_idx = []
        for resp in resp_measures_idx:

            new_col = f'{resp}_log1p'
            df_data[new_col] = np.log1p(df_data[resp])
            print(f'{df_data[[resp, new_col]].describe()}\n')
            updated_idx.append(new_col)

        resp_measures_idx = updated_idx

    print(f"{df_data[['ahi', 'ahi_log1p']].describe()}\n")


    # for col in features: print(col)

    # %% formal names to target
    resp_measures_idx_formal = {'resp-oa-total_idx_log1p': 'OA/#h',
                                'resp-ca-total_idx_log1p': 'CA/#h',
                                'resp-ma-total_idx_log1p': 'MA/#h',
                                'resp-hi_hypopneas_only-total_idx_log1p': 'HYP/#h',
                                'resp-ri_rera_only-total_idx_log1p': 'RERA/#h',
                                #
                                # 'resp-oa-total_log1p': 'OA',
                                # 'resp-ca-total_log1p': 'CA',
                                # 'resp-ma-total_log1p': 'MA',
                                # 'resp-hi_hypopneas_only-total_log1p': 'HYP',
                                # 'resp-ri_rera_only-total_log1p': 'RERA',
                                }
    # assert len(resp_measures_idx_formal) == len(resp_measures_idx)
    # df_data.rename(columns=resp_measures_idx_formal, inplace=True)
    # %%
    target = resp_measures[0]

    assert set(resp_measures_idx).issubset(
        df_data.columns), f"Missing columns in df_data: {set(resp_measures_idx) - set(df_data.columns)}"
    # df_data[[*resp_measures_idx_formal.keys()]]
    # %% regression model
    if path_regressors.is_file():
        with open(path_regressors, 'rb') as f:
            xgb_regressors = pickle.load(f)
    else:
        xgb_regressors = {}
        for target, tgt_lbl in resp_measures_idx_formal.items():
            xgb_regressors[target] = train_regressors(data=df_data,
                                                     target=target,
                                                     tgt_lbl=tgt_lbl,
                                                     features=features,
                                                      n_trials=N_TRIALS,
                                                      num_boost_round=N_BOOSTING_ROUNDS,
                                                     stratify_col=stratify_col,
                                                     dir_path=path_dir)

        with open(path_dir.joinpath('xgb_regressors.pkl'), 'wb') as f:
            pickle.dump(xgb_regressors, f)

    # %% stack results for final classifier or regressor
    df_raw = df_data.copy()

    col_base = [stratify_col, 'dem_age', 'dem_gender', 'dem_bmi', 'dem_race', 'sleep_hours']
    X_train_stack, X_val_stack, X_test_stack = stack_regressors_and_data(xgb_regressors=xgb_regressors,
                                                                              df_raw=df_raw,
                                                                              include_columns=col_base)

    # Define classifier and regressor targets
    target_ahi_classifier_multi = 'osa_four_numeric'
    target_ahi_classifier_binary_five = 'osa_binary_numeric'
    target_ahi_classifier_binary_fifteen = 'binary_fifteenth_numeric'

    target_ahi_classifier_list = [
        target_ahi_classifier_multi,
        target_ahi_classifier_binary_five,
        target_ahi_classifier_binary_fifteen,
    ]
    targets_ahi = target_ahi_classifier_list + [target_ahi_regressor]


    # Create datasets with proper target values
    y_train = df_raw.loc[X_train_stack.index, targets_ahi]
    y_val = df_raw.loc[X_val_stack.index, targets_ahi]
    y_test = df_raw.loc[X_test_stack.index, targets_ahi]

    df_train = pd.merge(X_train_stack, y_train, left_index=True, right_index=True, how='left')
    df_val = pd.merge(X_val_stack, y_val, left_index=True, right_index=True, how='left')
    df_test = pd.merge(X_test_stack, y_test, left_index=True, right_index=True, how='left')

    # %% compute final model
    # ======== TRAIN CLASSIFIER ========
    target_ahi_classifier_with_params = {
                                        'osa_binary_numeric':[
                                                            "binary:logistic",
                                                              # "binary:logitraw"
                                        ],
                                        'binary_fifteenth_numeric': [
                                            "binary:logistic",
                                                                     # "binary:logitraw"
                                                                     ],
                                        'osa_four_numeric': [
                                            "multi:softprob",
                                             # "multi:softmax",
                                             # "multi:softprob",
                                             # "rank:pairwise",
                                             # "rank:ndcg",
                                             # "rank:map"
                                            ],
                                        }

    # # Train models
    for target_classifier, objectives in target_ahi_classifier_with_params.items():
        name_thresh = target_classifier.split("_")[1]  # e.g., "four", "binary"
        for objective in objectives:
            print(f'---Training classifier {target_classifier} with objective {objective}---')
            name_loss = objective.split(":")[1]
            path_final_classifier = path_dir.joinpath(f'final_classifier_{name_thresh}_{name_loss}')
            path_final_classifier.mkdir(parents=True, exist_ok=True)
            features_classifier = [fet for fet in df_train.columns if not fet in targets_ahi]
            NUM_CLASSES = len(np.unique(y_train[target_classifier]))
            label_dict = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"} if NUM_CLASSES == 4 else {0: "No OSA",
                                                                                                        1: "OSA"}

            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(y_train[target_classifier]),
                                                 y=y_train[target_classifier])

            # Compute scale_pos_weight only if binary
            if NUM_CLASSES == 2:
                neg = np.sum(df_train[target_classifier] == 0)
                pos = np.sum(df_train[target_classifier] == 1)
                scale_pos_weight = neg / pos
            else:
                scale_pos_weight = None

            in_params_classifier = {
                'objective': objective,
                'eval_metric': 'mlogloss' if 'multi' in objective else 'logloss',
                'num_class': NUM_CLASSES if 'multi' in objective else None,
                'num_boost_round': N_BOOSTING_ROUNDS,
                'scale_pos_weight': scale_pos_weight if NUM_CLASSES == 2 else None,
                'tree_method': 'hist',
                'sampling_method': 'gradient_based',
            }

            # Clean out None values from params (e.g., num_class or scale_pos_weight for binary)
            in_params_classifier = {k: v for k, v in in_params_classifier.items() if v is not None}


            train_xgb_collect_defined_folds_wrapper(
                df_train=df_train,
                df_val=df_val,
                df_test=df_test,
                target_col=target_classifier,
                features=features_classifier,
                stratify_col=stratify_col,
                model_params=in_params_classifier,
                model_path=path_final_classifier,
                n_trials=N_TRIALS,
                model_type='classifier',
                label_dict=label_dict,
            )

    # # ======== TRAIN REGRESSOR ========
    path_final_regressor = path_dir.joinpath(f'final_regressor')
    features_regressor = [fet for fet in df_train.columns if not fet in targets_ahi]

    in_params_regressor = {
        'objective': 'reg:gamma',  # 'reg:squarederror',
        'eval_metric': "rmse",

        # 'objective': 'reg:tweedie',
        # 'tweedie_variance_power': 1.3,
        # 'eval_metric': "tweedie-nloglik@1.3",
        # 'gamma': 0.1,
        'max_bin': 512,
        'num_parallel_tree': 10,
        'early_stopping_rounds': 100,
        'num_boost_round': N_BOOSTING_ROUNDS
        # 'updater': 'coord_descent',
        # 'feature_selector': 'greedy'
    }

    train_xgb_collect_defined_folds_wrapper(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        target_col=target_ahi_regressor,
        features=features_regressor,
        stratify_col=stratify_col,
        model_params=in_params_regressor,
        model_type='regressor',
        model_path=path_final_regressor,
        n_trials=N_TRIALS
    )


    # %% Simpler models that predict the AHI and classifies it, no stacking
    # collect all the features avoid leakage
    sections_fet = [sect for sect in sections if not sect in 'resp']
    features = [col for col in df_data.columns if col.startswith(tuple(sections_fet))]
    features = [fet for fet in features if not fet in [stratify_col, target_ahi_regressor]]

    # # ======== TRAIN REGRESSOR ========
    in_params_regression = {
        'objective': 'reg:gamma',
        # 'tweedie_variance_power': 1.5,
        'eval_metric': "rmse",
        'max_bin': 256,
        'num_parallel_tree': 10,
        'early_stopping_rounds': 100,
        'num_boost_round': N_BOOSTING_ROUNDS
    }

    # for gamma dist
    df_model = df_data.copy()
    epsilon = 0.001
    df_model[target_ahi_regressor] = df_model[target_ahi_regressor] + epsilon

    train_xgb_collect_wrapper(
        in_params=in_params_regression,
        df_data=df_model,
        features=features,
        target_col=target_ahi_regressor,
        model_type='regressor',
        optimization=True,
        n_trials=N_TRIALS,
        val_size=0.3,
        test_size=0.1,
        stratify_col=stratify_col,
        model_path=path_dir.joinpath('base_regressor'),
        resample=True,
        use_gpu=True,
    )

    # # ======== TRAIN CLASSIFIER ========
    for target_classifier, objectives in target_ahi_classifier_with_params.items():
        name_thresh = target_classifier.split("_")[1]  # e.g., "four", "binary"
        for objective in objectives:
            print(f'---Training classifier {target_classifier} with objective {objective}---')
            name_loss = objective.split(":")[1]
            path_final_classifier = path_dir.joinpath(f'base_classifier_{name_thresh}_{name_loss}')
            path_final_classifier.mkdir(parents=True, exist_ok=True)
            NUM_CLASSES = len(np.unique(df_data[target_classifier]))
            label_dict = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"} if NUM_CLASSES == 4 else {0: "No OSA",
                                                                                                        1: "OSA"}

            class_weights = compute_class_weight(class_weight='balanced',
                                                 classes=np.unique(df_data[target_classifier]),
                                                 y=df_data[target_classifier])

            # Compute scale_pos_weight only if binary
            if NUM_CLASSES == 2:
                neg = np.sum(df_data[target_classifier] == 0)
                pos = np.sum(df_data[target_classifier] == 1)
                scale_pos_weight = neg / pos
            else:
                scale_pos_weight = None

            in_params_classifier = {
                'objective': objective,
                'eval_metric': 'mlogloss' if 'multi' in objective else 'logloss',
                'num_class': NUM_CLASSES if 'multi' in objective else None,
                'num_boost_round': N_BOOSTING_ROUNDS,
                'scale_pos_weight': scale_pos_weight if NUM_CLASSES == 2 else None,
                'tree_method': 'hist',
                'sampling_method': 'gradient_based',
            }


            train_xgb_collect_wrapper(
                in_params=in_params_classifier,
                df_data=df_data,
                features=features,
                target_col=target_classifier,
                model_type='classifier',
                optimization=True,
                n_trials=N_TRIALS,
                val_size=0.3,
                test_size=0.1,
                stratify_col=stratify_col,
                model_path=path_final_classifier,
                resample=True,
                use_gpu=True,
                label_dict=label_dict
            )













