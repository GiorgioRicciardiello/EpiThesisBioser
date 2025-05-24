
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
    N_TRIALS = 28
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
    path_dir = config.get('results')['dir'].joinpath('test_base_models')
    path_dir.mkdir(parents=True, exist_ok=True)
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

    # %% Simpler models that predict the AHI and classifies it, no stacking
    # collect all the features avoid leakage
    sections_fet = [sect for sect in sections if not sect in 'resp']
    features = [col for col in df_data.columns if col.startswith(tuple(sections_fet))]
    features = [fet for fet in features if not fet in [stratify_col, target_ahi_regressor]]

    # # ======== TRAIN REGRESSOR ========
    regressor_objectives = [
        # GAMMA
        {'objective': 'reg:gamma', 'eval_metric': 'rmse', 'max_bin': 512},
        {'objective': 'reg:gamma', 'eval_metric': 'rmse', 'max_bin': 1024},

        # SQUARED ERROR
        {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_bin': 512},
        {'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'max_bin': 1024},

        # TWEEDIE - 1.5
        {'objective': 'reg:tweedie', 'eval_metric': 'tweedie-nloglik@1.5', 'tweedie_variance_power': 1.5,
         'max_bin': 1024},
        {'objective': 'reg:tweedie', 'eval_metric': 'tweedie-nloglik@1.5', 'tweedie_variance_power': 1.5,
         'max_bin': 1024},

        # TWEEDIE - 1.3
        {'objective': 'reg:tweedie', 'eval_metric': 'tweedie-nloglik@1.3', 'tweedie_variance_power': 1.3,
         'max_bin': 1024},
        {'objective': 'reg:tweedie', 'eval_metric': 'tweedie-nloglik@1.3', 'tweedie_variance_power': 1.3,
         'max_bin': 1024},

        # POISSON
        {'objective': 'count:poisson', 'eval_metric': 'rmse', 'max_bin': 1024},

        # HUBER LOSS for robust regression (used with outlier-heavy data)
        {'objective': 'reg:pseudohubererror', 'eval_metric': 'mae', 'max_bin': 1024}
    ]


    # Train loop over different objectives
    for params in regressor_objectives:
        obj = params['objective'].replace(':', '')
        metric = params['eval_metric'].replace('@', '_')
        bin_size = params.get('max_bin', 1024)
        output_path = path_dir.joinpath(f'regressor_{obj}_{metric}')
        output_path.mkdir(parents=True, exist_ok=True)

        # Build full parameter dictionary
        in_params_regression = {
            **params,
            'max_bin': 256,
            'num_parallel_tree': 10,
            'early_stopping_rounds': 100,
            'num_boost_round': N_BOOSTING_ROUNDS,
        }
        df_model = df_data.copy()
        if 'gamma' in obj:
            epsilon = 0.001
            df_model[target_ahi_regressor] = df_model[target_ahi_regressor] + epsilon

        print(f"\n\n▶ Training with objective = {params['objective']} | eval_metric = {params['eval_metric']}")

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
            model_path=output_path,
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










