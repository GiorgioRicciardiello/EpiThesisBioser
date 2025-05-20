"""
Evalaute the model obtained from the iterative regression algorithm
"""
import pathlib
from tqdm import tqdm
import pandas as pd
from config.config import config, sections, col_remove
import numpy as np
from src.ml_models.my_xgboost_optuna import XGBoostModel
from typing import Optional, Tuple
from iterative_regression.utils import ahi_class
import warnings
from sklearn.model_selection import train_test_split
from src.ml_models.data_class import Datasets
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if __name__ == '__main__':
    TRAIN = True
    num_class = None
    model_name = 'reg_after_lasso'
    target: str = 'AHI_log1p'  # ['SAO2_Min', 'SAO2_Per'
    str_target_class = f'AHI_multiclass'
    learning_task = 'regression'

    model_name_ml = model_name + f'_xgboost_{num_class}'
    path_base_register = config.get('results_path').joinpath('presentation', model_name, 'base_register.xlsx')


    # %% Loading data, preparing target
    if not path_base_register.exists():
        str_warn = f'Unable to locate file {path_base_register.name}'
        warnings.warn(str_warn)
    # read the feature selected for the best model
    feature_names = pd.read_excel(path_base_register)
    feature_names = feature_names.iloc[:, -1]
    columns = feature_names.to_list().copy()
    columns.append(target)
    # read the dataset and select columns of interest
    pp_data = pd.read_csv(config.get('data_pre_proc_path', None) / 'pp_data_no_nans.csv')
    pp_data = pp_data.loc[:, columns]

    # classification target
    if not num_class is None:
        pp_data[str_target_class] = pp_data[target].apply(lambda x: np.exp(x) - 1)
        pp_data[str_target_class] = ahi_class(x=pp_data[str_target_class], num_class=num_class).values
        pp_data[str_target_class] = pp_data[str_target_class].astype(int)
        pp_data.drop(columns=target, inplace=True)
        target = str_target_class
        pp_data.rename(mapper={str_target_class: target},
                       inplace=True)
        model_name = model_name + f'_{num_class}'
    # %% output of model folder
    path_model = config.get('models_path').joinpath(model_name_ml)
    if not path_model.exists():
        path_model.mkdir(parents=True, exist_ok=True)
    # %% regression model
    # %% Split up our features and labels based on the train, valid, and test indices defined above
    train_val_df, test_df = train_test_split(pp_data,
                                             test_size=0.2,
                                             stratify=pp_data.AHI_multiclass,
                                             random_state=0)
    # Split the temporary data into equal halves for validation and testing, again stratifying by the
    # 'modality_combo' column
    train_df, val_df = train_test_split(train_val_df,
                                        test_size=0.2,
                                        stratify=train_val_df.AHI_multiclass,
                                        random_state=0)
    del train_val_df

    ds = Datasets(
        train_X=train_df[feature_names],
        valid_X=val_df[feature_names],
        test_X=test_df[feature_names],
        train_y=train_df[target],
        valid_y=val_df[target],
        test_y=test_df[target],
    )
    ds.get_shape()
    ds.plot_stratified_distribution(save_plot=True,
                                    show_plot=False,
                                    output_path=None)

    xgb_model = XGBoostModel(
        n_boosting=15000,
        ds=ds,
        learning_task=learning_task,
        path_model_save=path_model.joinpath('xgboost_model.json'),
        path_model_results=path_model,
    )

    def elastic_net():
        ### temporary elastic net
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import classification_report
        from sklearn.metrics import accuracy_score, classification_report, f1_score
        from sklearn.model_selection import GridSearchCV, StratifiedKFold
        GRID_SEARCH = False
        train_val_df, test_df = train_test_split(pp_data,
                                                 test_size=0.2,
                                                 stratify=pp_data.AHI_multiclass,
                                                 random_state=0)
        train_x = train_val_df[feature_names]
        train_y = train_val_df[target]

        test_x = test_df[feature_names]
        test_y = test_df[target]

        elastic_net_model = SGDClassifier(loss='log_loss',
                          penalty='elasticnet',
                          l1_ratio=0.5,
                          alpha=0.001,
                          max_iter=2000,
                          tol=None,
                          random_state=42,
                            learning_rate='adaptive',
                            eta0=0.01,
                          warm_start=True)
        if GRID_SEARCH:
            # GridSearchCV
            param_grid = {
                'alpha': [0.001, 0.01, 0.1, 1, 2],  # Regularization strength
                'l1_ratio': [0.1, 0.5, 0.9],   # Balance between L1 and L2 regularization
            }
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid_search = GridSearchCV(elastic_net_model, param_grid, cv=cv, scoring='accuracy', verbose=1, n_jobs=-1)
            grid_search.fit(train_x, train_y)
            # Predict on test set with the best model
            best_model = grid_search.best_estimator_
            pred_test = best_model.predict(test_x)

            # grid_search.best_params_
            # Out[3]: {'alpha': 0.001, 'l1_ratio': 0.5}
            #
            # Classification report
            print(classification_report(test_y, pred_test))
            report_df = pd.DataFrame(classification_report(test_y, pred_test, output_dict=True)).transpose()
            print(report_df)
            report_df.to_excel(f'SGDClassifier_report.xlsx')

        # Fit the model
        elastic_net_model.fit(ds.train_X, ds.train_y)

        # Predict on training and validation set
        pred_train = elastic_net_model.predict(ds.train_X)
        pred_test = elastic_net_model.predict(ds.test_X)
        print(classification_report(ds.test_y, pred_test))
        report = classification_report(ds.test_y, pred_test, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
    ELASTIC_NET = False
    if ELASTIC_NET:
        elastic_net()

    if TRAIN:
        # optimization
        # best_params, cv_results_df = xgb_model.run_hparam_search_optuna(n_splits=5,
        #                                                                 ds=ds,
        #                                                                 n_trials=10)
        # train best model
        predictions_xgb, metrics_xgb = xgb_model.train_and_eval_model_lr_scheduler(ds=ds)
        # save
        xgb_model.save_my_model()
        # get current model
        model_xgb = xgb_model.get_model()

    else:
        # load pre-existing model
        model_xgb = xgb_model.load_model()
        # TODO: make a method that does the predictions from a loaded model

    # %% evaluate the model