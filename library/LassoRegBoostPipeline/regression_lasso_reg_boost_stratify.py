"""
We will run all the pipeline as a regression objective and then the prediction will be transformed to the classes and
then the model results will be evaluated as a classification task
"""
import pathlib
from typing import Optional, Tuple
import pandas as pd
from pandas import DataFrame

from config.config import config, sections, col_remove
from src.ml_models.my_xgboost_optuna import XGBoostModel
from src.ml_models.data_class import Datasets
from ml_tabular_data.dataclass import TargetSelectionSplitting
from utils import (validate_input, name_model_and_create_directory, slice_by_gender, get_dataset,
                   iterative_feature_selection, reduction_elastic_net_lasso, generate_output_report)
import json

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import warnings
# Suppress warnings
warnings.filterwarnings("ignore")

def generate_single_regression_df(sampler_method: str,
                                  gender: str,
                                  lasso: bool = False) -> dict:
    """
    Generate a single regression model configuration.
    :param sampler_method:
    :param gender:
    :param lasso: if to run the lasso operation
    :return:
    """
    if sampler_method not in ['boundaries', 'interior_domain',
                              'oversample_minority', 'undersample_majority',
                              'smote', 'ADASYN', 'all']:
        raise ValueError(f'Undefined sampler method')

    if gender not in ['male', 'female', 'both']:
        raise ValueError(f'Undefined gender')

    input_regression = {
        ####
        'model_name': 'under_reg_both',  # str
        ####
        'path_features_from_base_register': None,
        # config.get('results_path').joinpath('presentation', 'reg_after_lasso', 'base_register.xlsx'),
        'path_data': config.get('data_pre_proc_path', None) / 'pp_data_no_nans.csv',
        'path_output': config.get('models_path'),  # pathlib.path
        ####
        'regression_target': 'AHI',  # AHI_log1p
        'learning_task': 'regression',
        ####
        'classification_num_class': None,  # integer or none,
        'classification_target_class': None,  # str
        'train': True,  # bool, if to train
        'ml_algorithm': 'xgboost',
        'split_gender': gender,  # ['male', 'female', 'both']
        ####
        'TargetSelectionSplitting_discrete_column': 'AHI_multiclass',
        'TargetSelectionSplitting_cont_column': 'AHI',
        # 'TargetSelectionSplitting_selection': 'boundaries',
        'TargetSelectionSplitting_method': sampler_method,
        'TargetSelectionSplitting_stratify': 'AHI_multiclass',
        'TargetSelectionSplitting_percent': 0.30,
        'TargetSelectionSplitting_equal_samples_class': False,
        ####
        'LassoReduction_target': 'AHI_log1p' if lasso else None,
        'LassoReduction_model': 'regression' if lasso else None,
        'LassoReduction_stratify': 'AHI_multiclass' if lasso else None,
        ####
        'IterativeFeatureReg_output_path': 'OlsFeatSelc',
    }

    return input_regression


def regression_to_classification(
        pred_true_lbls_path: pathlib.Path,
        output_path: pathlib.Path,
) -> pd.DataFrame:
    """
    Since we save the the json with the predictions and the true values, we can use those file and avoid making
    xgboost matrices or predictions.
    Convert the regression to a classification results and evaluate the results
    :param pred_true_lbls_path:json file with predictions and true values dictionaries
            >> pred_true_lbls.keys()
            >> dict_keys(['predictions', 'true_labels'])
            >> pred_true_lbls.get('predictions').keys()
            >> dict_keys(['train', 'valid', 'test'])
    :param output_path: folder directory where to save the csv and images of each report
    :return:
        frame of the classification report and confusion matrix of each split
    """
    def open_json_file(file_path) -> dict:
        with open(file_path, 'r') as file:
            return json.load(file)

    def logp1_to_ahi(ahi_logp1: np.ndarray) -> np.ndarray:
        return np.exp(ahi_logp1) - 1

    def ahi_to_class(ahi: list[float],
                     thresholds: Optional[list[int]] = None
                     ) -> list[int]:
        """
        From ahi in regression to ahi in multi class
        :param ahi: ahi values
        :param thresholds: values to threshold
        :return:
        """
        if thresholds is None:
            thresholds = [5, 15, 30]
        ahi = pd.Series(ahi)
        bins = [-float('inf')] + thresholds + [float('inf')]
        labels = [0, 1, 2, 3]
        return pd.cut(ahi, bins=bins, labels=labels).astype(int).tolist()

    sns.set_context("talk")

    # Create the output directory if it does not exist
    output_path.mkdir(parents=True, exist_ok=True)

    def plot_regression(y_true: pd.Series, y_pred: pd.Series):
        # Scatter plot of true vs predicted values
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=y_true, y=y_pred, label='Predictions', alpha=0.6, s=50)

        # Plot the diagonal line (perfect predictions)
        max_val = max(max(y_true), max(y_pred))
        min_val = min(min(y_true), min(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

        # Calculate residuals
        residuals = y_true - y_pred

        # Plot standard deviation lines
        mean_pred = np.mean(y_pred)
        std_pred = np.std(y_pred)
        plt.axhline(mean_pred + std_pred, color='green', linestyle='--', label='Mean + 1 Std Dev')
        plt.axhline(mean_pred - std_pred, color='green', linestyle='--', label='Mean - 1 Std Dev')

        # Plot mean line
        plt.axhline(mean_pred, color='blue', linestyle='--', label='Mean Prediction')

        # Grid
        plt.grid(True)

        # Labels and title
        plt.xlabel('True Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title('True vs Predicted Values', fontsize=16)
        plt.legend()

        # Show plot
        plt.show()

        # Residuals plot
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, bins=30, color='blue', alpha=0.6)
        plt.axvline(np.mean(residuals), color='red', linestyle='--', label='Mean Residual')
        plt.axvline(np.mean(residuals) + np.std(residuals), color='green', linestyle='--', label='Mean + 1 Std Dev')
        plt.axvline(np.mean(residuals) - np.std(residuals), color='green', linestyle='--', label='Mean - 1 Std Dev')

        # Grid
        plt.grid(True)

        # Labels and title
        plt.xlabel('Residuals', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Residuals Distribution', fontsize=16)
        plt.legend()

        # Show plot
        plt.show()

    col_index_mapper = {'0': 'Normal',
                        '1': 'Mild',
                        '2': 'Moderate',
                        '3': 'Severe'}

    # Define AHI thresholds for 4 classes
    ahi_thresholds = [5, 15, 30]

    # read the true and predictions and separate the dictionaries
    pred_true_lbls = open_json_file(pred_true_lbls_path)
    predictions = pred_true_lbls['predictions']
    true_labels = pred_true_lbls['true_labels']

    # pd.Series(true_labels.get('train')).describe()

    # from log(ahi+1) to ahi
    predictions = {key: logp1_to_ahi(ahi_logp1=value) for key, value in predictions.items()}
    true_labels = {key: logp1_to_ahi(ahi_logp1=value) for key, value in true_labels.items()}

    # from regression to classification, apply the AHI thresholds for a 4 class problem
    predicted_classes = {key: ahi_to_class(ahi=value, thresholds=ahi_thresholds) for key, value in
                         predictions.items()}
    true_classes = {key: ahi_to_class(ahi=value, thresholds=ahi_thresholds) for key, value in true_labels.items()}

    # report the classification
    report_with_conf_matrix_all = pd.DataFrame()
    for split_ in predicted_classes.keys():
        report = classification_report(y_true=true_classes.get(split_),
                                       y_pred=predicted_classes.get(split_),
                                       output_dict=True)

        report_df = pd.DataFrame(report).transpose()
        conf_matrix = confusion_matrix(true_classes.get(split_), predicted_classes.get(split_))
        conf_matrix_df = pd.DataFrame(conf_matrix,
                                      index=report_df.index[:-3],
                                      columns=report_df.index[:-3])
        report_df.rename(mapper=col_index_mapper,
                         axis=1,
                         inplace=True)
        report_df.rename(index=col_index_mapper,
                         inplace=True)
        conf_matrix_df.rename(mapper=col_index_mapper,
                              axis=1,
                              inplace=True)
        conf_matrix_df.rename(index=col_index_mapper,
                              inplace=True)

        # Append the confusion matrix to the report DataFrame
        report_with_conf_matrix = pd.concat([report_df, conf_matrix_df], axis=1)

        # Plotting all in one figure ()
        fig, (ax1, ax2) = plt.subplots(nrows=2,
                                       figsize=(10, 12))
        # Heatmap for confusion matrix
        sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1)
        ax1.set_title(f'Confusion Matrix - {split_.capitalize()}')
        ax1.set_xlabel('Predicted Labels')
        ax1.set_ylabel('True Labels')
        ax1.set_yticklabels(ax1.get_yticklabels(),
                            rotation=0)

        # Table for classification report
        ax2.axis('off')
        table = ax2.table(cellText=report_df.round(4).values,
                          colLabels=report_df.columns,
                          rowLabels=report_df.index,
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        ax2.set_title(f'Classification Report - {split_.capitalize()}')
        plt.tight_layout()
        plt.savefig(output_path.joinpath(f'ConfMxReport{split_.capitalize()}.png'), dpi=300)
        plt.show()

        report_with_conf_matrix['Split'] = split_
        report_with_conf_matrix_all = pd.concat([report_with_conf_matrix_all,
                                                 report_with_conf_matrix])

    report_with_conf_matrix_all.to_csv(output_path.joinpath(f'ReportSplits.csv'),
                                       index=True)
    return report_with_conf_matrix_all


def lasso_reg_boost_pipeline(input_dict: dict,
                             n_boosting: int,
                             optuna_n_splits: Optional[int] = 5,
                             optuna_n_trials: Optional[int] = 10,
                             ) -> tuple[dict, DataFrame]:
    # %% validate the input
    if not validate_input(input_dict=input_dict):
        raise ValueError(f'Error in input validation, check: {input_dict}')

    result_path = name_model_and_create_directory(model_path=config.get('models_path'),
                                                  input_dict=input_dict,
                                                  name_model=False)

    # %% 1- load the list of features we will use the in the model
    pp_data, columns_names_initial, columns = get_dataset(
        dataset_path=pathlib.Path(config.get('data_pre_proc_path', None) / 'pp_data_no_nans.csv'),
        columns_path=None,  # input_dict.get('path_features_from_base_register') ,
        targets_name=input_dict.get('target'),
        sections=sections,
        manual_col_removal=col_remove,
    )

    if input_dict.get('learning_task') == 'regression':
        target = input_dict.get('regression_target')
    else:
        target = input_dict.get('classification_target_class')

    if input_dict.get('split_gender') == 'both':
        pp_data = slice_by_gender(df=pp_data,
                                  gender_column='DisplayGender',
                                  gender_select=input_dict.get('split_gender'))
    if input_dict.get('split_gender') in ['male', 'female']:
        if input_dict.get('split_gender') == 'male':
            columns_to_drop = ['menopausal_stage_code', 'DisplayGender']
        else:
            columns_to_drop = ['DisplayGender']
        pp_data.drop(columns=columns_to_drop, inplace=True)
        columns_names_initial = [col for col in columns_names_initial if col not in columns_to_drop]
        columns = [col for col in columns if col not in columns_to_drop]
    # %% 2- select the observations based on the target and split
    target_split = TargetSelectionSplitting(
        dataset=pp_data,
        # selection=input_dict.get('TargetSelectionSplitting_selection'),
        method=input_dict.get('TargetSelectionSplitting_method'),
        discrete_column=input_dict.get('TargetSelectionSplitting_discrete_column'),
        cont_column=input_dict.get('TargetSelectionSplitting_cont_column'),
        num_classes=input_dict.get('classification_num_class'),
        stratify=input_dict.get('TargetSelectionSplitting_stratify'),
        equal_samples_class=input_dict.get('TargetSelectionSplitting_equal_samples_class'),
        percent=input_dict.get('TargetSelectionSplitting_percent'),
        output_path=result_path
    )

    train_df, val_df, test_df = target_split.split_train_val_test(plot=False)

    # %% 3 - feature selection unless pre-specified
    columns_lasso = None
    lasso_coeff_df = None
    if input_dict.get('path_features_from_base_register') is None:
        # the splits sliced based on the target criteria, we want to use these ones
        frame_combined_splits = pd.concat([train_df, val_df],
                                          axis=0)

        #  3.2 - run Lasso
        if input_dict.get('LassoReduction_model') is not None:
            columns_lasso, lasso_coeff_df = reduction_elastic_net_lasso(
                frame={'train': train_df, 'val': val_df},
                stratify_column=input_dict.get('TargetSelectionSplitting_stratify'),
                target=input_dict.get('LassoReduction_target'),
                model=input_dict.get('LassoReduction_model'),
                sections=None,  # sections,
                feature_names=columns_names_initial,
                output_path=result_path.joinpath('lasso')
            )
        else:
            columns_lasso = columns_names_initial

        #  3.2 - run linear regression to reduce the features
        # TODO: OR plots are not being save, we need to fix this
        feature_names = iterative_feature_selection(frame=frame_combined_splits,
                                                    target_continuous='AHI_log1p',
                                                    model_name=result_path.name,
                                                    features_from_lasso=columns_lasso,
                                                    col_manual_removal=col_remove,
                                                    max_iterations=100,
                                                    output_path=result_path.joinpath(
                                                        input_dict.get('IterativeFeatureReg_output_path')),
                                                    )

        input_dict['path_features_from_base_register'] = feature_names

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
                                        output_path=result_path)

    else:
        feature_names = input_dict['path_features_from_base_register']
        ds = Datasets(
            train_X=train_df[feature_names],
            valid_X=val_df[feature_names],
            test_X=test_df[feature_names],
            train_y=train_df[target],
            valid_y=val_df[target],
            test_y=test_df[target],
        )
        ds.get_shape()
        ds.plot_stratified_distribution()

    # %% 3- train and optimize the model
    if input_dict.get('learning_task') == 'regression':
        xgb_model = XGBoostModel(
            n_boosting=n_boosting,
            ds=ds,
            learning_task=input_dict.get('learning_task'),
            path_model_save=result_path.joinpath('xgboost_model.json'),
            path_model_results=result_path,
        )
    else:
        xgb_model = XGBoostModel(
            n_boosting=n_boosting,
            ds=ds,
            learning_task=input_dict.get('learning_task'),
            num_classes=input_dict.get('classification_num_class'),
            path_model_save=result_path.joinpath('xgboost_model.json'),
            path_model_results=result_path,
        )

    if input_dict.get('train'):
        # optimization
        lr_scheduler = False
        # train best model
        if lr_scheduler:
            predictions_xgb, metrics_df = xgb_model.train_and_eval_model_lr_scheduler(ds=ds)
        else:
            best_params, cv_results_df, predictions_xgb, metrics_df = (
                xgb_model.run_hparam_search_optuna(n_splits=optuna_n_splits,
                                                   ds=ds,
                                                   n_trials=optuna_n_trials,
                                                   early_stopping_rounds=200,
                                                   lr_decay=None,
                                                   plt_show=False))
        # save
        xgb_model.save_my_model()
        # get current model
        model_xgb = xgb_model.get_model()

    else:
        # load pre-existing model
        model_xgb = xgb_model.load_model()
        predictions, metrics_df = xgb_model.predict_on_loaded_model(xgb_model=model_xgb,
                                                                    ds=ds)
    # %% evaluate the model
    # importance_types = xgb_model.plot_features_separate_figures(model_xgb,
    #                                                             plot_path=result_path,
    #                                                             render_plot=True,
    #                                                             display=False,
    #                                                             )
    # %% save the dict
    output_path_str = str(result_path.joinpath('input_dict.json'))
    # Iterate through the dictionary and convert WindowsPath objects to strings
    for key, value in input_dict.items():
        if isinstance(value, pathlib.Path):
            input_dict[key] = str(value)
    with open(output_path_str, 'w') as f:
        json.dump(input_dict, f)

    # %% generate output report folder
    global_report = generate_output_report(
        features_initial=columns_names_initial,
        features_after_lasso=columns_lasso,
        lasso_coeff_df=lasso_coeff_df,
        features_after_reg=feature_names,
        metrics_ml=metrics_df,
        output_path=result_path,
        input_config=input_dict,
        splits_target={'train': ds.train_y, 'val': ds.valid_y, 'test': ds.test_y}
    )

    # %% classify the regression
    report_with_conf_matrix = regression_to_classification(
        pred_true_lbls_path=result_path.joinpath('SplitPredTrueLbls.json'),
        output_path=result_path.joinpath('RegToClass')
    )

    return global_report, report_with_conf_matrix


if __name__ == '__main__':
    # TODO: 1 age & gender slicing
    # TODO: 2 make a dummy dataset, arausal_plmi to predict the plmi, or ESS questions and scores
    # TODO: 3 remove the logp1 in the target
    # TODO: 4 "DI" as the target, we need to avoid removing it from the create_dataset()
    # TODO: 5 plot by age the AHI and the ODI and separate by gender as well
    # TODO: bootstraping - Sherlock
    # TODO: backward and forward propagation
    # TODO: cluster analysis group by ahi
    config_model = generate_single_regression_df(sampler_method='undersample_majority',
                                                 gender='male',
                                                 lasso=False)

    report, report_classifier = lasso_reg_boost_pipeline(input_dict=config_model,
                                      n_boosting=40000,  # 3000
                                      optuna_n_splits=5,  # 5
                                      optuna_n_trials=10  # 10
                                      )

    print(report)
    print(report_classifier)

    #