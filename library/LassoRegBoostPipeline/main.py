"""
Call the pipeline that does Lasso, linear regression and XGBoost
"""
import pathlib
from typing import Optional
import pandas as pd
from config.config import config, sections, col_remove
from src.ml_models.my_xgboost_optuna import XGBoostModel
from src.ml_models.data_class import Datasets
from ml_tabular_data.dataclass import TargetSelectionSplitting
from config.model_configurations import generate_classification_df, generate_regression_df
from utils import (validate_input, name_model_and_create_directory, slice_by_gender, get_dataset,
                   iterative_feature_selection, reduction_elastic_net_lasso, generate_output_report)
import json
import warnings


def lasso_reg_boost_pipeline(input_dict: dict,
                             n_boosting: int,
                             optuna_n_splits: Optional[int] = 5,
                             optuna_n_trials: Optional[int] = 10,
                             ) -> dict:
    """
    Wrapper of the Lasso - Regression - Xgboost pipeline
    :param input_dict:
    :param n_boosting:
    :param optuna_n_splits:
    :param optuna_n_trials:
    :return:
    """
    # %% validate the input
    if not validate_input(input_dict=input_dict):
        raise ValueError(f'Error in input validation, check: {input_dict}')

    result_path = name_model_and_create_directory(model_path=config.get('models_path'),
                                                  input_dict=input_dict,
                                                  name_model=False)

    # %% 1- load the list of features we will use the in the model
    pp_data, columns_names_initial, columns = get_dataset(
        dataset_path=pathlib.Path(config.get('data_pre_proc_path', None) / 'pp_data_no_nans.csv'),
        columns_path=None,  # input_dict.get('path_features_from_base_register'),
        targets_name=input_dict.get('target'),
        sections=sections,
        manual_col_removal=col_remove,
    )

    if input_dict.get('learning_task') == 'regression':
        target = input_dict.get('regression_target')
    else:
        target = input_dict.get('classification_target_class')

    pp_data = slice_by_gender(df=pp_data,
                              gender_column='DisplayGender',
                              gender_select=input_dict.get('split_gender'))

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
        columns_lasso, lasso_coeff_df = reduction_elastic_net_lasso(
            frame={'train': train_df, 'val': val_df},
            stratify_column=input_dict.get('TargetSelectionSplitting_stratify'),
            target=input_dict.get('LassoReduction_target'),
            model=input_dict.get('LassoReduction_model'),
            sections=None,  # sections,
            feature_names=columns_names_initial,
            output_path=result_path.joinpath('lasso')
        )

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
    importance_types = xgb_model.plot_features_separate_figures(model_xgb,
                                                                plot_path=result_path,
                                                                render_plot=True,
                                                                display=False,
                                                                )
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

    return global_report


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    # %% select the input to use
    df_classification = generate_classification_df()
    df_regression = generate_regression_df()
    # config_model = 'boundaries_4_classes_both_eq_samples'
    for config_model in df_classification.columns:
        # config_model = 'all_4_both'
        input_dict = df_classification.loc[:, config_model].to_dict()
        if len(input_dict.keys()) < 2:
            input_dict = input_dict.get(config_model)
        lasso_reg_boost_pipeline(input_dict=input_dict,
                                 n_boosting=80000,  # 3000
                                 optuna_n_splits=5,  # 5
                                 optuna_n_trials=10  # 10
                                 )

    pass

    # %% predict from a saved model -> this works
    # test_predictions = model_xgb.predict(ds.test_X)
    # if input_dict.get('learning_task') == 'regression':
    #     # evaluate regressor
    #     mse = mean_squared_error(ds.test_y, test_predictions)
    #     rmse = mse ** 0.5
    #     mae = mean_absolute_error(ds.test_y, test_predictions)
    #     r2 = r2_score(ds.test_y, test_predictions)
    #
    #     # Create a dictionary with the metrics
    #     metrics = {
    #         'Mean Squared Error': [mse],
    #         'Root Mean Squared Error': [rmse],
    #         'Mean Absolute Error': [mae],
    #         'R-squared': [r2]
    #     }
    #
    #     # Convert dictionary to DataFrame
    #     metrics_df = pd.DataFrame(metrics)
    #
    #     # Display the DataFrame
    #     print(metrics_df)
    #
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(ds.test_y, test_predictions, alpha=0.3)
    #     plt.plot([ds.test_y.min(), ds.test_y.max()],
    #              [ds.test_y.min(), ds.test_y.max()],
    #              'k--',
    #              lw=4)  # Ideal line
    #     plt.xlabel('True Values')
    #     plt.ylabel('Predictions')
    #     plt.title('True vs. Predicted Values')
    #     plt.show()
    #
    # else:
    #     # evalaute classifier
    #     # Print classification report
    #     print(classification_report(ds.test_y, test_predictions))
    #
    #     # Calculate and display accuracy
    #     accuracy = accuracy_score(ds.test_y, test_predictions)
    #     print("Accuracy:", accuracy)
    #
    #     # Generate confusion matrix
    #     conf_matrix = confusion_matrix(ds.test_y, test_predictions)
    #
    #     # Plotting confusion matrix
    #     plt.figure(figsize=(10, 7))
    #     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues")
    #     plt.title('Confusion Matrix')
    #     plt.xlabel('Predicted Label')
    #     plt.ylabel('True Label')
    #     plt.show()
    #
    # # %% statistical testing of the features
    # # is there a difference in the responses based on the target?
    # features = model_xgb.feature_names
    #
    # pp_data['SA_Sleep_Snore']
