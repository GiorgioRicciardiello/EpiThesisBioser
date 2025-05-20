"""
Test the xgboost model with a synthetic dataset.

Results: The pipeline works wells and model learns from the data in all data splits.

  Splits       RMSE        MAE        R2
0  train  54.894772  43.365742  0.862093
1  valid  62.865651  49.732956  0.819595
2   test  62.428727  49.231129  0.819719


"""
import pathlib
from typing import Optional, Tuple, Union, Any
import pandas as pd
from pandas import DataFrame
from config.config import config, sections, col_remove
from src.ml_models.my_xgboost_optuna import XGBoostModel
from src.ml_models.data_class import Datasets
from sklearn.datasets import make_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.datasets import load_diabetes, fetch_california_housing
from typing import Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import json

def synthetic_regressor(dataset_option: str = 'synthetic') -> Tuple[Union[DataFrame, Any], Any]:
    """
    Create synthetic dataset or load a pre-existing sklearn dataset to test the XGBoost model
    :param dataset_option:
    :return:
    """
    if dataset_option not in ['synthetic', 'housing', 'diabetes']:
        raise ValueError(f'Option {dataset_option} not available for dataset creation')

    if dataset_option == 'synthetic':
        # Setting random seed for reproducibility
        np.random.seed(42)

        # Parameters
        n_samples = 5000
        n_features = 20  # 50 ordinal, 10 binary, 4 continuous
        n_informative = 8
        # 5 ordinal, 2 binary, 1 continuous
        n_ordinal = 5
        n_binary = 2
        n_continuous = n_features - n_ordinal - n_binary
        # Generate synthetic features
        X, y = make_regression(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               effective_rank=None,  # Make full rank for simplicity
                               n_targets=1,
                               bias=10,  # adds a constant value to the target variable
                               tail_strength=0.8,  # higher means more of the singular values are significant
                               noise=0,
                               shuffle=True,
                               random_state=42)

        # Separate features into ordinal, binary, and continuous
        column_ordinal = np.random.choice(X.shape[1],
                                          n_ordinal,
                                          replace=False)

        # Exclude ordinal columns when selecting binary columns
        available_columns_for_binary = np.setdiff1d(np.arange(X.shape[1]),
                                                    column_ordinal)
        columns_binary = np.random.choice(available_columns_for_binary,
                                          n_binary,
                                          replace=False)
        # Continuous features are the remaining columns
        remaining_columns = np.setdiff1d(np.arange(X.shape[1]),
                                         np.concatenate((column_ordinal, columns_binary)))

        ordinal_features = X[:, column_ordinal]
        binary_features = X[:, columns_binary]
        continuous_features = X[:, remaining_columns]

        print("Original Matrix shape:", X.shape)
        print("Ordinal Features shape:", ordinal_features.shape)
        print("Binary Features shape:", binary_features.shape)
        print("Continuous Features shape:", continuous_features.shape)

        # Transform ordinal features to discrete values between 0 and 5
        kbin = KBinsDiscretizer(n_bins=6,
                                encode='ordinal',
                                strategy='uniform')
        ordinal_features = kbin.fit_transform(ordinal_features)

        # Transform binary features to 0 and 1
        binary_features = np.where(binary_features > np.median(binary_features), 1, 0)

        # Combine all features
        X = np.hstack((ordinal_features, binary_features, continuous_features))

        # Generate chi-square distributed target
        # target = np.random.chisquare(df=2, size=n_samples)

        # Create DataFrame for better readability
        # col_ord = {col: f'ord_{col}' for col in column_ordinal}
        # col_bin = {col: f'bin_{col}' for col in columns_binary}
        # col_cont = {col: f'cont_{col}' for col in remaining_columns}
        # col_ord.update(col_bin)
        # col_ord.update(col_cont)
        #
        # X = pd.DataFrame(X, columns=[col for col in range(X.shape[1])])
        # X.rename(columns=col_ord, inplace=True)
        columns = ([f"ord_{i}" for i in range(len(column_ordinal))] +
                   [f"bin_{i}" for i in range(len(columns_binary))] +
                   [f"cont_{i}" for i in range(len(remaining_columns))])
        X = pd.DataFrame(X, columns=columns)

        y = pd.Series(y)
        y.name = 'target'

    elif dataset_option == 'housing':
        X, y = fetch_california_housing(return_X_y=True, as_frame=False)

    elif dataset_option == 'diabetes':
        X, y = load_diabetes(return_X_y=True, as_frame=False)

    return X, y


def plot_feature_vs_target(x: pd.Series, y: pd.Series):
    if 'continuous' in x.name:  # continuous features
        corr = np.corrcoef(m=x, y=y)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=x, y=y)
        plt.title(f'Relationship between {x.name} and Target'
                  f'\nCorrelation {np.round(corr, 3)}')
        plt.xlabel(x.name)
        plt.ylabel('Target')
        plt.show()
    else:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=x, y=y)
        plt.title(f'Relationship between {x.name} and Target')
        plt.xlabel(x.name)
        plt.ylabel('Target')
        plt.show()


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

    plot_regression(y_true=true_labels, y_pred=predictions)

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


if __name__ == '__main__':
    # %% INPUT
    input_dict = {
        ####
        'model_name': 'test_boost',  # str
        ####
        'path_features_from_base_register': None,
        # config.get('results_path').joinpath('presentation', 'reg_after_lasso', 'base_register.xlsx'),
        'path_output': config.get('models_path'),  # pathlib.path
        ####
        'regression_target': 'AHI',  # AHI_log1p
        'learning_task': 'regression',
        ####
        'classification_num_class': None,  # integer or none,
        'classification_target_class': None,  # str
        'train': True,  # bool, if to train
        'ml_algorithm': 'xgboost',
        ####
    }

    n_boosting_rounds = 1000  # 3000
    optuna_n_splits = 5  # 5
    optuna_n_trials = 10

    result_path = config.get('models_path').joinpath(input_dict.get('model_name'))
    # Create the directory if it does not exist
    if not result_path.exists():
        result_path.mkdir(parents=True, exist_ok=True)
        print(f"Model path has been created with name: {result_path}")

    # %% ###########################################################################
    # %% Synthetic dataset
    X, y = synthetic_regressor(dataset_option='synthetic')
    # # evaluate the synthetic dataset
    # rank = np.linalg.matrix_rank(X)
    #
    # for col_idx in range(0, X.shape[1] - 55):
    #     plot_feature_vs_target(x=X.iloc[:, col_idx],
    #                            y=y)
    # split and set as dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X,
                                                        y,
                                                        test_size=0.3,
                                                        random_state=42)

    # Now split the temporary set into validation and test set
    X_val, X_test, y_val, y_test = train_test_split(X_temp,
                                                    y_temp,
                                                    test_size=0.5,
                                                    random_state=42)
    ds = Datasets(
        train_X=X_train,
        valid_X=X_val,
        test_X=X_test,
        train_y=y_train,
        valid_y=y_val,
        test_y=y_test,
    )
    ds.get_shape()
    # visualize the target
    plt.figure(figsize=(14, 7))
    sns.kdeplot(ds.train_y, label='Train Set', color='blue', fill=True, alpha=0.3)
    sns.kdeplot(ds.valid_y, label='Validation Set', color='green', fill=True, alpha=0.3)
    sns.kdeplot(ds.test_y, label='Test Set', color='red', fill=True, alpha=0.3)
    plt.title('KDE Plot of Train, Validation, and Test Sets', fontsize=16)
    plt.xlabel('Target Value', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.legend(title='Datasets', title_fontsize='13', fontsize='12')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # %% ###########################################################################
    # Train the model

    if input_dict.get('learning_task') == 'regression':
        xgb_model = XGBoostModel(
            n_boosting=n_boosting_rounds,
            ds=ds,
            learning_task=input_dict.get('learning_task'),
            path_model_save=result_path.joinpath('xgboost_model.json'),
            path_model_results=result_path,
        )
    else:
        xgb_model = XGBoostModel(
            n_boosting=n_boosting_rounds,
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
                                                                display=True,
                                                                figsize=(6, 8)
                                                                )

    report_with_conf_matrix = regression_to_classification(
        pred_true_lbls_path=result_path.joinpath('SplitPredTrueLbls.json'),
        output_path=result_path.joinpath('RegToClass')
    )