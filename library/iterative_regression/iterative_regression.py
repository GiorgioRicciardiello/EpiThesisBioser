import pathlib
import pandas as pd
import statsmodels.api as sm
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from io import StringIO
import re
from typing import Optional, Union
import scipy.stats as stats
import shutil

def get_base_vars(base_var_iteration:dict,
                  selected_key:str) -> list:
    """
    Get the base variables based on the selected key. If the selected_key is not the first one, then it will return
    the .extend() from the first to the selected one

    Args:
        base_var_iteration (dict): Dictionary containing base variable lists. Values must be lists
        selected_key (str): Key of the dictionary to select the base variables.

    Returns:
        list: List of base variables based on the selected key.
    """
    base_vars = []
    keys = list(base_var_iteration.keys())

    if selected_key not in keys:
        raise ValueError(f'please define the appropriate base among the options')
    if selected_key in keys:
        index = keys.index(selected_key) + 1
        for key in keys[:index]:
            if isinstance(base_var_iteration[key], list):
                base_vars.extend(base_var_iteration[key])
            else:
                raise ValueError(f'The values in {key} must be stored as list')
    return base_vars


def remove_base_from_iteratives(base_features: list,
                                iterative_features: list) -> list:
    """
    Remove base features from the iterative features list.

    Args:
        base_features (list): List of base features.
        iterative_features (list): List of iterative features.

    Returns:
        list: List of iterative features with base features removed.
    """
    return [iter_feat for iter_feat in iterative_features if not iter_feat in base_features]



class IterativeRegression:
    def __init__(self,
                 data: pd.DataFrame,
                 target: str,
                 base_features: list[str],
                 iterative_features: list[str],
                 out_path: pathlib.Path,
                 trial_name: str,
                 re_write_dir:bool=True,
                 evaluate_single_coefficient:bool=True):
        """
        RegressionModel class that iterates over iterative features, runs linear regression models with sm.OLS,
        collects metrics, and builds a DataFrame with the model results.

        On each iteration the summary report for ms.OLS is read as a string and the parameters/results of each variable
        are extracted. The class keeps tracks of the current iterative feature, so it can extract it from the summary
        report of the current model and save in an output_df where it will agglomerate the results of each iterative
        features.

        We can opt to create a folder report for each iterative model (evaluate_single_coefficient = False) or
        just preserve one .csv table where we get the parameters of each iterative feature obtained from each model
        (evaluate_single_coefficient = True).

        :param target: pd.Series, dependent variable
        :param base_features: pd.DataFrame, independent variables we that will always be present in the model
        :param iterative_features: pd.DataFrame, independent variables that will be iterated one by one
        :param out_path: pathlib.Path, folder directory to save the model results
        :param trial_name: str, name of the model trial
        :param re_write_dir: bool, if to re-write an existing directory (True) or not (False)
        :param evaluate_single_coefficient: True if create sub folders to save each model results separately, else False
        """
        # self._columns = ['variable', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]',
        #            'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
        #            'No. Observations', 'Dep. Variable', 'odds', 'odds_ci_low', 'odds_ci_high', 'mse', 'responses_count']

        self._columns = {'variable': 'object',
                         'coef': 'object',
                         'std err': 'object',
                         't': 'object',
                         'P>|t|': 'object',
                         '[0.025': 'object',
                         '0.975]': 'object',
                   'Adj. R-squared': 'object',
                         'F-statistic': 'object',
                         'Prob (F-statistic)': 'object',
                   'No. Observations': 'object',
                         'Dep. Variable': 'object',
                         'odds': 'object',
                         'odds_ci_low': 'object',
                         'odds_ci_high': 'object',
                         'mse': 'object',
                         'responses_count': 'object'}

        self.data = data
        # self.data.column = [col.strip() for col in data.columns]
        self.base_features = base_features  # [col.strip() for col in base_features]
        self.iterative_features = iterative_features  # [col.strip() for col in iterative_features]
        self.target = target
        self.out_path = out_path
        self.trial_name = trial_name
        self.trial_folder = None
        self.models_folders = None
        self.output_df = None
        self.mse = None
        self.__summary_ols_results_dict = None
        self.summary_parameters_df = None
        self.summary_distribution_tests_df = None
        self._iter_feature = None
        self.odds = None
        self.responses_count_df = None
        self.__base_model_name = 'base_model'
        self._remove_rows()
        self.evaluate_single_coefficient = evaluate_single_coefficient
        if re_write_dir:
            self._mkdir()
        self._create_output_table()
        self._create_all_models_parameters_output()
        self._report()

    def _report(self):
        print(f'\nBase features {len(self.base_features)}')
        for i, base in enumerate(self.base_features):
            if i == len(self.base_features) - 1:  # If it's the last element
                print(f'\t{base} ** ')
            else:
                print(f'\t{base}')

    def fit_iterative_models(self) -> pd.DataFrame:
        """
        Fit the base model and iterate over the iterative features and evaluate each model models
        :return:
            pd.Dataframe, frame of the coefficient, p value, odds, etc. of each iterated feature in the model
        """
        # base model is always evaluated
        self._fit_base_model()
        self._populate_all_models_params_output()
        for self._iter_feature in tqdm(self.iterative_features,
                                        desc="Fitting Models"):
            # observations_idx = self.data[~self.data[self._iter_feature].isna()].index
            features = self.base_features + [self._iter_feature]
            current_data_df = self.data.dropna(subset=features).copy()

            self._fit_and_evaluate_model(features=current_data_df[features],
                                         target=current_data_df[self.target])

            # self._populate_output_df()
            self._populate_all_models_params_output()
            self._populate_output_df()
            self._save_current_model_results()
        # remove nan row, could be done better
        self.output_df = self.output_df.loc[self.output_df['model'] != 'base_model']
        self._save_global_output()
        return self.output_df

    def _fit_base_model(self):
        """Fit and evaluate the base model"""
        self._iter_feature = self.__base_model_name
        # perform linear regression
        self._fit_and_evaluate_model(features=self.data[self.base_features],
                                     target=self.data[self.target])

    def fit_final_model(self, columns:list[str], target:str) -> tuple[DataFrame, DataFrame, DataFrame]:
        """
        Compute the final OLS for the model using the selected columns.

        The function parses the OLS results into dataframes

        :param columns: list[str], columns to use as dependent variables
        :param target: str, column to use as independent variable
        :return:
            frames of the stats.ols results
            df_model_metrics: metrics as AUROC, R**2
            df_model_params: parameters coeff, p values and ci
            df_stats_tests: statistical test of regression model
        """
        model = sm.OLS(endog=self.data[target],
                       exog=sm.add_constant(self.data[columns]),
                       missing='drop').fit()
        # predictions = model.predict(sm.add_constant(self.data[columns]))
        summary_ols_results = model.summary().tables[0].as_csv()

        summary_parameters = model.summary().tables[1].as_csv()
        summary_distribution_tests = model.summary().tables[2].as_csv()

        # Model metrics
        summary_ols_results_dict = {}
        summary_ols_results_cl = summary_ols_results.split('\n')[1::]
        summary_ols_results_cl = [re.sub(r'(?<!\n)\s+', ' ', line) for line in summary_ols_results_cl]
        for summary_lst in summary_ols_results_cl:
            # Define a pattern to extract key-value pairs
            pattern = re.compile(r'([^,:]+):\s*,\s*([^,]+)')
            matches = re.findall(pattern, summary_lst)
            # Update the result dictionary
            for key, value in matches:
                summary_ols_results_dict[key.strip()] = value.strip()

        df_model_metrics = pd.DataFrame(summary_ols_results_dict, index=[0])
        df_model_metrics.drop(columns=['Method', 'Date', 'Time', 'Covariance Type'],
                                         inplace=True)

        # Model parameters
        # Use StringIO to simulate a file-like object
        data_io = StringIO(summary_parameters)
        df_model_params = pd.read_csv(data_io)
        df_model_params.columns = [col.strip() for col in df_model_params.columns]
        # Rename the columns
        df_model_params.rename(columns={'': 'variable'}, inplace=True)
        df_model_params['variable'] = df_model_params['variable'].apply(lambda x: x.strip())
        # we need the exact p-values
        df_model_params.loc[:, 'P>|t|'] = model.pvalues.values

        # Model Distribution Test
        # Add column names to the string
        summary_distribution_tests_cl = 'Test,Value,Test,Value\n' + summary_distribution_tests
        data_io = StringIO(summary_distribution_tests_cl)
        df_stats_tests = pd.read_csv(data_io)
        df_stats_tests.columns = [col.strip() for col in df_stats_tests.columns]

        return df_model_metrics, df_model_params, df_stats_tests

    def _fit_and_evaluate_model(self,
                                features: pd.DataFrame,
                                target: pd.Series):
        """
        Method dedicate to fit the logistic regression model with the given features.
        :param features: pd.Dataframe, dependent variables of the model
        :param target: pd.Series, independent variable of the model
        :return: None
        """
        self.responses_count_df = self._generate_responses_count_df(features=features)

        model = sm.OLS(endog=target,
                       exog=sm.add_constant(features),
                       missing='drop').fit()
        predictions = model.predict(sm.add_constant(features))

        self._summary_to_structured_object(model=model,
                                            summary_ols_results=model.summary().tables[0].as_csv(),
                                           summary_parameters=model.summary().tables[1].as_csv(),
                                           summary_distribution_tests=model.summary().tables[2].as_csv())

        self.mse = mean_squared_error(y_true=target,
                                      y_pred=predictions)


    def _generate_responses_count_df(self, 
                                     features:pd.DataFrame) -> pd.DataFrame:
        """
        From the dataframe of only features, generate a frame that contains for continuous columns the mean and std,
        and for multi_class columns the count and percentage of each discrete number in the column.

        It saves in a str() structure, so it can later be added to the output_df in a single column
        :param features: pd.DataFrame, frame of the feature we want to describe
        :return:
        """
        def _is_what_type(feature: pd.Series) -> str:
            if 0 < len(feature.unique()) < 9:
                return 'multi_class'
            else:
                return 'continuous'
        response_count = {}
        for col_ in features.columns:
            # col_ = 'DisplayGender'
            if isinstance(features[col_], pd.DataFrame):
                raise ValueError(f'Dataframe and not series, columns: {col_}')
            dtype_ = _is_what_type(feature=features[col_])
            if dtype_ == 'multi_class':
                count = features[col_].value_counts()
                percent = features[col_].value_counts(normalize=True) * 100
                # Create a dictionary with unique elements as keys and tuples of count and percentage as values.
                element_counts_pct = {k: (v, np.round(percent[k], 1)) for k, v in count.items()}
                response_count[col_] = str(element_counts_pct)
            else:
                response_count[col_] = str({'mean': np.round(features[col_].describe()['mean'], 1),
                                            'std': np.round(features[col_].describe()['mean'], 3),
                                            'min':np.round(features[col_].describe()['min'], 2),
                                            'max':np.round(features[col_].describe()['min'], 2)})

        self.responses_count_df = pd.DataFrame(response_count, index=['responses_count']).T
        self.responses_count_df.reset_index(inplace=True)
        self.responses_count_df.rename(columns={'index': 'variable'}, inplace=True)
        self.responses_count_df['model'] = self._get_iter_feature() # to marge only in this model rows
        return self.responses_count_df

    def _get_iter_feature(self) -> str:
        """returns iterator which is a private variable of the class"""
        return self._iter_feature

    @staticmethod
    def _compute_odds(coeff:Union[float, pd.Series],
                      ci_low:Union[float, pd.Series],
                      ci_high:Union[float, pd.Series],
                      return_type:Optional[str]='dict'
                      )-> Union[dict, pd.DataFrame]:
        """
        Compute the odds form the coefficients of a logistic regression model
        :param coeff: float, beta coefficient from the model
        :param return_type: str, return type, frame or dict
        :param ci_low: float, lower bound of the confidence interval of the coefficient
        :param ci_high: float, higher bound of the confidence interval of the coefficient
        :return:
            dict, odds, odds  confidence intervals

        Usage:
            with implementation = 'apply'

            result_df = self.global_params_output_df['coef'].apply(calculate_odds_and_ci)

        """
        # For when we want to define the alpha: we must use the standard error
        # odds_df['odds'] = coef_df['coef'].apply(lambda x: np.round(np.exp(x), 2))
        #
        # # Assuming coef_df is a DataFrame with a column 'std_err' representing the standard error
        # odds_df['odds_ci_low'] = coef_df.apply(lambda row: np.round(np.exp(row['coef'] - 1.96 * row['std_err']), 2),
        #                                        axis=1)
        # odds_df['odds_ci_high'] = coef_df.apply(lambda row: np.round(np.exp(row['coef'] + 1.96 * row['std_err']), 2),
        #                                         axis=1)


        if isinstance(coeff, pd.Series) and isinstance(ci_low, pd.Series) and isinstance(ci_high, pd.Series):
            odds = coeff.apply(lambda x: np.round(np.exp(x), 2))
            odds_ci_low = ci_low.apply(lambda x: np.round(np.exp(x), 2))
            odds_ci_high = ci_high.apply(lambda x: np.round(np.exp(x), 2))

            return pd.DataFrame({'odds': odds, 'odds_ci_low': odds_ci_low, 'odds_ci_high': odds_ci_high})

        else:
            results = {
                'odds': np.round(np.exp(coeff), 2),
                'odds_ci_low': np.round(np.exp(ci_low), 3),
                'odds_ci_high': np.round(np.exp(ci_high), 3),

            }
            if return_type == 'frame':
                return pd.DataFrame(results, index=[0])
            else:
                return results

    @staticmethod
    def _calculate_odds_ratio_ci(coeff: Union[float, pd.Series],
                                str_err: Union[float, pd.Series],
                                alpha: float = 0.05) -> Union[dict, pd.DataFrame]:
        """
        Compute the odds ratio and confidence intervals given the beta coefficient, standard deviation, and alpha
        :param coeff: float or pd.Series, beta coefficient
        :param str_err: float or pd.Series, standard error
        :param alpha: float, significance level
        :return: dict or pd.DataFrame, odds ratio and confidence intervals
        """
        def calculate_single_odds_ratio_ci(single_coeff, single_str_err):
            # Calculate the odds from the coefficient
            odds = np.round(np.exp(single_coeff), 2)

            # Calculate the z-score based on alpha (two-tailed test)
            z_score = stats.norm.ppf(1 - alpha / 2)

            # Calculate the margin of error
            margin_of_error = z_score * single_str_err

            # Calculate confidence interval
            lower_ci = odds - margin_of_error
            upper_ci = odds + margin_of_error

            return {
                'odds': odds,
                'odds_ci_low': np.round(lower_ci, 3),
                'odds_ci_high': np.round(upper_ci, 3),
            }

        if isinstance(coeff, pd.Series) and isinstance(str_err, pd.Series):
            # If input is Series, apply the function to each element
            return pd.DataFrame([calculate_single_odds_ratio_ci(c, se) for c, se in zip(coeff, str_err)],
                                     columns=['odds', 'odds_ci_low', 'odds_ci_high'])

        else:
            # If input is single values, apply the function to the single values
            return calculate_single_odds_ratio_ci(coeff, str_err)

    def _remove_rows(self):
        """
        Base features must not contain nans, re move the rows with nans from the base and the features we will
        iterate
        :return:
        """
        # observation where we do not have a base variable, or we do not have a target must be removed
        self.data = self.data.dropna(subset=self.base_features + [self.target])
        # print(f'Dimension after removing nans in base or target {self.data.shape}')
        self.data.reset_index(inplace=True,
                              drop=True)

    def _summary_to_structured_object(self,
                                      model,
                                      summary_ols_results: str,
                                      summary_parameters: str,
                                      summary_distribution_tests: str):
        """
        Parse the results from statsmodels.iolib.summary.Summary to a DataFrame. This is specific to the format
        the statsmodels api returns the summaries
        :param model: sm.OLS model
        :param summary_ols_results: str,
        :param summary_parameters:  str,
        :param summary_distribution_tests: str,
        """
        # Summary OLS results section
        self.__summary_ols_results_dict = {}
        summary_ols_results_cl = summary_ols_results.split('\n')[1::]
        summary_ols_results_cl = [re.sub(r'(?<!\n)\s+', ' ', line) for line in summary_ols_results_cl]
        for summary_lst in summary_ols_results_cl:
            # Define a pattern to extract key-value pairs
            pattern = re.compile(r'([^,:]+):\s*,\s*([^,]+)')
            matches = re.findall(pattern, summary_lst)
            # Update the result dictionary
            for key, value in matches:
                self.__summary_ols_results_dict[key.strip()] = value.strip()

        self.summary_ols_results_df = pd.DataFrame(self.__summary_ols_results_dict, index=[0])
        self.summary_ols_results_df.drop(columns=['Method', 'Date', 'Time', 'Covariance Type'],
                                         inplace=True)
        # Summary Parameters section
        # Use StringIO to simulate a file-like object
        data_io = StringIO(summary_parameters)
        self.summary_parameters_df = pd.read_csv(data_io)
        self.summary_parameters_df.columns = [col.strip() for col in self.summary_parameters_df.columns]
        # Rename the columns
        self.summary_parameters_df.rename(columns={'': 'variable'}, inplace=True)
        self.summary_parameters_df['variable'] = self.summary_parameters_df['variable'].apply(lambda x: x.strip())
        # we need the exact p-values
        self.summary_parameters_df.loc[:, 'P>|t|'] = model.pvalues.values

        # Summary Distribution Test
        # Add column names to the string
        summary_distribution_tests_cl = 'Test,Value,Test,Value\n' + summary_distribution_tests
        data_io = StringIO(summary_distribution_tests_cl)
        self.summary_distribution_tests_df = pd.read_csv(data_io)
        self.summary_distribution_tests_df.columns = [col.strip() for col in self.summary_distribution_tests_df.columns]

    def _create_output_table(self):
        """
        Create the dataframe we will populate after each OLS regression result.
        Each row of the dataframe is the result of a regression model, and the rows are marked by the current iterative
        variable we are testing.
        :return:
        """
        # columns = ['variable', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]',
        #            'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
        #            'No. Observations', 'Dep. Variable']
        # extra_metrics = ['odds', 'odds_ci_low', 'odds_ci_high', 'mse']  # dependency with the odds ratio function
        # columns.extend(extra_metrics)
        index_lbls = [self.__base_model_name]
        index_lbls.extend(self.iterative_features)
        # self.output_df = pd.DataFrame(np.nan, columns=columns, index=index_lbls)
        self.output_df = pd.DataFrame(np.nan, columns=['model'] + [*self._columns.keys()], index=range(0, len(index_lbls)))
        self.output_df['model'] = index_lbls

        self.output_df = self.output_df.astype(self._columns)


    def _create_all_models_parameters_output(self):
        """
        Pre-allocate the frame that will contain the model parameters out put global_params_output_df
        :return:
        """
        # columns = ['variable', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]',
        #            'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
        #            'No. Observations', 'Dep. Variable', 'odds', 'odds_ci_low', 'odds_ci_high', 'mse', 'responses_count']
        # extra_metrics = ['odds', 'odds_ci_low', 'odds_ci_high', 'mse']  # dependency with the odds ratio function
        # columns.extend(extra_metrics)
        model_names = [self.__base_model_name]
        model_names.extend(self.iterative_features)
        rows_per_model = len(self.base_features) + 1  # 2  # constant and the new iterative feature
        total_rows = rows_per_model * len(model_names)
        self.global_params_output_df = pd.DataFrame(np.nan,
                                                    columns=['model'] + [*self._columns.keys()],
                                                    index=range(0, total_rows))
        # Repeat each model name rows_per_model times
        model_names_repeated = np.repeat(model_names, rows_per_model)
        # Assign the repeated model names to the 'variable' column
        self.global_params_output_df['model'] = model_names_repeated
        # Remove the firsts occurrence of 'BaseModel' in the 'variable' column
        while (self.global_params_output_df[self.global_params_output_df['model'] == self.__base_model_name].shape[0] >
               len(self.base_features)):
            # print('potato')
            self.global_params_output_df = self.global_params_output_df.drop(
                self.global_params_output_df[self.global_params_output_df['model'] == self.__base_model_name].index[0])
            self.global_params_output_df.reset_index(drop=True,
                                                     inplace=True)
        self.global_params_output_df = self.global_params_output_df.astype(self._columns)

    def _get_base_model_name(self):
        return self.__base_model_name
    def _populate_output_df(self):
        """
        Populate the output dataframe with the results of the current OLS model and extra metrics
        :return:
        """
        common_columns = self.output_df.columns.intersection(self.summary_parameters_df.columns)
        self.output_df.loc[self.output_df['model'] == self._iter_feature,
        common_columns] = self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
                                                         self._iter_feature, common_columns].values

        # allocate the model results
        common_columns = self.output_df.columns.intersection(self.summary_ols_results_df.columns)
        self.output_df.loc[self.output_df['model'] == self._iter_feature,
        common_columns] = self.summary_ols_results_df[common_columns].values

        # compute the odds ratio and return a dataframe
        odds = self._compute_odds(
            coeff=self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
                                                 self._iter_feature, 'coef'].values[0],
            ci_low=self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
                                                  self._iter_feature, '[0.025'].values[0],
            ci_high=self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
                                                   self._iter_feature, '0.975]'].values[0],
            return_type='frame'
        )

        common_columns = self.output_df.columns.intersection(odds.columns)
        self.output_df.loc[self.output_df['model'] == self._iter_feature,common_columns] = odds[common_columns].values

        self.output_df.loc[self.output_df['model'] == self._iter_feature,'mse'] = self.mse

        # iter_ = self._get_iter_feature()
        # TODO include the responses count
        response_count = self.global_params_output_df.loc[
            (self.global_params_output_df['model'] == self._iter_feature) & 
            (self.global_params_output_df['variable'] == self._iter_feature), 
            'responses_count'].values[0]

        self.output_df.loc[self.output_df['model'] == self._iter_feature, 'responses_count'] = response_count

    def _populate_all_models_params_output(self):
        """
        Populate the all models parametes dataframe using the commun columns between the summaries and the
        pre-allocated frame
        :return:
        """
        indexes_global = np.where(self.global_params_output_df['model'] == self._iter_feature)[0]
        summary_dim = self.summary_parameters_df.shape[0]

        # Handle dimension miss-match (few cases when this happens, and usually single row difference)
        if len(indexes_global) > summary_dim:
            # we need to drop rows from the global output
            while len(indexes_global) > summary_dim:
                self.global_params_output_df = self.global_params_output_df.drop(
                    self.global_params_output_df[self.global_params_output_df['model'] == self._iter_feature].index[
                        0])
                self.global_params_output_df.reset_index(drop=True,
                                                         inplace=True)
                indexes_global = np.where(self.global_params_output_df['model'] == self._iter_feature)[0]
        if len(indexes_global) < summary_dim:
            # we need to add rows to the global output
            while len(indexes_global) < summary_dim:
                first_matching_index = \
                self.global_params_output_df[self.global_params_output_df['model'] == self._iter_feature].index[0]
                new_row = self.global_params_output_df.loc[[first_matching_index]].copy()
                top_half = self.global_params_output_df.loc[:first_matching_index]
                bottom_half = self.global_params_output_df.loc[first_matching_index+1:]
                self.global_params_output_df = pd.concat([top_half, new_row, bottom_half], axis=0,
                                                         ignore_index=True)
                indexes_global = np.where(self.global_params_output_df['model'] == self._iter_feature)[0]

        # allocate the parameters results
        common_columns = self.global_params_output_df.columns.intersection(self.summary_parameters_df.columns)
        self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
        common_columns] = self.summary_parameters_df[common_columns].values

        # allocate the model results
        common_columns = self.global_params_output_df.columns.intersection(self.summary_ols_results_df.columns)
        self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
        common_columns] = self.summary_ols_results_df[common_columns].values

        odds = self._compute_odds(
            coeff=self.summary_parameters_df['coef'],
            ci_low=self.summary_parameters_df['[0.025'],
            ci_high=self.summary_parameters_df['0.975]'],
            return_type='frame'
        )

        common_columns = self.global_params_output_df.columns.intersection(odds.columns)
        self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
        common_columns] = odds[common_columns].values

        self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
        'mse'] = self.mse

        # TODO: its crearing multiple x and y columns when it should not
        # 'base_model'
        self.global_params_output_df = self.global_params_output_df.merge(
            self.responses_count_df,
            on=['model', 'variable'],  # The common columns
            how='left'  # Use 'left' to keep all records from `global_params_output_df`
        )
        # iter_feature = self._get_iter_feature()
        self.global_params_output_df['responses_count'] = self.global_params_output_df.apply(
            lambda row: self._combine_responses_count(row, col_alias='responses_count'), axis=1)

        self.global_params_output_df.drop(columns=['responses_count_x', 'responses_count_y'],
                                          inplace=True)

    @staticmethod
    def _combine_responses_count(row, 
                                 col_alias:str='responses_count') -> Union[pd.Series, float]:
        """
         Dataframe with two columns containing nan and values. We want to make them into a single one without loss of
        data of either column.
        :param row:
        :param col_alias: str, column aliases that has the suffixes _x and _y
        :return:
        """
        col_x = row[f'{col_alias}_x']
        col_y = row[f'{col_alias}_y']
        # If both 'col_x' and 'col_y' are NaN, return to NaN
        if pd.isna(col_x) and pd.isna(col_y):
            return np.nan
        # If only 'col_x' is not NaN, return col_x
        elif pd.notna(col_x) and pd.isna(col_y):
            return col_x
        # If only 'col_y' is not NaN, return col_y
        elif pd.isna(col_x) and pd.notna(col_y):
            return col_y
        # If both 'mrn_x' and 'mrn_y' have non-NaN values, return -1
        elif pd.notna(col_x) and pd.notna(col_y):
            return -1
        else:
            return -2

    def _mkdir(self):
        """
        Create the nested directory to save the results of the current model.
        results/<trial_name>/<model_results>/.....[sub folders, one per feature]
        :return:
        """
        self.trial_folder = self.out_path.joinpath(self.trial_name)
        if self.trial_folder.exists():
            overwrite = input("Directory already exists. Do you want to overwrite it? (1 for yes, 0 for no): ")
            if overwrite == '1':
                shutil.rmtree(str(self.trial_folder))  # Remove the existing directory
            else:
                self.trial_folder = self.trial_folder.joinpath('_2')
                print(f'Trial being saved in folder {self.trial_folder.name}')
        # wire the directory for the new trial
        self.trial_folder.mkdir(parents=True, exist_ok=True)
        if not self.evaluate_single_coefficient:
            # we will not be saving the results of each model in a separate folder
            self.models_folders = self.trial_folder.joinpath('ModelResults')
            self.models_folders.mkdir(parents=True, exist_ok=True)
            sub_folders = [self.__base_model_name] + self.iterative_features
            for iter_feat in sub_folders:
                self.models_folders.joinpath(iter_feat).mkdir(parents=True, exist_ok=True)

    def _save_global_output(self):
        """Save the dataframes in the specific directory"""

        output_path = self.trial_folder.joinpath('SingCoeffSumm.xlsx')
        self.output_df.insert(0, 'Include in Next base', 0)
        self.output_df.sort_values(by='P>|t|', inplace=True)

        self.output_df.to_excel(output_path, index=False)
        print(f"Successfully saved with pandas at {output_path}")
        self._check_and_print_file_existence(output_path, display=False)

        # Handle saving of global parameters output
        global_params_output_path = self.out_path.joinpath('GlobOutput.xlsx')
        self.global_params_output_df.to_excel(global_params_output_path, index=False)
        print(f"Successfully saved global parameters with pandas at {global_params_output_path}")
        self._check_and_print_file_existence(global_params_output_path, display=False)


        # output_path = self.trial_folder.joinpath('single_coefficient_summary.xlsx')
        # global_params_output_path = self.trial_folder.joinpath('global_output.xlsx')
        # # include new column to mark which ones will remain and which ones will not
        # self.output_df.insert(0, 'Include in Next base', 0)
        # self.output_df.sort_values(by='P>|t|', inplace=True)
        #
        # self.output_df.to_excel(output_path)
        # self.global_params_output_df.to_excel(global_params_output_path)
        #
        # self._check_and_print_file_existence(output_path, display=True)
        # self._check_and_print_file_existence(global_params_output_path, display=True)

    def _save_current_model_results(self):
        """
        Each OLS result is saved in a separate folder as a .csv file. The folder name follows the name of the
        feature we are iterating with self._iter_feature.
        Saves the model summary and the estimated parameters
        :return:
        """
        if not self.evaluate_single_coefficient:
            summary_ols_results_path = self.models_folders.joinpath(self._iter_feature, 'SummOlsRes.csv')
            summary_parameters_path = self.models_folders.joinpath(self._iter_feature, 'SummOlsParams.csv')
            summary_distribution_tests_path = self.models_folders.joinpath(self._iter_feature,
                                                                           'SummDistTests.csv')

            # summary_ols_results_df = pd.DataFrame([self.__summary_ols_results_dict])
            self.summary_ols_results_df.to_csv(summary_ols_results_path,
                                               index=False)
            self.summary_parameters_df.to_csv(summary_parameters_path,
                                              index=False)
            self.summary_distribution_tests_df.to_csv(summary_distribution_tests_path,
                                                      index=False)

            # Check if files exist and print messages
            # self._check_and_print_file_existence(summary_ols_results_path)
            # self._check_and_print_file_existence(summary_parameters_path)
            # self._check_and_print_file_existence(summary_distribution_tests_path)
        else:
            return

    def _check_and_print_file_existence(self, 
                                        file_path:pathlib.Path, 
                                        display:Optional[bool]=False):
        """Check if files exist and print messages"""
        if file_path.exists():
            if display:
                print(f"Success Saving File '{file_path.name}' in \n\t '{file_path}' exists.")
        else:
            raise ValueError(f"Error Saving File '{file_path.name}' in \n\t '{file_path}'")

    def get_iteration_results(self) -> pd.DataFrame:
        return self.output_df


class SelectNextBase:
    def __init__(self,
                 output_path: pathlib.Path,
                 to_ignore_significance: Optional[list[str]] = None,
                 criteria: str = 'P>|t|',
                 # direction: str = 'minimize',
                 column_variables: str = 'variable',
                 eta: Optional[float] = None,
                 odds_bounds_columns: Optional[list] = None,
                 variable_to_ignore_in_base: Optional[list] = None,
                 alpha: float = None, ):
        """
                From the output of IterativeRegression, which is a frame with p values, OR, OR CI, variable name, etc.
        This class will determine which one is the next best more suitable variable to adjust from the iterative
        variables.

        The method select_base() reads the frame, sorts by p value from smallest to largest, then filters out
        any variable where the OR CI contains 1. From the selection it will get the iterative value with smallest p
        value and return it appended to the current base variable. This way we get the next base

        Call the save_base() method at the end, so we get an Excel of the traceability of bases.

        :param output_path: athlib.Path, path to save the table results

        :param to_ignore_significance: list[str], variable that we want to account for but noe include in the model
            selection. These variables will be counted as how many times they were candidates for significance,
            but they will not be included in the model
        :param criteria: str, which column of the result from the model we want to evaluate

        :param column_variables: str, name of the column where are the variable from the iterative features

        :param eta: float, stopping criteria to criteria magnitude and avoid overfitting. If None, we use the alpha
            and Bonferroni correction

        :param odds_bounds_columns: list[str], list with the names of the OR CI column names

        :param variable_to_ignore_in_base: Optional[list[str]], in case we want to avoid placing confounders as
            significant we can pass them here as a list and they will be ingored in the selection

        :param alpha: float, level of significance
        """
        # if direction not in ['minimize', 'maximize']:
        #     raise ValueError(f"Invalid value for direction: {direction}. Choose 'minimize' or 'maximize'.")

        if output_path.suffix not in ['.xlsx', '.csv']:
            raise ValueError(f"Invalid output file type: {output_path.suffix}. Choose '.xlsx' or '.csv'.")

        if alpha is None and eta is None:
            raise ValueError(f'At least one criteria must be implemented, alpha or eta')

        self.criteria_values = {}
        # self.direction = direction
        self.criteria = criteria
        self.column = column_variables
        if to_ignore_significance is not None:
            self.to_ignore = {key: count for key, count in zip(to_ignore_significance,[0]*len(to_ignore_significance) )}
        else:
            self.to_ignore = None
        self.eta = eta
        self.alpha = alpha
        self.next_base = None
        self.ignored_df = None
        self.base_register = {}
        self._base_counter = 0
        self.output_path = output_path
        if odds_bounds_columns is None or len(odds_bounds_columns) != 2:
            self.odds_bounds_columns = ['odds_ci_low', 'odds_ci_high']
        else:
            self.odds_bounds_columns = odds_bounds_columns

    def _avoid_unwanted_variables(self, next_candidate:str) -> bool:
        """
        There are some column that would not like to include as base in our model because of domain knowledge,
        here we will check on that. We will also count how many times that variable was a candidate so we can decide
        how important it was.
        :param next_candidate: str, next possible candidate that we will check if its not in the list of to_ignore
        :return:
            bool, True if the candidate is in the to ignore list
        """
        if next_candidate in self.to_ignore.keys():
            self.to_ignore[next_candidate] += 1
            return True
        else:
            return False

    def select_base(self,
                    iterations_result:pd.DataFrame,
                    current_base:list) -> list:
        """
        From the current base and the results of p values of each iterative variable in the model, determine
        which ones is the most suitable candidated based on the criteria.

        :param iterations_result: pd.Dataframe, results from the model obtained from the iteraive class
        :param current_base: list[str], list of variables names used in the current adjustment for the model
        :return: [] if the criteria for signficane is fulliled and we should stop, else returns a list of strings with
        the next base of the model
        """
        # if self.direction == 'minimize':
        # Sort in descending order to get the smallest values on top
        candidate_row_idx: int = 0
        iterations_result.sort_values(by=self.criteria,
                                 ascending=True,
                                 inplace=True)
        # Odds ratio criteria, should not contain 1
        filtered_df = iterations_result[(iterations_result[self.odds_bounds_columns[0]] > 1) |
                                        (iterations_result[self.odds_bounds_columns[1]] < 1)]
        filtered_df.reset_index(inplace=True, drop=True)
        # Stopping criteria: check if the smallest p-value is greater than eta
        if self._stopping_criteria(criteria_val=filtered_df.at[candidate_row_idx, self.criteria]):
            return []

        # else:
        #     # Sort in ascending order to get the greatest values on top
        #     iterations_result.sort_values(by=self.criteria,
        #                              ascending=False,
        #                              inplace=True)
        #     # Odds ratio criteria, should not contain 1
        #     filtered_df = iterations_result[(iterations_result[self.odds_bounds_columns[0]] > 1) |
        #                                     (iterations_result[self.odds_bounds_columns[1]] < 1)]
        #     filtered_df.reset_index(inplace=True, drop=True)
        #     # Stopping criteria: check if the smallest p-value is less than eta
        #     if self._stopping_criteria(criteria_val=filtered_df.at[0,self.criteria]):
        #         return []
        # Odds ratio criteria, should not contain 1
        filtered_df.reset_index(inplace=True, drop=True)

        # from there the CI of the OR does not contain 1, return the variable with smallest p value, which is on top
        self.base_register[f"{self._base_counter}_{current_base[-1]}"] = current_base
        self._base_counter += 1
        # Include in the current the next base candidate to adjust in the model
        next_base = current_base.copy()
        base_candidate = filtered_df.at[candidate_row_idx, self.column]
        if self.to_ignore is not None:
            # while the next base candidate is in the to ignore list, go for the next row and check if good candidate
            while self._avoid_unwanted_variables(next_candidate=base_candidate):
                # TODO: drop the row of the frame instead, so we do not see it in the results
                candidate_row_idx += 1
                base_candidate = filtered_df.at[candidate_row_idx, self.column]
                if self._stopping_criteria(criteria_val=filtered_df.at[candidate_row_idx, self.criteria]):
                    return []
        # include the base candidate in the next base
        next_base.append(base_candidate)
        self.criteria_values[filtered_df.at[candidate_row_idx, self.column]] = filtered_df.at[candidate_row_idx, self.criteria]
        # save each iteration and re-write file to avoid losing all the data tracks
        self.save_base()
        if self.to_ignore is not None:
            self.save_to_ignore_count()
        if self._stopping_criteria(num_pooled_variables=iterations_result.shape[0],
                                   criteria_val=filtered_df.at[candidate_row_idx, self.criteria]):
            return []
        return next_base

    def _stopping_criteria(self, criteria_val: float,
                           num_pooled_variables:Optional[int]=None) -> bool :
        """
        Wrapped to call the different stopping criteria methods. Currently, the Bonferroni is the one implemented
        :param criteria_val: float, criteria value e.g., p value
        :param num_pooled_variables: int, number of pooled variables if the Bonferroni correction for p value is going to
        be implemented
        :return: True if the stopping criteria needs to be implemented
        """
        if self.eta is not None and criteria_val is not None:
            # if self.direction == 'minimize':
            if criteria_val <= self.eta:
                return True
            else:
                return False
            # else:
            #     if criteria_val <= self.eta:
            #         return True
        if num_pooled_variables is not None and criteria_val is not None:
            return self._bonferroni_correction(num_pooled_variables=num_pooled_variables, p_value=criteria_val)

    def _bonferroni_correction(self, num_pooled_variables: int, p_value: float) -> bool:
        """
        Implement the Bonferroni correction
                α_new = α_original / n

        :param num_pooled_variables: int, The total number of comparisons or tests being performed
        :param p_value: float, the obtained p value in the best candidate e.g., the smallest p value we got
        :return: True if we need to stop the loop
        """
        # apply Bonferroni correction
        alpha_corr = self.alpha / num_pooled_variables
        print('\nBonferroni Correction Procedure:\n')
        print(f'Initial significance level (alpha): {self.alpha}')
        print(f'Number of tests (i.e., number of variables in the pool): {num_pooled_variables}')
        print(f'Corrected significance level (alpha_corr = alpha / number of tests): {alpha_corr}\n')
        print(f'Obtained p-value: {p_value}')

        if p_value > alpha_corr:
            print(f'\nThe obtained p-value is greater than the corrected significance level. '
                  f'The stopping condition met, stop further tests.\n')
            return True  # stop if min p-value is greater than corrected alpha
        else:
            print(f'\nThe obtained p-value is less than the corrected significance level. Continue further tests.\n')
            return False

    def save_base(self) -> pd.DataFrame:
        """
        Parse the dictionary of lists as a DataFrame and save it.

        This function equalizes the list lengths in the self.base_register dictionary, pads them with None so we have
        same length among all the columns when constructing the frame, and saves the resulting DataFrame to a file
        specified by self.output_path.
        :return:
            base register
        """

        # Find the length of the longest list
        max_len = max(len(val_list) for val_list in self.base_register.values())

        # Create a new dictionary where all lists have been padded to the same length
        padded_base_register = {
            key: val_list + [None] * (max_len - len(val_list))
            for key, val_list in self.base_register.items()
        }

        # Convert to DataFrame
        df = pd.DataFrame(padded_base_register)

        # Write the DataFrame to a file
        if self.output_path.suffix == '.xlsx':
            df.to_excel(self.output_path, index=False)
        else:
            df.to_csv(self.output_path, index=False)
        return df

    def save_to_ignore_count(self):
        """Save the counts as a dataframe in the folder"""
        ignore_path_file = self.output_path.parent.joinpath('ignored_col_count.xlsx')
        self.ignored_df = pd.DataFrame(list(self.to_ignore.items()), columns=['names_to_ignore', 'count'])
        self.ignored_df.sort_values(by='count', ascending=False, inplace=True)
        self.ignored_df.to_excel(ignore_path_file, index=False)

    def get_ignored_counts(self) -> pd.DataFrame:
        return self.ignored_df


# from typing import Tuple
#
# def combine_responses_count(col_x:pd.Series,col_y:pd.Series) -> Union[float, Tuple[float, float], int]:
#     """
#     Dataframe with two columns containing nan and values. We want to make them into a single one without loss of
#     data of either column.
#     Usage:
#             asq_fil_mlog_df["mrn"] = asq_fil_mlog_df.apply(
#                     lambda row: combine_mrn(row["mrn_x"], row["mrn_y"]), axis=1)
#     :param col_x:
#     :param col_y:
#     :param conflict:str, default both, When merging MRNS columns, both column might have a value
#         if conflict = both
#             returns tuple with value of both columns
#         if conflict = deduce
#             return float, this method is for a common error when both mrns are the same but one has an extra
#             decimal place bcause of the way it was stored. The one with the extra was stored as a string
#     :return:
#     """
#     # col_x = frame_x_y[0]
#     # col_y = frame_x_y[1]
#     # If both 'col_x' and 'col_y' are NaN, set 'mrn' to NaN
#     if pd.isna(col_x) and pd.isna(col_y):
#         return np.nan
#     # If only 'col_x' is not NaN, return col_x
#     elif pd.notna(col_x) and pd.isna(col_y):
#         return col_x
#     # If only 'col_y' is not NaN, return col_y
#     elif pd.isna(col_x) and pd.notna(col_y):
#         return col_y
#     # If both 'mrn_x' and 'mrn_y' have non-NaN values, check if they are equal
#     elif pd.notna(col_x) and pd.notna(col_y):
#         return -1
#     else:
#         return -2


# class IterativeRegression:
#     """
#     RegressionModel class that iterates over iterative features, runs linear regression models with sm.OLS,
#     collects metrics, and builds a DataFrame with the model results.
#     """
#
#     def __init__(self, target: pd.Series,
#                  base_features: pd.DataFrame,
#                  iterative_features: pd.DataFrame,
#                  out_path: pathlib.Path,
#                  trial_name: str,
#                  re_write_dir: bool = True):
#         """
#
#         :param target: pd.Series, dependent variable
#         :param base_features: pd.DataFrame, independent variables we that will always be present in the model
#         :param iterative_features: pd.DataFrame, independent variables that will be iterated one by one
#         :param out_path: pathlib.Path, folder directory to save the model results
#         :param trial_name: str, name of the model trial
#         :param re_write_dir: bool, if to re-write an existing directory (True) or not (False)
#         """
#         self.base_features_df = base_features
#         self.base_features_df.columns = [col.strip() for col in base_features.columns]
#         self.iterative_features_df = iterative_features
#         self.iterative_features_df.columns = [col.strip() for col in iterative_features.columns]
#         self.target = target
#         self.out_path = out_path
#         self.trial_name = trial_name
#         self.trial_folder = None
#         self.models_folders = None
#         self.output_df = None
#         self.mse = None
#         self.__summary_ols_results_dict = None
#         self.summary_parameters_df = None
#         self.summary_distribution_tests_df = None
#         self._iter_feature = None
#         self.odds = None
#         self.__base_model_name = 'base_model'
#         self._remove_rows()
#         if re_write_dir:
#             self._mkdir()
#         self._create_output_table()
#         self._create_all_models_parameters_output()
#
#     def fit_iterative_models(self):
#         """
#         Fir the base model and iterate over the iterative features and evaluate each model models
#         :return:
#         """
#         # base model is always evalauted
#         self._fit_base_model()
#         self._populate_all_models_params_output()
#
#         for self._iter_feature in tqdm(self.iterative_features_df.columns, desc="Fitting Models"):
#             observations_idx = self.iterative_features_df[
#                 ~self.iterative_features_df[self._iter_feature].isna()].index
#
#             current_features = pd.concat([self.base_features_df.loc[observations_idx, :],
#                                           self.iterative_features_df.loc[observations_idx, self._iter_feature]
#                                           ],
#                                          axis=1).copy()
#             self._fit_and_evaluate_model(features=current_features,
#                                          target=self.target[observations_idx])
#
#             self._populate_output_df()
#             self._populate_all_models_params_output()
#             self._save_current_model_results()
#
#         self._save_global_output()
#
#     def _fit_base_model(self):
#         """Fit and evaluate the base model"""
#         self._iter_feature = self.__base_model_name
#         self._fit_and_evaluate_model(self.base_features_df,
#                                      target=self.target)
#
#     def _fit_and_evaluate_model(self,
#                                 features: pd.DataFrame,
#                                 target: pd.Series):
#         """
#         Method dedicate to fit the logistic regression model with the given features.
#         :param features: pd.Dataframe, dependent variables of the model
#         :param target: pd.Series, independent variable of the model
#         :return: None
#         """
#         model = sm.OLS(endog=target,
#                        exog=sm.add_constant(features),
#                        missing='drop').fit()
#         predictions = model.predict(sm.add_constant(features))
#
#         self._summary_to_structured_object(model=model,
#                                            summary_ols_results=model.summary().tables[0].as_csv(),
#                                            summary_parameters=model.summary().tables[1].as_csv(),
#                                            summary_distribution_tests=model.summary().tables[2].as_csv())
#
#         self.mse = mean_squared_error(y_true=target,
#                                       y_pred=predictions)
#
#     @staticmethod
#     def _compute_odds(coeff: Union[float, pd.Series],
#                       ci_low: Union[float, pd.Series],
#                       ci_high: Union[float, pd.Series],
#                       return_type: Optional[str] = 'dict'
#                       ) -> Union[dict, pd.DataFrame]:
#         """
#         Compute the odds form the coefficients of a logistic regression model
#         :param coeff: float, beta coefficient from the model
#         :param return_type: str, return type, frame or dict
#         :param ci_low: float, lower bound of the confidence interval of the coefficient
#         :param ci_high: float, higher bound of the confidence interval of the coefficient
#         :return:
#             dict, odds, odds  confidence intervals
#
#         Usage:
#             with implementation = 'apply'
#
#             result_df = self.global_params_output_df['coef'].apply(calculate_odds_and_ci)
#
#         """
#         # For when we want to define the alpha: we must use the standard error
#         # odds_df['odds'] = coef_df['coef'].apply(lambda x: np.round(np.exp(x), 2))
#         #
#         # # Assuming coef_df is a DataFrame with a column 'std_err' representing the standard error
#         # odds_df['odds_ci_low'] = coef_df.apply(lambda row: np.round(np.exp(row['coef'] - 1.96 * row['std_err']), 2),
#         #                                        axis=1)
#         # odds_df['odds_ci_high'] = coef_df.apply(lambda row: np.round(np.exp(row['coef'] + 1.96 * row['std_err']), 2),
#         #                                         axis=1)
#
#         if isinstance(coeff, pd.Series) and isinstance(ci_low, pd.Series) and isinstance(ci_high, pd.Series):
#             odds = coeff.apply(lambda x: np.round(np.exp(x), 2))
#             odds_ci_low = ci_low.apply(lambda x: np.round(np.exp(x), 2))
#             odds_ci_high = ci_high.apply(lambda x: np.round(np.exp(x), 2))
#
#             return pd.DataFrame({'odds': odds, 'odds_ci_low': odds_ci_low, 'odds_ci_high': odds_ci_high})
#
#         else:
#             results = {
#                 'odds': np.round(np.exp(coeff), 2),
#                 'odds_ci_low': np.round(np.exp(ci_low), 3),
#                 'odds_ci_high': np.round(np.exp(ci_high), 3),
#
#             }
#             if return_type == 'frame':
#                 return pd.DataFrame(results, index=[0])
#             else:
#                 return results
#
#     @staticmethod
#     def _calculate_odds_ratio_ci(coeff: Union[float, pd.Series],
#                                  str_err: Union[float, pd.Series],
#                                  alpha: float = 0.05) -> Union[dict, pd.DataFrame]:
#         """
#         Compute the odds ratio and confidence intervals given the beta coefficient, standard deviation, and alpha
#         :param coeff: float or pd.Series, beta coefficient
#         :param str_err: float or pd.Series, standard error
#         :param alpha: float, significance level
#         :return: dict or pd.DataFrame, odds ratio and confidence intervals
#         """
#
#         def calculate_single_odds_ratio_ci(single_coeff, single_str_err):
#             # Calculate the odds from the coefficient
#             odds = np.round(np.exp(single_coeff), 2)
#
#             # Calculate the z-score based on alpha (two-tailed test)
#             z_score = stats.norm.ppf(1 - alpha / 2)
#
#             # Calculate the margin of error
#             margin_of_error = z_score * single_str_err
#
#             # Calculate confidence interval
#             lower_ci = odds - margin_of_error
#             upper_ci = odds + margin_of_error
#
#             return {
#                 'odds': odds,
#                 'odds_ci_low': np.round(lower_ci, 3),
#                 'odds_ci_high': np.round(upper_ci, 3),
#             }
#
#         if isinstance(coeff, pd.Series) and isinstance(str_err, pd.Series):
#             # If input is Series, apply the function to each element
#             return pd.DataFrame([calculate_single_odds_ratio_ci(c, se) for c, se in zip(coeff, str_err)],
#                                 columns=['odds', 'odds_ci_low', 'odds_ci_high'])
#
#         else:
#             # If input is single values, apply the function to the single values
#             return calculate_single_odds_ratio_ci(coeff, str_err)
#
#     def _remove_rows(self):
#         """
#         Base features must not contain nans, re move the rows with nans from the base and the features we will
#         iterate
#         :return:
#         """
#         # Get indexes where any column has NaN values
#         indexes_with_nan = self.base_features_df[self.base_features_df.isna().any(axis=1)].index
#         self.base_features_df.drop(index=indexes_with_nan, inplace=True)
#         # self.base_features_df.reset_index(inplace=True, drop=True)
#
#         self.iterative_features_df.drop(index=indexes_with_nan, inplace=True)
#         # self.iterative_features_df.reset_index(inplace=True, drop=True)
#
#         if not self.base_features_df.shape[0] == self.iterative_features_df.shape[0]:
#             raise ValueError(
#                 f'Error IterativeRegression: Removing rows caused a difference in row length, it should '
#                 f'be the same')
#
#     def _summary_to_structured_object(self,
#                                       model,
#                                       summary_ols_results: str,
#                                       summary_parameters: str,
#                                       summary_distribution_tests: str):
#         """
#         Parse the results from statsmodels.iolib.summary.Summary to a DataFrame. This is specific to the format
#         the statsmodels api returns the summaries
#         :param model: sm.OLS model
#         :param summary_ols_results: str,
#         :param summary_parameters:  str,
#         :param summary_distribution_tests: str,
#         """
#         # Summary OLS results section
#         self.__summary_ols_results_dict = {}
#         summary_ols_results_cl = summary_ols_results.split('\n')[1::]
#         summary_ols_results_cl = [re.sub(r'(?<!\n)\s+', ' ', line) for line in summary_ols_results_cl]
#         for summary_lst in summary_ols_results_cl:
#             # Define a pattern to extract key-value pairs
#             pattern = re.compile(r'([^,:]+):\s*,\s*([^,]+)')
#             matches = re.findall(pattern, summary_lst)
#             # Update the result dictionary
#             for key, value in matches:
#                 self.__summary_ols_results_dict[key.strip()] = value.strip()
#
#         self.summary_ols_results_df = pd.DataFrame(self.__summary_ols_results_dict, index=[0])
#         self.summary_ols_results_df.drop(columns=['Method', 'Date', 'Time', 'Covariance Type'],
#                                          inplace=True)
#         # Summary Parameters section
#         # Use StringIO to simulate a file-like object
#         data_io = StringIO(summary_parameters)
#         self.summary_parameters_df = pd.read_csv(data_io)
#         self.summary_parameters_df.columns = [col.strip() for col in self.summary_parameters_df.columns]
#         # Rename the columns
#         self.summary_parameters_df.rename(columns={'': 'variable'}, inplace=True)
#         self.summary_parameters_df['variable'] = self.summary_parameters_df['variable'].apply(lambda x: x.strip())
#         # we need the exact p-values
#         self.summary_parameters_df.loc[:, 'P>|t|'] = model.pvalues.values
#
#         # Summary Distribution Test
#         # Add column names to the string
#         summary_distribution_tests_cl = 'Test,Value,Test,Value\n' + summary_distribution_tests
#         data_io = StringIO(summary_distribution_tests_cl)
#         self.summary_distribution_tests_df = pd.read_csv(data_io)
#         self.summary_distribution_tests_df.columns = [col.strip() for col in
#                                                       self.summary_distribution_tests_df.columns]
#
#     def _create_output_table(self):
#         """
#         Create the dataframe we will populate after each OLS regression result.
#         Each row of the dataframe is the result of a regression model, and the rows are marked by the current iterative
#         variable we are testing.
#         :return:
#         """
#         columns = ['variable', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]',
#                    'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
#                    'No. Observations', 'Dep. Variable']
#         extra_metrics = ['odds', 'odds_ci_low', 'odds_ci_high', 'mse']  # dependency with the odds ratio function
#         columns.extend(extra_metrics)
#         index_lbls = [self.__base_model_name]
#         index_lbls.extend(self.iterative_features_df.columns)
#         # self.output_df = pd.DataFrame(np.nan, columns=columns, index=index_lbls)
#         self.output_df = pd.DataFrame(np.nan, columns=['model'] + columns, index=range(0, len(index_lbls)))
#         self.output_df['model'] = index_lbls
#
#     def _populate_output_df(self):
#         """
#         Populate the output dataframe with the results of the current OLS model and extra metrics
#         :return:
#         """
#         common_columns = self.output_df.columns.intersection(self.summary_parameters_df.columns)
#         self.output_df.loc[self.output_df['model'] == self._iter_feature,
#         common_columns] = self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
#                                                          self._iter_feature, common_columns].values
#
#         # allocate the model results
#         common_columns = self.output_df.columns.intersection(self.summary_ols_results_df.columns)
#         self.output_df.loc[self.output_df['model'] == self._iter_feature,
#         common_columns] = self.summary_ols_results_df[common_columns].values
#
#         # compute the odds ratio and return a dataframe
#         odds = self._compute_odds(
#             coeff=self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
#                                                  self._iter_feature, 'coef'].values[0],
#             ci_low=self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
#                                                   self._iter_feature, '[0.025'].values[0],
#             ci_high=self.summary_parameters_df.loc[self.summary_parameters_df['variable'] ==
#                                                    self._iter_feature, '0.975]'].values[0],
#             return_type='frame'
#         )
#
#         common_columns = self.output_df.columns.intersection(odds.columns)
#         self.output_df.loc[self.output_df['model'] == self._iter_feature, common_columns] = odds[
#             common_columns].values
#
#         self.output_df.loc[self.output_df['model'] == self._iter_feature, 'mse'] = self.mse
#
#     def _create_all_models_parameters_output(self):
#         """
#         To test for confounders or association it's important to evaluate how the beta coefficients change in each model
#
#         :return:
#         """
#         columns = ['variable', 'coef', 'std err', 't', 'P>|t|', '[0.025', '0.975]',
#                    'Adj. R-squared', 'F-statistic', 'Prob (F-statistic)',
#                    'No. Observations', 'Dep. Variable']
#         extra_metrics = ['odds', 'odds_ci_low', 'odds_ci_high', 'mse']  # dependency with the odds ratio function
#         columns.extend(extra_metrics)
#         model_names = [self.__base_model_name]
#         model_names.extend(self.iterative_features_df.columns)
#         rows_per_model = self.base_features_df.shape[1] + 2  # constant and the new iterative feature
#         total_rows = rows_per_model * len(model_names)
#         self.global_params_output_df = pd.DataFrame(np.nan,
#                                                     columns=['model'] + columns,
#                                                     index=range(0, total_rows))
#         # Repeat each model name rows_per_model times
#         model_names_repeated = np.repeat(model_names, rows_per_model)
#         # Assign the repeated model names to the 'variable' column
#         self.global_params_output_df['model'] = model_names_repeated
#         # Remove the first occurrence of 'BaseModel' in the 'variable' column
#         self.global_params_output_df = self.global_params_output_df.drop(
#             self.global_params_output_df[self.global_params_output_df['model'] == self.__base_model_name].index[0])
#         self.global_params_output_df.reset_index(drop=True,
#                                                  inplace=True)
#
#     def _populate_all_models_params_output(self):
#         """
#         Populate the all models parametes dataframe using the commun columns between the summaries and the
#         pre-allocated frame
#         :return:
#         """
#         # allocate the parameters results
#         common_columns = self.global_params_output_df.columns.intersection(self.summary_parameters_df.columns)
#         self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
#         common_columns] = self.summary_parameters_df[common_columns].values
#
#         # allocate the model results
#         common_columns = self.global_params_output_df.columns.intersection(self.summary_ols_results_df.columns)
#         self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
#         common_columns] = self.summary_ols_results_df[common_columns].values
#
#         odds = self._compute_odds(
#             coeff=self.summary_parameters_df['coef'],
#             ci_low=self.summary_parameters_df['[0.025'],
#             ci_high=self.summary_parameters_df['0.975]'],
#             return_type='frame'
#         )
#
#         common_columns = self.global_params_output_df.columns.intersection(odds.columns)
#         self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
#         common_columns] = odds[common_columns].values
#
#         self.global_params_output_df.loc[self.global_params_output_df['model'] == self._iter_feature,
#         'mse'] = self.mse
#
#     def _mkdir(self):
#         """
#         Create the nested directory to save the results of the current model.
#         results/<trial_name>/<model_results>/.....[sub folders, one per feature]
#         :return:
#         """
#         self.trial_folder = self.out_path.joinpath(self.trial_name)
#         if self.trial_folder.exists():
#             overwrite = input("Directory already exists. Do you want to overwrite it? (1 for yes, 0 for no): ")
#             if overwrite == '1':
#                 shutil.rmtree(str(self.trial_folder))  # Remove the existing directory
#             else:
#                 self.trial_folder.joinpath('_2')
#                 print(f'Trial being saved in folder {self.trial_folder.name}')
#         # wire the directory for the new trial
#         self.trial_folder.mkdir(parents=True, exist_ok=True)
#         self.models_folders = self.trial_folder.joinpath('model_results')
#         self.models_folders.mkdir(parents=True, exist_ok=True)
#         sub_folders = [self.__base_model_name]
#         sub_folders.extend(self.iterative_features_df.columns)
#         for iter_feat in sub_folders:
#             self.models_folders.joinpath(iter_feat).mkdir(parents=True, exist_ok=True)
#
#     def _save_global_output(self):
#         """Save the dataframes in the specific directory"""
#         output_path = self.trial_folder.joinpath('single_coefficient__summary.xlsx')
#         global_params_output_path = self.trial_folder.joinpath('global_output.xlsx')
#         # include new column to mark which ones will remain and which ones will not
#         self.output_df.insert(0, 'Include in Next base', 0)
#         self.output_df.to_excel(output_path)
#         self.global_params_output_df.to_excel(global_params_output_path)
#
#         self._check_and_print_file_existence(output_path, display=True)
#         self._check_and_print_file_existence(global_params_output_path, display=True)
#
#     def _save_current_model_results(self):
#         """
#         Each OLS result is saved in a separate folder as a .csv file. The folder name follows the name of the
#         feature we are iterating with self._iter_feature.
#         Saves the model summary and the estimated parameters
#         :return:
#         """
#         summary_ols_results_path = self.models_folders.joinpath(self._iter_feature, 'summary_ols_results.csv')
#         summary_parameters_path = self.models_folders.joinpath(self._iter_feature, 'summary_parameters.csv')
#         summary_distribution_tests_path = self.models_folders.joinpath(self._iter_feature,
#                                                                        'summary_distribution_tests.csv')
#
#         # summary_ols_results_df = pd.DataFrame([self.__summary_ols_results_dict])
#         self.summary_ols_results_df.to_csv(summary_ols_results_path,
#                                            index=False)
#         self.summary_parameters_df.to_csv(summary_parameters_path,
#                                           index=False)
#         self.summary_distribution_tests_df.to_csv(summary_distribution_tests_path,
#                                                   index=False)
#
#         # Check if files exist and print messages
#         # self._check_and_print_file_existence(summary_ols_results_path)
#         # self._check_and_print_file_existence(summary_parameters_path)
#         # self._check_and_print_file_existence(summary_distribution_tests_path)
#
#     def _check_and_print_file_existence(self, file_path: pathlib.Path, display: Optional[bool] = False):
#         """Check if files exist and print messages"""
#         if file_path.exists():
#             if display:
#                 print(f"Success Saving File '{file_path.name}' in \n\t '{file_path}' exists.")
#         else:
#             raise ValueError(f"Error Saving File '{file_path.name}' in \n\t '{file_path}'")
#
