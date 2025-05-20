"""
Classs dedicated to create teh table one of the paper

Author: Giorgio Ricciardiello
contact: giocrm@stanford.edu
Year: 2024

"""

import numpy as np
import pandas as pd
import re
from typing import Union, List, Optional, List, Dict
from scipy.stats import fisher_exact, mannwhitneyu, shapiro, ttest_ind, chi2_contingency,f_oneway, kruskal
from tabulate import tabulate

class MakeTableOne:
    def __init__(self, df: pd.DataFrame,
                 continuous_var: list[str],
                 categorical_var: list[str],
                 strata: Optional[str]=None,
                 index_mapper:Dict[str,Dict[int,str]]=None):
        """
        Class to easily create a table one from the dataframe.
        Important: The columns must not be of mixed data types e.g., strings and numeric. Only one type is accepted
        nans are ignored in all counts
        :param df:
        :param continuous_var:
        :param categorical_var:
        :param strata:

        usage:

            continuous_var = ['Age', 'BMI', 'ESS']
            categorical_var = ['sex', 'DQB10602', 'hla_positive']
            column_groups = 'Race'

            tableone_constructor = MakeTableOne(df=df_race,
                                            continuous_var=continuous_var,
                                            categorical_var=categorical_var,
                                            strata=column_groups)

            table_one = tableone_constructor.create_table()

                      variable      Caucasian  ...         Mixed       Other
            0            Count            512  ...             9           1
            1              Age  45.17 (16.39)  ...  31.44 (8.13)  30.0 (nan)
            2              BMI     28.0 (6.6)  ...  23.26 (3.63)  30.3 (nan)
            3              ESS   16.92 (4.59)  ...  16.78 (2.91)  16.0 (nan)
            4           sex__0   274 (53.52%)  ...    7 (77.78%)    0 (0.0%)
            5           sex__1   238 (46.48%)  ...    2 (22.22%)  1 (100.0%)
            6      DQB10602__0   225 (43.95%)  ...    3 (33.33%)  1 (100.0%)
            7      DQB10602__1    278 (54.3%)  ...    6 (66.67%)    0 (0.0%)
            8  hla_positive__0   161 (31.45%)  ...      0 (0.0%)  1 (100.0%)
            9  hla_positive__1   351 (68.55%)  ...    9 (100.0%)    0 (0.0%)

        """
        self.df = df
        self.continuous_var = continuous_var
        self.categorical_var = categorical_var
        self.strata = 'SingleDistributionTable' if strata is None else strata
        self.index = continuous_var + categorical_var
        self._check_columns_input()
        self.index_categorical = None
        self.index_mapper = index_mapper if index_mapper is not None else {}
        self.tableone = self._create_empty_table()

    def create_table(self) -> pd.DataFrame:
        """Pipeline to create the table"""
        self._populate_count()
        self._populate_continuous_columns()
        self._populate_categorical_columns()
        if not self.index_mapper is None:
            self._re_map_indexes_to_strings()

        print(tabulate(self.tableone,
                       headers=[*self.tableone.columns],
                       showindex=False,
                       tablefmt="fancy_grid"))
        return self.tableone

    def _check_columns_input(self):
        """Check if all the inputs are in the dataset"""
        for col in self.index:
            if '__' in col:
                raise ValueError(f"Column {col} has __ in it's name, this will break the code."
                                 f" Please rename the column without the __ ")
            if not col in self.df.columns:
                raise ValueError(f'Column {col} is not in the dataframe')

    def _create_categorical_row_indexes(self) -> list[str]:
        """
        Categorical, ordinal, discrete columns will have one row per unique value with their count
        spawned over the columns. Here we create the row indexes for each of these columns
        :return:
        """
        index_categorical = []
        for cat_col in self.categorical_var:
            index_categorical.extend(self._create_index(self.df, col=cat_col))
        return index_categorical

    def _create_empty_table(self) -> pd.DataFrame:
        """
        Create the empty table one. The columns are the unique values of the strata and the rows are the
        continuous variables + the categorical variables expanded to all their unique values
        :return:
        """
        self.index_categorical = self._create_categorical_row_indexes()
        indexes = ['Count'] + self.continuous_var + self.index_categorical
        if self.strata != 'SingleDistributionTable':
            tableone = pd.DataFrame(index=indexes, columns=self.df[self.strata].unique())
        else:
            tableone = pd.DataFrame(index=indexes, columns=[self.strata])
        tableone.index.name = 'variable'
        tableone.reset_index(inplace=True)
        return tableone

    def _populate_count(self):
        if self.strata == 'SingleDistributionTable':
            self.tableone.loc[self.tableone['variable'] == 'Count', self.strata] = self.df.shape[0]
        else:
            for column_group_ in self.df[self.strata].unique():
                self.tableone.loc[self.tableone['variable'] == 'Count', column_group_] = (
                    self.df[self.df[self.strata] == column_group_].shape)[0]

    def _populate_continuous_columns(self):
        """
        Strata are the columns. Moves through the columns and calculates the metrics (mean ± sd) of the variable
        which is a row.
        """
        if self.strata == 'SingleDistributionTable':
            for cont_var in self.continuous_var:
                self.tableone.loc[self.tableone['variable'] == cont_var,
                self.strata] = self._continuous_var_dist(frame=self.df, col=cont_var)
        else:
            for column_group_ in self.df[self.strata].unique():
                for cont_var in self.continuous_var:
                    self.tableone.loc[self.tableone['variable'] == cont_var,
                    column_group_] = self._continuous_var_dist(frame=self.df[self.df[self.strata] == column_group_],
                                                               col=cont_var)

    def _populate_categorical_columns(self):
        """
        Strata are the columns. Moves through the columns and calculates the metrics (count, percent) of the variable
        which is a row.
        """
        if self.strata == 'SingleDistributionTable':
            for cat_ in self.index_categorical:
                col = cat_.split('__')[0]
                category = cat_.split('__')[1]
                self.tableone.loc[self.tableone['variable'] == cat_, self.strata] = self._categorical_var_dist(
                    frame=self.df, col=col, category=int(float(category)))
        else:
            for column_group_ in self.df[self.strata].unique():
                for cat_ in self.index_categorical:
                    col = cat_.split('__')[0]
                    category = cat_.split('__')[1]
                    self.tableone.loc[self.tableone['variable'] == cat_, column_group_] = self._categorical_var_dist(
                        frame=self.df[self.df[self.strata] == column_group_],
                        col=col,
                        category=int(float(category)))

    def _create_index(self, frame: pd.DataFrame,
                      col: str,
                      prefix: Optional[str] = None) -> list[str]:
        """
        Create the indexes that will be used as rows of table one based on the unique values of the current column.
        We place a prefix to differentiate it from other similar-named rows.
        Eg. gender becomes gender__0, gender__1
        :param frame: dataset
        :param col: column to map as indexes
        :param prefix: rename the column with a more suitable name.
        :return:
        """
        if prefix is None:
            prefix = col
        unique_vals = frame.dropna(subset=[col])[col].unique()
        if all([isinstance(val, str) for val in unique_vals]):
            unique_vals = self._map_index_str_to_int(prefix=prefix, unique_vals=unique_vals)
        elif any([isinstance(val, str) for val in unique_vals]):
            raise ValueError(f'The column {col} is of mixed data types, strings and numeric. Please set to one or the '
                             f'other')
        unique_vals.sort()
        return [f'{prefix}__{int(i)}' for i in unique_vals]

    def _map_index_str_to_int(self, prefix:str, unique_vals:list[str]) -> list[int]:
        """
        If the cell values are strings, we convert them to integers.
        :param prefix:
        :param unique_vals:
        :return:
        """
        self.index_mapper[prefix] = {int_name:name for name, int_name in enumerate(unique_vals)}
        self.df[prefix] = self.df[prefix].map(self.index_mapper[prefix])
        return list(self.index_mapper[prefix].values())

    def _re_map_indexes_to_strings(self):
        """
        re-map the variables that were all indexes to their original string names
        e.g.,
        If the column Race in df had unique values ['Caucasian', 'Black', 'Latino', 'Asian', 'Mixed', 'Other']
        rename the Race__numeric to their respective string names
        'Race__0': 'Caucasian',
         'Race__1': 'Black',
         'Race__2': 'Latino',
         'Race__3': 'Asian',
         'Race__4': 'Mixed',
         'Race__5': 'Other'}
            For each prefix in index_mapper, insert a header row showing the
            question name (i.e. the prefix) above the block of __0, __1, ... rows,
            then replace those coded names with their string labels.
        :return:
        """
        for prefix in self.index_mapper.keys():
            mapper = {f'{prefix}__{numeric}':string for string, numeric in self.index_mapper.get(prefix).items()}
            # find all rows to be renamed
            mask = self.tableone['variable'].str.startswith(f'{prefix}__')
            if not mask.any():
                continue

            # insert header row above the first matching index
            first_idx = mask.idxmax()
            header = {col: '' for col in self.tableone.columns}
            header['variable'] = prefix
            header_df = pd.DataFrame([header])

            top = self.tableone.iloc[:first_idx]
            bottom = self.tableone.iloc[first_idx:]
            self.tableone = pd.concat([top, header_df, bottom], ignore_index=True)
            # rename your coded rows
            self.tableone['variable'] = self.tableone.variable.replace(mapper)

    @staticmethod
    def _continuous_var_dist(frame: pd.DataFrame,
                             col: str,
                             decimal: Optional[int] = 2) -> str:
        """
        Compute a distribution summary for a continuous variable, returning:
          mean (SD); median [IQR]

        Parameters
        ----------
        frame : pd.DataFrame
            The DataFrame (or subset) from which to compute the summary.
        col : str
            The name of the continuous column.
        decimal : int, optional
            Number of decimal places to round to (default: 2).

        Returns
        -------
        str
            A formatted string of the form:
            "μ (σ); median [Q1–Q3]",
            where μ and σ are the mean and standard deviation,
            and Q1/Q3 are the 25th/75th percentiles.
        """
        # drop missing values
        series = frame[col].dropna()
        if series.empty:
            return ''

        mean = series.mean()
        sd   = series.std()
        q1   = series.quantile(0.25)
        med  = series.median()
        q3   = series.quantile(0.75)

        fmt = f"{{:.{decimal}f}}"
        return (
            f"{fmt.format(mean)} ({fmt.format(sd)}); "
            f"{fmt.format(med)} [{fmt.format(q1)}–{fmt.format(q3)}]"
        )

    @staticmethod
    def _categorical_var_dist(frame: pd.DataFrame,
                              col: str,
                              category: Optional[int] = None,
                              decimal: Optional[int] = 2,
                              ) -> Union[str, list]:
        """
        Count the number of occurrences in a column, giving he number of evens and the percentage. Used for category columns

        :param frame: dataframe from there to compute the count on the columns
        :param col: column to compute the calculation of the count
        :param category: if we want to count a specific category of the categories
        :param decimal: decimal point to show in the table
        :return:
        """
        if category is not None:
            count = frame.loc[frame[col] == category, col].shape[0]
            non_nan_count = frame.loc[~frame[col].isna()].shape[0]  # use the non-nan count
            if frame.shape[0] == 0:
                return f'0'
            else:
                if non_nan_count > 0:
                    cell = f'{count} ({np.round((count / non_nan_count) * 100, decimal)}%)'
                else:
                    cell = f'{count} (-%)'

            return cell
        else:
            # return the count ordered by the index
            count = frame[col].value_counts()
            count = count.sort_index()
            non_nan_count = frame.loc[~frame[col].isna()].shape[0]  # use the non-nan count

            if count.shape[0] == 1:
                # binary data so counting the ones
                if non_nan_count > 0:
                    cell = f'{count[1]} ({np.round((count[1] / non_nan_count) * 100, decimal)}%)'
                else:
                    cell = f'{count} (0%)'

                return cell
            else:
                if non_nan_count > 0:
                    cell = [f'{count_} (0%)' for count_ in count]
                else:
                    cell = [f'{count_} (0%)' for count_ in count]
                return cell

    def remove_reference_categories(self):
        """
        Remove the reference group or zero group from the categories
        :return:
        """
        rmv_idx = self.tableone[self.tableone['variable'].str.contains("__0")].index
        self.tableone.drop(index=rmv_idx, inplace=True)

    def get_table(self):
        return self.tableone

    @staticmethod
    def group_variables_table(df: pd.DataFrame) -> pd.DataFrame:
        """
        Group the variable into their responses e.g.,
        rows
        cir_0200__1,
        cir_0200__2,
        cir_0200__3 becomes:

        cir_0200
            1,
            2,
            3
        :param df:
        :return:
        """
        # Create an empty DataFrame to store the transformed data
        df_transformed = pd.DataFrame(columns=df.columns)

        # Track previous group to avoid duplicate headers
        prev_group = None

        for i, row in df.iterrows():
            # Extract the prefix if the variable matches <name>__<int>
            prefix_match = row['variable'].split('__')[0] if '__' in row['variable'] else None

            # Check if a header row is needed
            if prefix_match and prefix_match != prev_group:
                # Insert a header row for the prefix
                prefix_row = [prefix_match] + ['--'] * (len(df.columns)-1)
                header_row = pd.DataFrame([prefix_row], columns=df.columns)
                df_transformed = pd.concat([df_transformed, header_row], ignore_index=True)
                prev_group = prefix_match

            # Add the current row, removing suffix if it's part of a group
            if prefix_match:
                row['variable'] = row['variable'].replace(f"{prefix_match}__", "")

            df_transformed = pd.concat([df_transformed, pd.DataFrame([row])], ignore_index=True)

        return df_transformed

    @staticmethod
    def merge_levels_others(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        """
        List of columns to sum, excluding the first element
        :param df:
        :param columns:
        :return:
        """
        col_head_index = df.loc[df['variable'] == columns[0]].index[0]
        columns = columns[1:]

        def extract_and_sum_percentages(column_values):
            """
            Iterate over the column values, and it will sum the percentage count str(<int> (<float>%))
            :param column_values:
            :return:
            """
            # Initialize totals for counts outside and inside parentheses
            count_total = 0.0
            percent_total = 0.0

            for value in column_values:
                # Extract the numbers outside and inside the parentheses
                match = re.match(r"(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)%\)", str(value))
                if match:
                    count_value = float(match.group(1))
                    percent_value = float(match.group(2))

                    # Sum these extracted values
                    count_total += count_value
                    percent_total += percent_value

            # Format the result as a string similar to the original format
            return f"{count_total} ({percent_total:.2f}%)"

        df_subset = df.loc[df['variable'].isin(columns), :]
        rows_other = {}
        for col_ in df.columns:
            if col_ == 'variable':
                continue
            rows_other[col_] = extract_and_sum_percentages(df_subset[col_])

        final_row = pd.DataFrame([rows_other], columns=df_subset.columns)
        final_row['variable'] = ' Other'
        # Reconstruct the dataframe with 'Other' row inserted below the column header row
        df_updated = df.copy()
        df_updated = df_updated.drop(index=df_subset.index)
        df_updated = pd.concat([df_updated.iloc[:col_head_index + 1],
                                final_row,
                                df_updated.iloc[col_head_index + 1:]],
                               ignore_index=True)
        return df_updated







def stats_test_binary_symptoms(data: pd.DataFrame,
                               columns: List[str],
                               strata_col: str = 'NT1',
                               SHOW: Optional[bool] = False
                               ) -> pd.DataFrame:
    """
    Perform Chi-Square Test or Fisher's Exact Test for binary values (1 as positive response)
    comparing the distribution between two groups.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing binary symptom responses and the group column.
    columns : List[str]
        Columns to perform the statistical test on.
    strata_col : str, default 'NT1'
        Column name for the grouping variable.
    SHOW : bool, default False
        Print the 2x2 contingency table for each test.

    Returns
    -------
    pd.DataFrame
        DataFrame with counts, percentages, p-value, effect size (odds ratio), and test method used.
    """
    results = []

    # Define a helper lambda to compute count and percentage for a boolean Series.
    get_counts_rates = lambda cond: (cond.sum(), cond.mean() * 100)

    if not len(data[strata_col].unique()) == 2:
        raise ValueError(f'Strata column {strata_col} must have exactly 2 groups')

    group0, group1 = data[strata_col].unique()

    for col in columns:
        if col == strata_col:
            continue

        df = data[[col, strata_col]].dropna().copy()

        # Ensure binary values
        unique_vals = set(df[col].unique())
        if unique_vals != {0, 1}:
            continue

        # Count responses for each group
        grp0 = (df[df[strata_col] == group0][col] == 1)
        grp1 = (df[df[strata_col] == group1][col] == 1)

        group0_n, group0_rate = get_counts_rates(grp0)
        group1_n, group1_rate = get_counts_rates(grp1)
        total_n, total_rate = get_counts_rates(df[col] == 1)

        # Create 2x2 contingency table
        # Create contingency table
        a = group0_n
        b = df[df[strata_col] == group0].shape[0] - group0_n
        c = group1_n
        d = df[df[strata_col] == group1].shape[0] - group1_n
        table = [[a, b], [c, d]]

        # Display table if SHOW is True
        if SHOW:
            headers = [f"{col} {group0}", f"{col} {group1}"]
            row_labels = [f"{strata_col} {group0}", f"{strata_col} {group1}"]
            table_with_labels = [[row_labels[i]] + row for i, row in enumerate(table)]
            headers_with_labels = ["Group"] + headers
            print(tabulate(table_with_labels, headers=headers_with_labels, tablefmt="grid"))

        # Compute expected counts for Chi-Square condition
        chi2_stat, p_chi2, dof, expected = chi2_contingency(table)
        expected_min = expected.min()

        if expected_min < 5:
            # Use Fisher's Exact Test if any expected count is <5
            odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
            test_method = "Fisher's Exact Test"
        else:
            # Use Chi-Square Test and compute OR manually
            p_value = p_chi2
            try:
                odds_ratio = (a * d) / (b * c) if b * c != 0 else np.nan
            except ZeroDivisionError:
                odds_ratio = np.nan
            test_method = "Chi-Square Test"

        # Store results
        res = {
            'Variable': col,
            f'{strata_col} {group0} N, (%)': f'{round(group0_n, 1)} ({round(group0_rate, 1)})',
            f'{strata_col} {group1} N, (%)': f'{round(group1_n, 1)} ({round(group1_rate, 1)})',
            'Total N, (%)': f'{round(total_n, 1)} ({round(total_rate, 1)})',
            'p-value': p_value,
            'p-value formatted': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
            'Effect Size (Odds Ratio)': round(odds_ratio, 3) if odds_ratio is not None else "N/A",
            'Stat Method': test_method
        }
        results.append(res)

    return pd.DataFrame(results)

def stats_test_continuous(data: pd.DataFrame,
                          columns: List[str],
                          strata_col: str = 'NT1',
                          SHOW: Optional[bool] = False
                          ) -> pd.DataFrame:
    """
    Perform statistical tests between continuous distributions across two groups.
    - First tests for normality using Shapiro-Wilk test.
    - If both distributions are normal, use an independent t-test.
    - Otherwise, use the Mann-Whitney U test.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the continuous symptom values and the grouping column.
    columns : List[str]
        Columns to perform the statistical test.
    strata_col : str, default 'NT1'
        Column name for the grouping.
    SHOW : bool, default False
        Print distribution summary statistics for each test.

    Returns
    -------
    pd.DataFrame
        DataFrame with mean, standard deviation, p-value, and test used.
    """
    results = []

    unique_groups = data[strata_col].dropna().unique()
    if len(unique_groups) != 2:
        raise ValueError(f'Strata column {strata_col} must have exactly 2 groups.')

    group0, group1 = unique_groups

    for col in columns:
        print(col)
        if col == strata_col:
            continue

        df = data[[col, strata_col]].dropna()
        group0_vals = df[df[strata_col] == group0][col]
        group1_vals = df[df[strata_col] == group1][col]

        if len(group0_vals) < 10 or len(group1_vals) < 10:
            continue
        # Normality test
        normal0 = shapiro(group0_vals).pvalue > 0.05
        normal1 = shapiro(group1_vals).pvalue > 0.05

        if normal0 and normal1:
            stat_test = 'Independent t-test'
            stat, p_value = ttest_ind(group0_vals, group1_vals, equal_var=False)
        else:
            stat_test = 'Mann-Whitney U test'
            stat, p_value = mannwhitneyu(group0_vals, group1_vals, alternative='two-sided')

        # Descriptive stats
        mean0, std0 = group0_vals.mean(), group0_vals.std()
        mean1, std1 = group1_vals.mean(), group1_vals.std()

        res = {
            'Variable': col,
            f'{strata_col} {group0} Mean (SD)': f'{mean0:.2f} ({std0:.2f})',
            f'{strata_col} {group1} Mean (SD)': f'{mean1:.2f} ({std1:.2f})',
            f'n {strata_col} {group0}': len(group0_vals),
            f'n {strata_col} {group1}': len(group1_vals),
            'p-value': p_value,
            'p-value formatted': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
            'Stat Method': stat_test
        }

        if SHOW:
            print(f"{col} - {stat_test}: p = {p_value:.4f}")

        results.append(res)

    return pd.DataFrame(results)




def stats_test_binary_symptoms_multigroup(data: pd.DataFrame,
                                          columns: List[str],
                                          strata_col: str = 'OSA_Severity',
                                          SHOW: Optional[bool] = False) -> pd.DataFrame:
    """
    Performs Chi-Square test for binary variables across >2 groups.
    Adds Cramér’s V as the effect size.

    Effect Size:
    ------------
    Cramér’s V is used to quantify the association between two nominal variables:
        V = sqrt(χ² / (n * (k - 1)))
    where:
        χ² = chi-square statistic
        n = total sample size
        k = min(number of rows, number of columns)
    """
    results = []

    for col in columns:
        if col == strata_col or col not in data.columns:
            continue

        df = data[[col, strata_col]].dropna()

        unique_vals = set(df[col].unique())
        if not unique_vals.issubset({0, 1}):
            continue

        contingency = pd.crosstab(df[col], df[strata_col])
        chi2_stat, p_value, dof, expected = chi2_contingency(contingency)

        # Compute Cramér’s V
        n = contingency.to_numpy().sum()
        k = min(contingency.shape)
        cramers_v = np.sqrt(chi2_stat / (n * (k - 1))) if k > 1 else np.nan

        if SHOW:
            print(f"\n{col} Contingency Table:\n", contingency)

        summary = {}
        for group in df[strata_col].unique():
            total = (df[strata_col] == group).sum()
            count = df[(df[strata_col] == group) & (df[col] == 1)].shape[0]
            pct = round(100 * count / total, 1) if total > 0 else 0.0
            summary[f'{group} N, (%)'] = f"{count} ({pct}%)"

        res = {
            'Variable': col,
            **summary,
            'p-value': p_value,
            'p-value formatted': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
            'Effect Size (Cramér\'s V)': round(cramers_v, 3),
            'Stat Method': 'Chi-Square'
        }
        results.append(res)

    return pd.DataFrame(results)


def stats_test_continuous_multigroup(data: pd.DataFrame,
                                     columns: List[str],
                                     strata_col: str = 'OSA_Severity',
                                     SHOW: Optional[bool] = False) -> pd.DataFrame:
    """
    Performs ANOVA or Kruskal-Wallis test for continuous variables across >2 groups.
    Adds:
    - Eta squared (η²) for ANOVA using sum of squares
    - Epsilon squared (ε²) for Kruskal-Wallis

    Effect Sizes:
    -------------
    - Eta squared (η²) for ANOVA:
        η² = SS_between / SS_total
    - Epsilon squared (ε²) for Kruskal-Wallis:
        ε² = (H - k + 1) / (n - k)
    """
    results = []
    groups = data[strata_col].dropna().unique()

    for col in columns:
        if col == strata_col or col not in data.columns:
            continue

        df = data[[col, strata_col]].dropna()
        group_vals = [df[df[strata_col] == g][col] for g in groups]

        if any(len(g) < 3 for g in group_vals):
            continue

        normal = all(shapiro(g).pvalue > 0.05 for g in group_vals)
        total_n = sum(len(g) for g in group_vals)
        k = len(group_vals)

        if normal:
            stat, p_value = f_oneway(*group_vals)
            eta_squared = _compute_anova_eta_squared(df, col, strata_col)
            effect_size = round(eta_squared, 3)
            stat_method = "ANOVA"
            effect_label = "Eta Squared (η²)"
        else:
            stat, p_value = kruskal(*group_vals)
            epsilon_squared = (stat - k + 1) / (total_n - k) if (total_n - k) > 0 else np.nan
            effect_size = round(epsilon_squared, 3)
            stat_method = "Kruskal-Wallis"
            effect_label = "Epsilon Squared (ε²)"

        summary = {
            'Variable': col,
            'Stat Method': stat_method,
            'p-value': p_value,
            'p-value formatted': f"{p_value:.4f}" if p_value >= 0.0001 else "<0.0001",
            f'Effect Size ({effect_label})': effect_size
        }

        for g in groups:
            values = df[df[strata_col] == g][col]
            summary[f'{g} Mean (SD)'] = f"{values.mean():.2f} ({values.std():.2f})"
            summary[f'n {g}'] = len(values)

        if SHOW:
            print(f"{col} ({stat_method}): p = {p_value:.4f}, Effect Size = {effect_size}")

        results.append(summary)

    return pd.DataFrame(results)

def _compute_anova_eta_squared(df: pd.DataFrame, col: str, strata_col: str) -> float:
    """
    Compute Eta Squared (η²) for ANOVA manually from sums of squares.
    """
    grand_mean = df[col].mean()
    groups = df[strata_col].unique()
    ss_between = 0
    ss_within = 0

    for g in groups:
        group_data = df[df[strata_col] == g][col]
        n_g = len(group_data)
        mean_g = group_data.mean()
        ss_between += n_g * ((mean_g - grand_mean) ** 2)
        ss_within += ((group_data - mean_g) ** 2).sum()

    ss_total = ss_between + ss_within
    eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
    return eta_squared

