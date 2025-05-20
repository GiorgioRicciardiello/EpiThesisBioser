"""
Functions to implement within subject comparison in clinical studies

Author: giocrm@stanford.edu
Date: January 2024
"""
import pathlib
from config.config import config
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.stats.contingency_tables import mcnemar, cochrans_q
from scipy.stats import chi2
from typing import Optional, Union, Dict, Tuple
from statsmodels.stats.power import TTestIndPower, NormalIndPower
from scipy.stats import binomtest, permutation_test, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import re
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon, ttest_rel, shapiro


def _beautify_variable_name(name: str) -> str:
    """
    Beautify a variable name by replacing underscores with spaces and capitalizing words.

    Parameters:
    -----------
    name : str
        Original variable name with underscores.

    Returns:
    --------
    str
        Beautified, human-readable variable name.
    """
    name = name.lower().replace("_", " ")
    name = re.sub(r'\b(pre|post)sleep\b', lambda m: m.group(0).capitalize(), name)
    return name.title()

def _format_p_value(p,
                    decimal_places:Optional[int]=4,
                    sci_decimal_places:Optional[int]=2):
    """
    Format p-values to display in scientific notation if very small,
    with customizable decimal places for both formats.

    Args:
        p (float): The p-value to format.
        decimal_places (int): Number of decimal places for regular rounding.
        sci_decimal_places (int): Number of decimal places for scientific notation.

    Returns:
        str: Formatted p-value as a string.

    Example Usage:
    # Apply formatting with dynamic decimal places
    df_results['p_value_formatted'] = df_results.p_value.apply(lambda p: format_p_value(p, decimal_places=5, sci_decimal_places=3))
        p_value	        p_value_formatted
        2.985507e-07	    2.986e-07
        3.904867e-07	    3.905e-07
        1.077162e-06	    1.077e-06
        2.367503e-05	    0.00002
        4.527726e-05	    0.00005

    """
    if p < 10 ** (-decimal_places):
        return f"{p:.{sci_decimal_places}e}"  # Scientific notation
    return round(p, decimal_places)  # Regular rounding

def apply_within_pairs_statistical_tests(df: pd.DataFrame,
                                         dtypes: dict[str, str],
                                         pair_id: str = 'id_subject',
                                         ordinal_test: Optional[str] = 'wilcoxon') -> pd.DataFrame:
    """
    Perform within-subject comparison statistics with rigorous measures, allowing dynamic selection of ordinal tests.
    :param df: DataFrame containing the data.
    :param dtypes: Dictionary mapping column names to their data types.
    :param pair_id: Column representing pairs for within-subject comparisons.
    :param ordinal_test: Test to use for ordinal data ('wilcoxon', 'sign_test', 'friedman', or 'permutation').
    :return: Dictionary with statistical results for each variable.
    """

    def compute_wilcoxon(diff_:np.ndarray) -> dict:
        """
        Compute the wilcoxon test and compute the effect size
        :param diff_:
        :return:
        """
        stats_wilcoxon = stats.wilcoxon(x=diff_,
                                        # y=values2,
                                        correction=True,
                                        zero_method='wilcox',
                                        alternative='two-sided',
                                        method='approx')
        stat = stats_wilcoxon.statistic
        p_val = stats_wilcoxon.pvalue
        n = len(diff_[diff_ != 0])  # Exclude zero differences (ties)
        # Calculate expected value and variance under H0
        E_T_plus = n * (n + 1) / 4
        Var_T_plus = n * (n + 1) * (2 * n + 1) / 24
        # Calculate Z-score
        Z = (stat - E_T_plus) / np.sqrt(Var_T_plus)
        # Calculate effect size r
        effect_size = Z / np.sqrt(n)
        method = 'Wilcoxon Signed-Rank Test'
        return {'effect_size':effect_size,
                'statistic':stat,
                'p_value':p_val,
                'method':method}

    def compute_sign_test(diff_: np.ndarray) -> dict:
        # Calculate the number of positive differences
        pos_differences = sum(d > 0 for d in diff_)

        # Perform binomial test (assumes 50% probability under null hypothesis)
        signtest = binomtest(pos_differences,
                             n=len(diff_),
                             p=0.5,
                             alternative='two-sided')
        # Effect size (Cohen's g)
        p_observed = pos_differences / n
        effect_size = p_observed - 0.5
        method = 'Sign Test'
        return {'effect_size': effect_size,
                'statistic': signtest.statistic,
                'p_value': signtest.pvalue,
                'method': method}

    def compute_friedman(data: np.ndarray) -> dict:
        """
        Compute Friedman test and calculate the effect size using Kendall's W.
        :param data: A 2D numpy array where each row corresponds to a subject and columns represent conditions.
        :return: Dictionary with effect size, test statistic, p-value, and method.
        """
        # Perform the Friedman test
        stat, p_val = friedmanchisquare(*data.T)

        # Calculate number of subjects and conditions
        n, k = data.shape

        # Compute Kendall's W (effect size)
        effect_size = stat / (n * (k - 1))
        method = 'Friedman Test'

        return {'effect_size': effect_size,
                'statistic': stat,
                'p_value': p_val,
                'method': method}

    def compute_permutation_paired(group1: np.ndarray,
                                   group2: np.ndarray,
                                   n_permutations=10000) -> dict:
        """
        Compute permutation paired test and calculate effect size (Cohen's d).
        :param group1: First group of paired measurements.
        :param group2: Second group of paired measurements.
        :param n_permutations: Number of permutations for the test.
        :return: Dictionary with effect size, test statistic, p-value, and method.
        """
        # Calculate differences between paired measurements
        differences = group1 - group2

        # Permutation test
        result = permutation_test(
            data=(group1, group2),
            statistic=lambda x, y: np.mean(x - y),  # Mean difference
            permutation_type='samples',
            alternative='two-sided',
            n_resamples=n_permutations
        )

        # Calculate Cohen's d for paired differences
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        effect_size = mean_diff / std_diff
        method = 'Permutation Paired Test'

        return {'effect_size': effect_size,
                'statistic': result.statistic,
                'p_value': result.pvalue,
                'method': method}

    def apply_p_value_correction(df: pd.DataFrame) -> pd.DataFrame:
        """Correct the p value with the multiple testing rule"""
        df['p_value_adjusted'] = multipletests(df['p_value'], method='fdr_bh')[1]
        df['p_value_formatted'] = df.p_value.apply(_format_p_value)
        df['p_value_adjusted_formatted'] = df.p_value_adjusted.apply(_format_p_value)
        return df

    # Map test names to their corresponding functions
    ordinal_tests = {
        'wilcoxon': compute_wilcoxon,
        'sign_test': compute_sign_test,
        'friedman': compute_friedman,
        'permutation': compute_permutation_paired
    }

    # Check if the selected ordinal test is valid
    if ordinal_test not in ordinal_tests:
        raise ValueError(f"Invalid ordinal_test '{ordinal_test}'. Choose from {list(ordinal_tests.keys())}.")

    results = {}
    for col, dtype in dtypes.items():
        # col = [*dtypes.keys()][0]
        # dtype = dtypes[col]
        if not col in df.columns:
            continue
        print(f'Hypothesis testing: {col} - dtype: {dtype}\n')
        # Extract the subject pairs for the given column
        paired_data = df[[col, pair_id]].copy().dropna()
        paired_data = paired_data.groupby(pair_id).agg(list)

        # Ensure we have exactly two observations per subject
        paired_data = paired_data[paired_data[col].apply(len) == 2]
        # Convert non-continuous values to integers
        if dtype in ['binary', 'ordinal', 'categorical']:
            paired_data[col] = paired_data[col].apply(lambda x: [int(val) for val in x])

        # Extract paired values
        values1 = [pair[0] for pair in paired_data[col]]   # baseline
        values2 = [pair[1] for pair in paired_data[col]]   # outcome

        n = len(values1)

        if n == 0:
            results[col] = {
                'sample_size': n,
                'statistic': None,
                'p_value': None,
                'effect_size': None,
                'method': 'No valid pairs',
                'dtype': dtype,
            }
            continue

        # diff = np.round(np.array(values1) - np.array(values2), 3)
        diff = np.array(values2) - np.array(values1)
        # symmetry: np.mean(diff), np.median(diff) -> must be similar
        if dtype == 'continuous':
            stat, normality_p = stats.shapiro(diff)
            print(f'\tShapiro test: {col}\t {stat}, {normality_p}\n')
            if normality_p >= 0.05:  # Data is normally distributed
                t_stat, p_val = stats.ttest_rel(values1, values2)
                method = 'Paired t-test'
                # effect_size = t_stat
                # Compute Cohen's d for paired samples
                mean_diff = np.mean(diff)
                sd_diff = np.std(diff, ddof=1)
                effect_size = mean_diff / sd_diff  # Cohen's d


            else:  # Data is not normally distributed
                res = compute_wilcoxon(diff_=diff)
                effect_size = res.get('effect_size')
                t_stat = res.get('statistic')
                p_val = res.get('p_value')
                method = res.get('method')


        elif dtype == 'binary':
            table = pd.crosstab(np.array(values1), np.array(values2))
            b = table.iloc[0, 1]  # Discordant pair (Yes, No)
            c = table.iloc[1, 0]  # Discordant pair (No, Yes)

            if (b + c) < 25:
                result = mcnemar(table, exact=True)
            else:
                result = mcnemar(table, exact=False)
            p_val = result.pvalue
            t_stat = None
            method = 'McNemar Test'
            effect_size = b / (b + c) if (b + c) > 0 else None  # Proportion of discordance

        # elif dtype == 'ordinal':
        #     # Wilcoxon signed-rank test for ordinal data, before-after or time_0 - time_1
        #     res = compute_wilcoxon(diff_=diff)
        #     effect_size = res.get('effect_size')
        #     t_stat = res.get('statistic')
        #     p_val = res.get('p_value')
        #     method = res.get('method')

        elif dtype == 'ordinal':
            if ordinal_test == 'friedman':
                reshaped_data = np.column_stack((values1, values2))
                res = ordinal_tests[ordinal_test](reshaped_data)
            else:
                res = ordinal_tests[ordinal_test](diff)
            # results[col] = {**res, 'sample_size': n, 'dtype': dtype}
            p_val = res.get('p_value')
            t_stat = res.get('statistic')
            effect_size = res.get('effect_size')
            method = res.get('method')

        elif dtype == 'categorical':
            table = pd.crosstab(np.array(values1), np.array(values2))

            if table.shape == (2, 2):  # Binary categorical data
                result = mcnemar(table, exact=True)
                p_val = result.pvalue
                t_stat = result.statistic
                b = table.iloc[0, 1]  # Off-diagonal counts
                c = table.iloc[1, 0]
                odds_ratio = b / c if c != 0 else np.inf
                effect_size = np.log(odds_ratio)  # Log odds ratio as effect size
                method = 'McNemar Test'

            else:
                # Stuart-Maxwell Test for multi-categorical data
                marginals = table.sum(axis=1) - table.sum(axis=0)
                observed = marginals ** 2
                expected = table.sum().sum() / (2 * table.shape[0])  # Expected under null
                test_stat = (observed / expected).sum()

                # Degrees of freedom
                dof = table.shape[0] - 1
                p_val = chi2.sf(test_stat, dof)

                # Effect size: Cramér's V
                total = table.values.sum()
                effect_size = np.sqrt(test_stat / (total * (table.shape[0] - 1)))

                t_stat = np.sqrt(test_stat)  # Pseudo t-statistic
                method = 'Stuart-Maxwell Test'

        else:
            p_val = None
            t_stat = None
            method = 'Unknown'
            effect_size = None

        results[col] = {
            'sample_size': n,
            'p_value': p_val,
            'effect_size': effect_size,
            'statistic': t_stat,
            'method': method,
            'dtype': dtype,
        }

    def interpret_effect(x) -> str:
        """
        This depends on how we are computing the differences of the two groups
        :param x:
        :return:
        """
        if x <0: return 'Baseline > Outcome'
        elif x == 0: return 'No Effect'
        else: return 'Baseline < Outcome'

    df_results = pd.DataFrame(results).T
    df_results.reset_index(inplace=True, drop=False, names='variable')
    df_results['effect_dir'] = df_results['effect_size'].apply(interpret_effect)
    df_results['effect_size'] = df_results['effect_size'].round(3)
    df_results = apply_p_value_correction(df=df_results)
    return df_results


def reshape_wide_to_long(df_wide: pd.DataFrame,
                         paired_tests_dict: dict,
                         subject_id_col: str = 'id_subject') -> pd.DataFrame:
    """
    Convert wide format (1 row per subject) to long format (2 rows per subject per test).
    Returns a DataFrame with columns: id_subject, variable, value, time.
    """
    long_rows = []

    for subj_id, row in df_wide.iterrows():
        for test_name, test_info in paired_tests_dict.items():
            pre_col, post_col = test_info['variable']

            if pre_col in df_wide.columns and post_col in df_wide.columns:
                long_rows.append({
                    'id_subject': subj_id,
                    'variable': pre_col,
                    pre_col: row[pre_col],
                    'time': 0
                })
                long_rows.append({
                    'id_subject': subj_id,
                    'variable': pre_col,
                    pre_col: row[post_col],
                    'time': 1
                })

    df_long = pd.DataFrame(long_rows)
    return df_long

def apply_within_pairs_from_dict(
    df: pd.DataFrame,
    paired_tests_dict: dict,
    pair_id: str = 'id_subject',
    group_col: Optional[str] = None,
    ordinal_test: str = 'wilcoxon'
) -> pd.DataFrame:
    """
    Perform within-subject pre/post statistical tests for variables defined in paired_tests_dict,
    optionally stratified by a grouping column (e.g., OSA severity).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame in wide format. Each row represents a subject with pre/post columns.
    paired_tests_dict : dict
        Dictionary defining pre/post variable pairs and test configurations.
        Each key is a test label, with value:
            {
                'variable': (pre_col, post_col),
                'type_test': 'Paired t-test' | 'Wilcoxon signed-rank test' (optional),
                'dtype': 'ordinal' | 'continuous' (default: 'ordinal'),
                'interpretation': str
            }
    pair_id : str
        Subject identifier column (not directly used in this function, but kept for clarity).
    group_col : str, optional
        Column to stratify by (e.g., 'osa_severity'). If None, test is performed on full sample.
    ordinal_test : str
        Default method for ordinal data if no override is provided. Default: 'wilcoxon'.

    Returns
    -------
    pd.DataFrame
        Results table with columns:
        ['Group', 'Test Name', 'sample_size', 'statistic', 'p_value', 'p_value_formatted',
         'effect_size', 'p_value_adjusted', 'p_value_adjusted_formatted',
         'method', 'direction', 'interpretation']
    """

    def _format_p(p, dec_places=4, sci_places=2):
        """Format p-values with scientific notation if small."""
        if p < 10**(-dec_places):
            return f"{p:.{sci_places}e}"
        return round(p, dec_places)

    results = []

    # Prepare group splits: either a single group (None) or groupby object
    grouped_dfs = [(None, df)] if group_col is None else df.groupby(group_col)

    for group_name, group_df in grouped_dfs:
        for test_name, info in paired_tests_dict.items():
            pre, post = info['variable']
            dtype = info.get('dtype', 'ordinal')
            override = info.get('type_test', None)
            interp = info.get('interpretation', '')

            # Skip if columns are missing
            if pre not in group_df.columns or post not in group_df.columns:
                continue

            subset = group_df[[pre, post]].dropna()
            x = subset[pre].astype(float).values
            y = subset[post].astype(float).values
            diff = y - x
            n = len(diff)

            if n == 0:
                results.append({
                    'Group': group_name,
                    'Test Name': test_name,
                    'sample_size': 0,
                    'statistic': np.nan,
                    'p_value': np.nan,
                    'p_value_formatted': None,
                    'effect_size': np.nan,
                    'method': 'No valid pairs',
                    'direction': 'No data',
                    'interpretation': interp
                })
                continue

            # Determine appropriate test
            if override == 'Paired t-test':
                method = 'Paired t-test'
            elif override == 'Wilcoxon signed-rank test':
                method = 'Wilcoxon signed-rank test'
            else:
                if dtype == 'ordinal':
                    method = 'Wilcoxon signed-rank test'
                else:
                    # Use Shapiro test to check normality for continuous data
                    stat_sh, p_sh = shapiro(diff)
                    method = 'Paired t-test' if p_sh > 0.05 else 'Wilcoxon signed-rank test'

            # Compute the test
            if method == 'Paired t-test':
                stat, pval = ttest_rel(x, y)
                d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else 0
            else:
                stat, pval = wilcoxon(x, y)
                # Approximate z-score for Wilcoxon effect size (r)
                z = (stat - n*(n+1)/4) / np.sqrt(n*(n+1)*(2*n+1)/24)
                d = z / np.sqrt(n)

            direction = ('Post > Pre' if d > 0 else 'Post < Pre') if d != 0 else 'No Change'

            results.append({
                'Group': group_name,
                'interpretation': interp,
                'Test Name': test_name,
                'effect_size': round(d, 3),
                'direction': direction,
                'p_value_formatted': _format_p(pval),
                'p_value_adjusted_formatted': None,
                'sample_size': n,
                'statistic': round(stat, 3),
                'p_value': pval,
                'p_value_adjusted': None,
                'method': method,
            })

    # Convert to DataFrame
    res_df = pd.DataFrame(results)

    # Apply FDR correction within each group
    if not res_df.empty:
        for group in res_df['Group'].unique():
            mask = res_df['Group'] == group
            pvals = res_df.loc[mask, 'p_value']
            if pvals.notna().any():
                adj = multipletests(pvals, method='fdr_bh')[1]
                res_df.loc[mask, 'p_value_adjusted'] = adj
                res_df.loc[mask, 'p_value_adjusted_formatted'] = res_df.loc[mask, 'p_value_adjusted'].apply(_format_p)

    return res_df

def calculate_wilcoxon_sample_size(effect_size, alpha=0.05, power=0.8):
    """
    Calculate the required sample size for the Wilcoxon signed-rank test.

    Parameters:
    - effect_size (float): Expected effect size (r).
    - alpha (float): Significance level (default is 0.05).
    - power (float): Statistical power (default is 0.8).

    Returns:
    - sample_size (int): Required number of pairs.
    """
    # Initialize power analysis
    analysis = NormalIndPower()

    # Calculate the z-scores for alpha (two-sided) and power
    z_alpha = np.abs(np.percentile(np.random.normal(size=100000), alpha / 2 * 100))
    z_beta = np.abs(np.percentile(np.random.normal(size=100000), alpha * 100))

    # Compute the required sample size (approximation based on normality assumption)
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')

    return int(np.ceil(sample_size))


# $$ Visualizations


def qqplot_pre_post_pairs_combined(data: pd.DataFrame,
                                    variable_pairs: dict,
                                    figsize: tuple = (12, 10),
                                    ncols: int = 2,
                                    save_dir: pathlib.Path = None) -> None:
    """
    Plot all Q-Q plots in a single figure for each variable pair with beautified titles.

    Parameters:
    -----------
    data : pd.DataFrame
        The dataset containing pre and post variables.
    variable_pairs : dict
        Dictionary mapping test names to tuples of pre/post column names.
    figsize : tuple
        Size of the full figure.
    ncols : int
        Number of columns in the subplot grid.
    save_dir : pathlib.Path
        Optional directory to save the plot.
    """
    valid_pairs = []
    for test_name, test_info in variable_pairs.items():
        pre_col, post_col = test_info['variable']
        if pre_col in data.columns and post_col in data.columns:
            df_valid = data[[pre_col, post_col]].dropna()
            diff = df_valid[pre_col] - df_valid[post_col]
            if len(diff) >= 10:
                valid_pairs.append((test_name, diff, pre_col, post_col))

    n_plots = len(valid_pairs)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i, (test_name, diff, pre_col, post_col) in enumerate(valid_pairs):
        stats.probplot(diff, dist="norm", plot=axes[i])
        pre_label = _beautify_variable_name(pre_col)
        post_label = _beautify_variable_name(post_col)
        axes[i].set_title(f"{test_name}\n({pre_label} - {post_label})", fontsize=10)
        axes[i].grid(True)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])  # remove unused subplots

    fig.tight_layout()
    if save_dir:
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir, dpi=300)
    plt.show()



def fit_osa_severity_mixedlm(df,
                             pre_col: str,
                             post_col: str,
                             subject_col: str = 'id_subject',
                             severity_col: str = 'osa_four',
                             output_path: pathlib.Path = None) :
    """
    Fit a linear mixed‐effects model with a time × OSA‐severity interaction.

    Model (in long format):
        score_{i,j} = β₀
                      + β₁·I[time_{i,j} = post]
                      + β₂·I[osa_four_j = Moderate]
                      + β₃·I[osa_four_j = Severe]
                      + β₄·I[time_{i,j} = post]·I[osa_four_j = Moderate]
                      + β₅·I[time_{i,j} = post]·I[osa_four_j = Severe]
                      + u_{0,i} + u_{1,i}·I[time_{i,j} = post]
                      + ε_{i,j}

    where
      • i indexes subjects, j indexes observations (pre vs. post)
      • u_{0,i} ~ N(0, σ²_u0) is each subject’s random intercept
      • u_{1,i} ~ N(0, σ²_u1) is each subject’s random slope on time
      • ε_{i,j} ~ N(0, σ²_e) is the residual error

    Returns
    -------
    mdf : MixedLMResults
        The fitted model; inspect `mdf.summary()` for estimates of β₀…β₅, σ²_u0, σ²_u1, σ²_e.
    """

    def _export_mixedlm_epistats(mdf,
                                 filename:pathlib.Path = None,) -> pd.DataFrame:
        """
        Extract and transform fixed‐effects from a MixedLMResults to epidemiological metrics.

        For each fixed‐effect β_k:
           • OR_k      = exp(β_k)
           • 95% CI_k  = exp(β_k ± 1.96·SE_k)
           • p‐value_k = p-value from mdf.pvalues
        Also appends the model log‐likelihood.

        Parameters
        ----------
        mdf : MixedLMResults
            A fitted statsmodels mixed‐effects model.
        filename : str
            CSV path to write the table with columns:
              [OR, CI_lower, CI_upper, SE, p_value, log_likelihood]
        """
        # 1) Base estimates, SEs, p-values
        params = mdf.params
        se = mdf.bse
        pvals = mdf.pvalues

        # 2) 95% confidence intervals on the log scale
        ci = mdf.conf_int()
        ci_low = ci.iloc[:, 0]
        ci_high = ci.iloc[:, 1]

        # 3) Exponentiate to get OR and CI on original scale
        or_vals = np.exp(params)
        ci_low_or = np.exp(ci_low)
        ci_high_or = np.exp(ci_high)

        # 4) Build the epidemiology table
        df_model = pd.DataFrame({
            'OR': or_vals,
            'CI_lower': ci_low_or,
            'CI_upper': ci_high_or,
            'SE': se,
            'p_value': pvals
        })

        # 5) Append log-likelihood as its own row
        llf = mdf.llf
        ll_df = pd.DataFrame({
            'OR': [np.nan],
            'CI_lower': [np.nan],
            'CI_upper': [np.nan],
            'SE': [np.nan],
            'p_value': [np.nan],
            'log_likelihood': [llf]
        }, index=['LogLikelihood'])

        ll_df['p_value'] = ll_df['p_value'].apply(_format_p_value)
        ll_df['CI'] = ll_df.apply(lambda row: (row['CI_lower'].round(4) + row['CI_upper'].round(4)), axis=1)
        ll_df['OR'] = ll_df['OR'].round(2)
        ll_df['SE'] = ll_df['SE'].round(3)

        # 6) Concatenate (will introduce the log_likelihood column)
        df_result = pd.concat([df_model, ll_df], axis=0)

        # 7) Write to CSV
        if filename:
            df_result.to_csv(filename, index=True)
        return df_result

    # 1) Select only the columns we need and drop missing pairs
    df_pair = df[[subject_col, severity_col, pre_col, post_col]].dropna()

    # 2) Melt to long format
    long = df_pair.melt(id_vars=[subject_col, severity_col],
                        value_vars=[pre_col, post_col],
                        var_name='time',
                        value_name='score')

    # 3) Recode time to a simple factor
    long['time'] = long['time'].map({pre_col: 'pre', post_col: 'post'})
    long['time'] = long['time'].astype('category')
    long[severity_col] = long[severity_col].astype('category')

    # 4) Fit mixed‐effects model with random intercept and slope for time
    md = smf.mixedlm("score ~ time * osa_four",
                     long,
                     groups=long[subject_col],
                     re_formula="~time")
    mdf = md.fit(method='lbfgs')  # you can switch optimizer if needed

    print(mdf.summary())

    return mdf, _export_mixedlm_epistats(mdf, output_path)

