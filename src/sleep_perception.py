"""
psg_sleep_perception_analysis.py

Author: Giorgio Ricciardiello
Date: 2025-05-13

Description:
    This script supports a doctoral research project analyzing the disconnect between
    subjective perception and objective severity in Obstructive Sleep Apnea (OSA).
    It evaluates how well the Apnea–Hypopnea Index (AHI) reflects patient-reported
    symptoms and investigates alternative approaches for stratifying OSA burden.

    The script performs the following:
    - Computes subjective change scores (post–pre) for sleepiness, tiredness, and alertness.
    - Runs paired Wilcoxon signed-rank tests and McNemar’s tests to assess perception changes.
    - Conducts Kruskal–Wallis and post hoc Mann–Whitney U tests stratified by OSA severity.
    - Calculates effect sizes (r, ε², Cliff's delta, Cramer's V) for interpretation.
    - Evaluates whether worse AHI objectively correlates with worse subjective perception.
    - Generates Q–Q plots and summary tables of respiratory events by post-sleep symptoms.

Usage:
    python psg_sleep_perception_analysis.py --config path/to/config.yaml

Inputs:
    - Cleaned dataset with pre- and post-sleep questionnaire responses
    - OSA severity classification (numeric or categorical)
    - Respiratory event metrics from PSG (e.g., OA, CA, MA, HYP, RERA)

Outputs:
    - CSV files with test results (paired, stratified, post hoc)
    - Diagnostic plots (Q–Q) and group comparison tables
    - Summary tables linking respiratory metrics to perceived symptoms

Dependencies:
    pandas, numpy, scipy, statsmodels, pyyaml, matplotlib, tqdm, tabulate
"""
import pathlib

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import shapiro, f_oneway, kruskal, chi2, chi2_contingency
from config.config import config,encoding
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from tabulate import tabulate
from library.TableOne.table_one import MakeTableOne
import matplotlib.pyplot as plt
import pathlib
from typing import List


# %%  First we will explore subjective change across objective severity (OSA) → Do patients with worse AHI feel worse?
def compute_deltas(df: pd.DataFrame,
                   columns_compare:Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute difference scores (post - pre) for ordinal sleep perception variables.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with pre- and post-sleep columns.
    columns_compare: Dict[str, List[str]], key is the new columns to save in the frame and value is a list of columns to
            compare. They can not be more than 2 columns, with time order as first and second.
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'delta_' columns for each ordinal item.
    """
    for new_col, cols in columns_compare.items():
        assert len(cols) == 2, "You can not compare more than 2 columns"
        df[new_col] = df[cols[1]] - df[cols[0]]

    return df


def _compute_wilcoxon(diff_: np.ndarray) -> dict:
    """
    Compute the Wilcoxon signed-rank test and effect size r.

    Parameters
    ----------
    diff_ : np.ndarray
        Array of paired differences (post - pre), excluding NaNs.

    Returns
    -------
    dict
        Dictionary containing effect size, test statistic, p-value, and method.
    """
    # Perform Wilcoxon signed-rank test
    stats_wilcoxon = stats.wilcoxon(
        x=diff_,
        correction=True,
        zero_method='wilcox',
        alternative='two-sided',
        method='approx'
    )
    stat = stats_wilcoxon.statistic
    p_val = stats_wilcoxon.pvalue

    # Exclude zero differences (ties) for effect size
    n = np.count_nonzero(diff_)

    # Expected value and variance under H0
    E_T_plus = n * (n + 1) / 4
    Var_T_plus = n * (n + 1) * (2 * n + 1) / 24

    # Compute Z-score
    Z = (stat - E_T_plus) / np.sqrt(Var_T_plus)

    # Effect size r
    effect_size = Z / np.sqrt(n) if n > 0 else np.nan

    return {
        'method': 'Wilcoxon Signed-Rank Test',
        'statistic': stat.round(3),
        'p_value': p_val,
        'effect_size_r': effect_size.round(3)
    }


def _compute_odds_ratio(contingency: pd.DataFrame) -> float:
    """
    Compute odds ratio for a 2x2 contingency table of discordant pairs.

    Parameters
    ----------
    contingency : pd.DataFrame
        2x2 table with rows=pre, cols=post.

    Returns
    -------
    float
        Odds ratio (b/c, where b = pre=0/post=1, c = pre=1/post=0).
    """
    b = contingency.loc[0, 1]
    c = contingency.loc[1, 0]
    return b / c if c != 0 else np.nan


def _compute_cramers_v(ct: pd.DataFrame) -> float:
    """
    Compute Cramer's V effect size for a contingency table.

    Parameters
    ----------
    ct : pd.DataFrame
        Contingency table.

    Returns
    -------
    float
        Cramer's V.
    """
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def paired_tests(df: pd.DataFrame,
                 pre_binary_question:str='presleep_physical_complaints_today',
                 post_binary_question:str='postsleep_feeling_sick',
                 test_name:str='physical') -> pd.DataFrame:
    """
    Perform within-subject tests comparing pre- vs post-sleep perceptions.

    - Wilcoxon signed-rank (with effect size) for ordinal deltas.
    - McNemar's test for binary change in physical complaints, with odds ratio.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'delta_' columns and binary complaint columns.

    Returns
    -------
    frame
        frame of test statistics, p-values, effect sizes, and OR.
    """
    results = {}

    # Wilcoxon signed-rank tests
    for var in ['sleepy', 'tired', 'alert']:
        diff = df[f'delta_{var}'].dropna().values
        results[f'wilcoxon_{var}'] = _compute_wilcoxon(diff)

    # McNemar's test + odds ratio
    contingency = pd.crosstab(
        df[pre_binary_question],
        df[post_binary_question]
    )
    mcn = mcnemar(contingency, exact=True)
    or_val = _compute_odds_ratio(contingency)
    results[f'mcnemar_{test_name}'] = {
        'method': 'McNemar’s Test',
        'statistic': mcn.statistic,
        'p_value': mcn.pvalue,
        'odds_ratio': f"{or_val:.2f}"
    }

    # Build DataFrame
    df_results = pd.DataFrame.from_dict(results, orient='index')

    reject, p_adj, _, _ = multipletests(df_results.p_value,
                                        method='fdr_bh',
                                        alpha=0.05)
    df_results['p_adjusted'] = p_adj
    df_results['reject'] = reject

    # Move the index into a column
    df_results.index.name = 'variable'
    df_results = df_results.reset_index()
    df_results['variable'] = df_results['variable'].str.replace('wilcoxon_', '').replace('mcnemar_physical', 'physical change')
    # Optional: reorder columns, filling missing ones with NaN
    cols = ['variable', 'method',  'p_value', 'p_adjusted', 'effect_size_r',  'odds_ratio', 'statistic', 'reject']
    df_results = df_results.reindex(columns=cols)
    return df_results


def stratified_tests(df: pd.DataFrame,
                     strata: str = 'osa_four_numeric',
                     alpha:float=0.05) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Test whether overnight changes differ by OSA severity.

    - Records sample sizes for each severity group.
    - Kruskal–Wallis for ordinal deltas across severity groups.
    - Pairwise Mann–Whitney U post hoc tests with Bonferroni correction.
    - Chi-square for binary complaint-change rates across severity, with Cramer's V.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with strata and test variables.
    strata : str
        Column name indicating OSA severity groups.

    Returns
    -------
    Dataframe
        Of stratified test results, including:
        - 'group_sizes': sample size per severity group
        - 'kruskal_<var>': KW statistic and p-value for each ordinal var
        - 'n_<var>': sample size per group for each delta var
        - 'posthoc_<var>': list of pairwise comparisons with U, p-values, and Bonferroni-adjusted p-values
        - 'chi2_physical_change': Chi-square, p-value, dof, and Cramer's V for physical complaints change
    """
    from scipy.stats import chi2
    # Record sample sizes per severity group
    group_sizes = df[strata].value_counts().to_dict()
    record_group_size = []
    for grp, size in group_sizes.items():
        record_group_size.append({
            'variable': 'group_size',
            'group1': grp,
            'test': 'count',
            'statistic': size,
        })
    df_record_group_size = pd.DataFrame.from_records(record_group_size)

    # Degrees of freedom and chi-squared critical value
    g = len(df[strata].unique())
    df_kruskal = g - 1
    chi2_critical = chi2.ppf(1 - alpha, df_kruskal) if g > 1 else np.nan

    # Kruskal–Wallis and post hoc for ordinal deltas
    records_kruskal = []
    records_pair_wise = [ ]
    for var in ['sleepy', 'tired', 'alert']:
        # Prepare grouped data
        series = df[f'delta_{var}']
        grouped = {lab: series[df[strata] == lab].dropna().values
                   for lab in df[strata].unique()}
        n = sum(len(gv) for gv in grouped.values())

        # Kruskal–Wallis
        stat, p_kw = kruskal(*grouped.values())
        epsilon2 = stat / (n - 1) if n > 1 else np.nan

        records_kruskal.append({
            'variable': var,
            'group1': 'all_groups',
            'test': 'kruskal_wallis',
            'statistic': stat,
            'p_uncorrected': p_kw,
            'epsilon_squared': epsilon2,
            'chi2_critical': chi2_critical
        })

        # Pairwise post-hoc
        labels = list(grouped.keys())
        p_vals = []
        comps = []
        for g1, g2 in combinations(labels, 2):
            d1, d2 = grouped[g1], grouped[g2]
            n1, n2 = len(d1), len(d2)
            u_stat, p_u = mannwhitneyu(d1, d2, alternative='two-sided')
            # effect sizes
            r_rb = (2 * u_stat) / (n1 * n2) - 1  # rank-biserial correlation
            mu_u = n1 * n2 / 2
            sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            Z = (u_stat - mu_u) / sigma_u
            r_z = Z / np.sqrt(n1 + n2)
            p_vals.append(p_u)
            comps.append((g1, g2, u_stat, p_u, r_rb, r_z))

        # Adjust p-values
        reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        for (g1, g2, u_stat, p_u, r_rb, r_z), p_cor, rej in zip(comps, p_adj, reject):
            records_pair_wise.append({
                'variable': var,
                'group1': g1,
                'group2': g2,
                'test': 'mannwhitney_posthoc',
                'statistic': u_stat,
                'p_uncorrected': p_u,
                'p_adjusted': p_cor,
                'r_rank_biserial': r_rb,
                'r_z': r_z,
                'cramers_v': np.nan
            })
    records_binary = []
    # Chi-square + Cramer's V for binary change
    df['physical_changed'] = df['presleep_physical_complaints_today'] != df['postsleep_feeling_sick']
    ct = pd.crosstab(df['physical_changed'], df[strata])
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    n = ct.values.sum()
    r, k = ct.shape
    cv = np.sqrt(chi2 / (n * (min(r, k) - 1)))
    records_binary.append({
        'variable': 'physical_changed',
        'group1': 'all_groups',
        'test': 'chi_square',
        'statistic': chi2,
        'p_uncorrected': p_chi,
        'cramers_v': cv
    })
     # kruskal wallis all groups
    df_records_kruskal = pd.DataFrame.from_records(records_kruskal)
    df_records_kruskal['statistic'] = df_records_kruskal['statistic'].round(3)
    df_records_kruskal['chi2_critical'] = df_records_kruskal['chi2_critical'].round(4)
    df_records_kruskal['epsilon_squared'] = df_records_kruskal['epsilon_squared'].round(4)

    # Post hoc pair wise results
    df_records_pair_wise = pd.DataFrame.from_records(records_pair_wise)
    columns_order = ['variable', 'group1', 'group2',
                     'test', 'r_rank_biserial', 'r_z',
                     'p_uncorrected', 'p_adjusted',
                     'statistic', 'cramers_v', ]
    df_records_pair_wise = df_records_pair_wise[columns_order]
    df_records_pair_wise['r_rank_biserial'] = df_records_pair_wise['r_rank_biserial'].round(3)
    df_records_pair_wise['r_z'] = df_records_pair_wise['r_z'].round(3)
    df_records_pair_wise['statistic'] = df_records_pair_wise['statistic'].round(3)

    # binary test groups
    df_records_binary = pd.DataFrame.from_records(records_binary)
    df_records_binary['cramers_v'] = df_records_binary['cramers_v'].round(3)
    df_records_binary['statistic'] = df_records_binary['statistic'].round(3)

    return df_records_kruskal,df_records_pair_wise, df_records_binary

# %% explore objective measures across subjective response → Do patients who feel worse actually have more apneas?


def generate_qq_grid_colored(df: pd.DataFrame,
                             variables: List[str],
                             group_col: str,
                             level_mapper:Dict[int,str] = None,
                             var_labels: List[str] = None,
                             output_path: pathlib.Path = None,
                             figsize_per_plot: tuple = (4, 3)):
    """
    Generate a grid of Q–Q plots by group, with color-coded rows and a shared legend.

    Parameters
    ----------
    df : pd.DataFrame
        The input dataframe.
    variables : List[str]
        Continuous variables to plot (each becomes a row).
    group_col : str
        Categorical grouping variable (each level becomes a column).
    var_labels : List[str], optional
        Formal labels for variables (used in legend).
    output_path : pathlib.Path, optional
        Where to save the image.
    figsize_per_plot : tuple
        Size of each subplot (width, height).
    """
    levels = sorted(df[group_col].dropna().unique())
    n_rows, n_cols = len(variables), len(levels)

    # Color palette per row
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in range(n_rows)]

    fig, axs = plt.subplots(n_rows, n_cols,
                            figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
                            sharex=False, sharey=False)

    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)
    if n_cols == 1:
        axs = np.expand_dims(axs, axis=1)

    if var_labels is None:
        var_labels = variables

    for row, var in enumerate(variables):
        for col, level in enumerate(levels):
            ax = axs[row, col]
            data = df[df[group_col] == level][var].dropna()
            stats.probplot(data, dist="norm", plot=ax)

            # Set line and marker color
            ax.get_lines()[1].set_color(colors[row])  # Q-Q line
            ax.get_lines()[0].set_color(colors[row])  # sample points
            if level_mapper:
                level = level_mapper.get(level)
            # Axis titles
            if row == 0:
                ax.set_title(f'{group_col} = {level}', fontsize=11)
            else:
                ax.set_title("")
            if col == 0:
                ax.set_ylabel('Sample Quantiles', fontsize=10)
            else:
                ax.set_ylabel("")
            if row == n_rows - 1:
                ax.set_xlabel('Theoretical Quantiles', fontsize=10)
            else:
                ax.set_xlabel("")

            ax.grid(alpha=0.5)

    fig.suptitle(f'Q–Q Plots by {group_col}', fontsize=20)

    # Add shared legend at top
    handles = [plt.Line2D([0], [0], color=colors[i], label=var_labels[i]) for i in range(n_rows)]
    fig.legend(handles=handles,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.95),
               ncol=n_rows,
               fontsize=18,
               frameon=False)

    # Add space at top to avoid clipping legend
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path.joinpath(f'qq_plot_{group_col}.png'), dpi=300)
    plt.show()
    plt.close()


def compare_resp_across_questionnaire_groups(df: pd.DataFrame,
                                             respiratory_vars: list,
                                             grouping_var: str,
                                             alpha: float = 0.05) -> pd.DataFrame:
    """
    Compare respiratory variables across levels of a post-sleep questionnaire variable.

            Effect Size (epsilon^2):

    epsilon^2 = H / (n - 1), where H is the kruskal-walli statistics and n is the total number of observations.
    This is a robust effect size measure for non-parametric tests, with values typically interpreted as:
    Small: ~0.01
    Medium: ~0.06
    Large: ~0.14

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing respiratory measures and grouping variable.
    respiratory_vars : list of str
        Continuous respiratory variables to compare.
    grouping_var : str
        Categorical variable with 2+ levels (e.g., 'postsleep_feeling_sleepy').
    alpha : float
        Significance level for normality testing and hypothesis testing (default 0.05).

    Returns
    -------
    pd.DataFrame
        Summary of tests applied per variable, including test name, statistic, p-value,
        significance, effect size (epsilon-squared for Kruskal-Wallis), and chi-squared
        critical value (for Kruskal-Wallis).
    """
    results = []

    levels = df[grouping_var].dropna().unique()
    levels.sort()
    g = len(levels)  # Number of groups
    df_kruskal = g - 1  # Degrees of freedom for Kruskal-Wallis

    # Compute chi-squared critical value for Kruskal-Wallis
    chi2_critical = chi2.ppf(1 - alpha, df_kruskal) if g > 1 else None

    for var in respiratory_vars:
        group_data = [df[df[grouping_var] == lvl][var].dropna() for lvl in levels]

        # Skip if not enough data
        if any(len(g) < 3 for g in group_data):
            continue

        # Total number of observations
        n = sum(len(g) for g in group_data)

        # Check normality across all groups
        normal_flags = [shapiro(g)[1] > alpha for g in group_data]
        is_normal = all(normal_flags)

        if is_normal:
            stat, p = f_oneway(*group_data)
            test = 'ANOVA'
            effect_size = None  # Effect size not computed for ANOVA here
            critical_value = None  # No chi-squared critical value for ANOVA
        else:
            stat, p = kruskal(*group_data)
            test = 'Kruskal–Wallis'
            # Compute epsilon-squared effect size: H / (n - 1)
            effect_size = stat / (n - 1) if n > 1 else None
            critical_value = chi2_critical

        results.append({
            'variable': var,  # The respiratory variable.
            'test': test,  # ANOVA or Kruskal-Wallis.
            'statistic': round(stat, 3),  # Test statistic (H for Kruskal-Wallis).
            'p_value': p,
            'significant': p < alpha,
            # Epsilon-squared for Kruskal-Wallis (None for ANOVA).
            'effect_size_epsilon2': round(effect_size, 3) if effect_size is not None else None,
            # Chi-squared critical value for Kruskal-Wallis (None for ANOVA).
            'chi2_critical_value': round(critical_value, 3) if critical_value is not None else None
        })

    df_results = pd.DataFrame(results)
    reject, p_adj, _, _ = multipletests(df_results.p_value,
                                        method='fdr_bh',
                                        alpha=alpha)
    df_results['p_adjusted'] = p_adj
    df_results['reject'] = reject
    col_oder = ['variable', 'test', 'statistic',
                'p_value','p_adjusted',
                'significant', 'reject',
                'effect_size_epsilon2', 'chi2_critical_value']
    df_results = df_results.reindex(columns=col_oder)

    return df_results

# Post Hoc test
def stratified_posthoc_continuous_by_ordinal_group(df: pd.DataFrame,
                                                   group_col: str,
                                                   continuous_vars: List[str],
                                                   alpha: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform Kruskal–Wallis and post hoc Mann–Whitney U tests for continuous variables
    across ordinal group levels (e.g., post-sleep perception levels).

    Parameters
    ----------
    df : pd.DataFrame
    group_col : str
        Column with ordinal group levels (e.g., 'Feels Tired').
    continuous_vars : list of str
        List of continuous variables to compare (e.g., AHI, OA, CA...).
    alpha : float
        Significance level.

    Returns
    -------
    Tuple of:
        - Kruskal–Wallis summary table
        - Pairwise post hoc results (Mann–Whitney U)
    """
    from scipy.stats import kruskal, mannwhitneyu, chi2
    from itertools import combinations

    results_kw = []
    results_posthoc = []

    levels = sorted(df[group_col].dropna().unique())
    df_kw = len(levels) - 1
    chi2_crit = chi2.ppf(1 - alpha, df_kw)

    for var in continuous_vars:
        grouped = {lvl: df[df[group_col] == lvl][var].dropna().values for lvl in levels}
        n_total = sum(len(vals) for vals in grouped.values())

        # Kruskal–Wallis test
        kw_stat, p_kw = kruskal(*grouped.values())
        epsilon2 = kw_stat / (n_total - 1) if n_total > 1 else np.nan

        results_kw.append({
            'variable': var,
            'group_col': group_col,
            'statistic': round(kw_stat, 3),
            'p_value': p_kw,
            'epsilon_squared': round(epsilon2, 3),
            'chi2_critical': round(chi2_crit, 3),
        })

        # Post hoc pairwise tests
        pairwise_results = []
        p_vals = []
        comps = []
        for g1, g2 in combinations(levels, 2):
            d1, d2 = grouped[g1], grouped[g2]
            n1, n2 = len(d1), len(d2)
            u_stat, p = mannwhitneyu(d1, d2, alternative='two-sided')

            # Effect sizes
            r_rb = (2 * u_stat) / (n1 * n2) - 1
            mu_u = n1 * n2 / 2
            sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            Z = (u_stat - mu_u) / sigma_u
            r_z = Z / np.sqrt(n1 + n2)

            comps.append((g1, g2, u_stat, p, r_rb, r_z))
            p_vals.append(p)

        reject, p_adj, _, _ = multipletests(p_vals, method='fdr_bh')
        for (g1, g2, u_stat, p, r_rb, r_z), p_corr, rej in zip(comps, p_adj, reject):
            results_posthoc.append({
                'variable': var,
                'group_col': group_col,
                'group1': g1,
                'group2': g2,
                'test': 'mannwhitney_posthoc',
                'statistic': round(u_stat, 3),
                'p_uncorrected': p,
                'p_adjusted': p_corr,
                'reject': rej,
                'r_rank_biserial': round(r_rb, 3),
                'r_z': round(r_z, 3)
            })

    df_kw = pd.DataFrame(results_kw)
    df_posthoc = pd.DataFrame(results_posthoc)
    return df_kw, df_posthoc


# %% main
if __name__ == '__main__':
    osa_mapper = {
        0: 'Normal',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
    }
    path_output = config.get('results')['dir'].joinpath('post_vs_pre')
    path_output.mkdir(exist_ok=True, parents=True)

    # Input data
    df = pd.read_csv(
        config.get('data')['pp_data']['q_resp'],
        low_memory=False
    )
    columns_compare = {
        'delta_sleepy': ['presleep_feel_sleepy_now', 'postsleep_feeling_sleepy'],
        'delta_tired': ['presleep_feel_tired_now', 'postsleep_feeling_tired'],
        'delta_alert': ['presleep_feel_alert_now', 'postsleep_feeling_alert']
    }

    #%% check they all have the same numerical values
    df_pre = pd.DataFrame()
    df_post = pd.DataFrame()
    for key, pairs in columns_compare.items():
        assert len(set(df[pairs[0]].unique()) ^ set(df[pairs[1]].unique())) == 0
        print()
        df_pre_tmp = pd.DataFrame.from_records(encoding.get(pairs[0]))
        # df_pre.rename({'definition':pairs[0]}, inplace=True, axis=1)
        df_pre_tmp.reset_index(inplace=True, drop=False, names=['questions'])
        df_pre_tmp.sort_values(by=['encoding'], inplace=True)
        df_pre_tmp['definition'] = key

        df_post_tmp = pd.DataFrame.from_records(encoding.get(pairs[1]))
        # df_post.rename({'definition':pairs[1]}, inplace=True, axis=1)
        df_post_tmp.reset_index(inplace=True, drop=False, names=['questions'])
        df_post_tmp.sort_values(by=['encoding'], inplace=True)
        df_post_tmp['definition'] = key

        df_pre = pd.concat([df_pre, df_pre_tmp], axis=0)
        df_post = pd.concat([df_post, df_post_tmp], axis=0)

    df_pairs = pd.concat([df_pre, df_post], axis=1)
    print(tabulate(
        df_pairs,
        headers='keys',
        tablefmt='psql',
        showindex=False
    ))

    # %% =================
    # First we will explore subjective change across objective severity (OSA) → Do patients with worse AHI feel worse?
    # ===================
    #%% Prepare data
    df = compute_deltas(df, columns_compare)

    if (not (path_output.joinpath('kruskal_tests.csv').is_file()) or
            not(path_output.joinpath('pairwise_tests.csv')) or
            not (path_output.joinpath('binary_tests.csv'))):
        # Run statistical tests
        df_paired_results = paired_tests(df,
                                         pre_binary_question='presleep_physical_complaints_today',
                                         post_binary_question='postsleep_feeling_sick',
                                         test_name='physical')
        df_paired_results['variable'] = df_paired_results['variable'].str.capitalize()

        df_paired_results.to_csv(path_output.joinpath('within_paired_tests.csv'), index=False)

        (df_records_kruskal,
         df_records_pair_wise,
         df_records_binary) = stratified_tests(df,
                                               strata='osa_four_numeric',
                                               alpha=0.05)

        for frame in [df_records_kruskal, df_records_pair_wise, df_records_binary]:
            if 'group1' in frame.columns:
                frame['group1'] = frame['group1'].replace(osa_mapper)
            if 'group2' in frame.columns:
                frame['group2'] = frame['group2'].replace(osa_mapper)

        df_records_kruskal['variable'] = df_records_kruskal['variable'].str.capitalize()
        df_records_pair_wise['variable'] = df_records_pair_wise['variable'].str.capitalize()
        df_records_binary['variable'] = df_records_binary['variable'].str.capitalize()


        df_records_kruskal.to_csv(path_output.joinpath('kruskal_tests.csv'), index=False)
        df_records_pair_wise.to_csv(path_output.joinpath('pairwise_tests.csv'), index=False)
        df_records_binary.to_csv(path_output.joinpath('binary_tests.csv'), index=False)



        # Display results
        print("=== Paired Tests (Pre vs. Post PSG) ===")
        print(tabulate(
            df_paired_results,
            headers='keys',
            tablefmt='psql',
            showindex=False
        ))

        print("\n=== Stratified Tests by OSA Severity ===")

        print("\n-Kruskal all groups ===")
        print(tabulate(
            df_records_kruskal,
            headers='keys',
            tablefmt='psql',
            showindex=False
        ))

        print("\n-Post Hoc within groups ===")
        print(tabulate(
            df_records_pair_wise,
            headers='keys',
            tablefmt='psql',
            showindex=False
        ))


        print("\n-Binary test ===")
        print(tabulate(
            df_records_binary,
            headers='keys',
            tablefmt='psql',
            showindex=False
        ))


    # ----------------------------------------------
    # 1. Define mappings and columns of interest
    # ----------------------------------------------

    # Formal display names for variables
    mapper_values = {
        'ahi': 'AHI',
        'resp-oa-total': 'Obstructive Apneas',
        'resp-ca-total': 'Central Apneas',
        'resp-ma-total': 'Mixed Apneas',
        'resp-hi_hypopneas_only-total': 'Hypopneas',
        'resp-ri_rera_only-total': 'RERAs',
        'dem_age': 'Age',
        'dem_bmi': 'BMI',
        'postsleep_feeling_sleepy': 'Feels Sleepy',
        'postsleep_feeling_tired': 'Feels Tired',
        'postsleep_feeling_alert': 'Feels Alert',
    }
    mapper_values_inv = {val:key for key, val in mapper_values.items()}
    # Variables to analyze
    respiratory_vars = [
        'ahi',
        'resp-oa-total',
        'resp-ca-total',
        'resp-ma-total',
        'resp-hi_hypopneas_only-total',
        'resp-ri_rera_only-total'
    ]

    post_sleep_vars = [val[1] for val in columns_compare.values()]
    adjustment_vars = ['dem_age', 'dem_bmi']

    # Combined list of variables to include in the analysis
    columns_tab = respiratory_vars + post_sleep_vars + adjustment_vars

    # ----------------------------------------------
    # 2. Identify variable types
    # ----------------------------------------------

    vars_bin = [var for var in columns_tab if df[var].max() <= 1]
    vars_ordinal = [var for var in columns_tab if df[var].nunique() <= 10 and var not in vars_bin]
    vars_continuous = [var for var in columns_tab if var not in vars_ordinal + vars_bin]
    vars_categorical = vars_ordinal + vars_bin  # categorical = ordinal + binary

    # ----------------------------------------------
    # 3. Rename columns using formal labels
    # ----------------------------------------------

    # Subset and rename DataFrame
    df_tab = df[columns_tab].copy().rename(columns=mapper_values)

    # Update variable lists with formal names
    vars_continuous = [mapper_values[var] for var in vars_continuous]
    vars_categorical = [mapper_values[var] for var in vars_categorical]
    respiratory_vars = [mapper_values[var] for var in respiratory_vars]
    post_sleep_vars = [mapper_values[var] for var in post_sleep_vars]


    # %% =================
    # Then, we will explore objective measures across subjective response → Do patients who feel worse actually have
    # more apneas?
    # ===================
    # %% Table one with all the respiratory measures and statistical test
    EXPLORE_MEASURES_ACROSS_SUBJECTIVNESS_RESPONSES = False
    if EXPLORE_MEASURES_ACROSS_SUBJECTIVNESS_RESPONSES:
        for strata in post_sleep_vars:
            tab_post_sleep = MakeTableOne(df=df_tab,
                                          continuous_var=vars_continuous,
                                          categorical_var=[],
                                          strata=strata)
            # col_labels_inv = encoding.get(mapper_values_inv.get((post_sleep_vars[0])))['encoding']
            # col_labels = {val:key for key, val in col_labels_inv.items()}
            df_post_sleep = tab_post_sleep.create_table()
            # df_post_sleep = df_post_sleep.rename(col_labels, axis=1)
            df_post_sleep.to_csv(path_output / f'table_one_{strata}.csv', index=False)




        # col_labels_inv = encoding.get(mapper_values_inv.get((post_sleep_vars[0])))['encoding']
        # col_labels = {val:key for key, val in col_labels_inv.items()}
        # 1. visualize distributions
        for grouping_var in post_sleep_vars:
            col_labels_inv = encoding.get(mapper_values_inv.get((grouping_var)))['encoding']
            col_labels = {val: key for key, val in col_labels_inv.items()}
            generate_qq_grid_colored(df_tab,
                                        variables=respiratory_vars,
                                        group_col=grouping_var,
                                        level_mapper=col_labels,
                                        output_path=path_output)

        for grouping_var in post_sleep_vars:
            df_resp_comparison = compare_resp_across_questionnaire_groups(df_tab,
                                                                          respiratory_vars,
                                                                          grouping_var)

            df_resp_comparison.to_csv(path_output / f"resp_by_{grouping_var}.csv", index=False)
            print(tabulate(
                df_resp_comparison,
                headers='keys',
                tablefmt='psql',
                showindex=False
            ))



        for perception_var in ['Feels Sleepy', 'Feels Tired', 'Feels Alert']:
            df_kw, df_posthoc = stratified_posthoc_continuous_by_ordinal_group(
                df=df_tab,
                group_col=perception_var,
                continuous_vars=respiratory_vars
            )
            col_labels_inv = encoding.get(mapper_values_inv.get((perception_var)))['encoding']
            col_labels = {val: key for key, val in col_labels_inv.items()}
            df_posthoc['group1'] = df_posthoc['group1'].replace(col_labels)
            df_posthoc['group2'] = df_posthoc['group2'].replace(col_labels)

            df_posthoc['group1'] = df_posthoc['group1'].str.capitalize()
            df_posthoc['group2'] = df_posthoc['group2'].str.capitalize()

            df_posthoc.to_csv(path_output / f"posthoc_kw_by_{perception_var}.csv", index=False)

            # Save or analyze df_kw and df_posthoc

    # %% regresion based model
    import statsmodels.formula.api as smf
    from sklearn.preprocessing import StandardScaler
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    import statsmodels.api as sm


    def ordinal_logistic_regression(df: pd.DataFrame,
                                    outcome_var: str,
                                    predictor_vars: List[str],
                                    adjust_vars: List[str],
                                    standardize_vars: List[str] = ['dem_age', 'dem_bmi']) -> pd.DataFrame:
        """
        Fit an ordinal logistic regression model with HC1 robust standard errors.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe containing all relevant variables.
        outcome_var : str
            Ordinal outcome variable with 4 levels (0 is reference).
        predictor_vars : List[str]
            List of primary predictor variables (e.g., AHI, OA, CA, etc.).
        adjust_vars : List[str]
            List of adjustment variables (e.g., age, bmi, gender, race).
        standardize_vars : List[str]
            Subset of adjust_vars to z-score (e.g., continuous variables).

        Returns
        -------
        pd.DataFrame
            Summary dataframe with ORs, 95% CI, robust SE, and p-values.
        """
        data = df.copy()

        # Z-score standardization
        means = data[standardize_vars].mean()
        medians = data[standardize_vars].median()
        std = data[standardize_vars].std()
        print("Means of standardized variables:\n", means)
        print("Medians of standardized variables:\n", medians)
        print("Std of standardized variables:\n", std)

        scaler = StandardScaler()
        data[standardize_vars] = scaler.fit_transform(data[standardize_vars])

        # Encode categorical variables manually
        categorical_vars = [var for var in adjust_vars if var not in standardize_vars]
        data = pd.get_dummies(data, columns=categorical_vars, drop_first=True)
        cols_dummy = [col for col in data.columns if any(var + "_" in col for var in categorical_vars)]
        data[cols_dummy] = data[cols_dummy].astype(int)

        # Final predictor set
        final_predictors = predictor_vars + standardize_vars + \
                           [col for col in data.columns if any(col.startswith(var + "_") for var in categorical_vars)]

        # Convert outcome to ordered categorical
        data[outcome_var] = pd.Categorical(data[outcome_var], ordered=True)

        # Fit model
        columns = final_predictors + [outcome_var]
        df_model = data.dropna(subset=columns).copy()
        model = OrderedModel(df_model[outcome_var],
                             df_model[final_predictors],
                             distr='logit')
        res = model.fit(method='bfgs', cov_type='HC1', disp=False)

        # Output summary
        params = res.params
        conf = res.conf_int()
        se = res.bse
        pvals = res.pvalues
        or_ = np.exp(params)
        or_ci = np.exp(conf)

        summary_df = pd.DataFrame({
            'OR': or_,
            '2.5% CI': or_ci[0],
            '97.5% CI': or_ci[1],
            'Robust SE': se,
            'p-value': pvals,
            'samples': len(df_model)
        })

        summary_df.index.name = 'Variable'
        summary_df.reset_index(inplace=True, drop=False, names=['Variable'])
        summary_df = summary_df.round(4)
        summary_df['CI'] = list(zip(summary_df['2.5% CI'].round(4), summary_df['97.5% CI'].round(4)))
        return summary_df

    outcome_vars = ['postsleep_feeling_sleepy','postsleep_feeling_tired', 'postsleep_feeling_alert']
    summary_df = ordinal_logistic_regression(
        df=df,
        outcome_var=outcome_vars[0],
        predictor_vars=['ahi', 'resp-oa-total', 'resp-ca-total', 'resp-ma-total',
                        'resp-hi_hypopneas_only-total', 'resp-ri_rera_only-total'],
        adjust_vars=['dem_age', 'dem_bmi', 'dem_gender', 'dem_race'],
        standardize_vars=['dem_age', 'dem_bmi']
    )

    print("\n-Binary test ===")
    print(tabulate(
        summary_df,
        headers='keys',
        tablefmt='psql',
        showindex=False
    ))


    # Great question. Let's break it down both **statistically** and in terms of your **research goal**.
    #
    # ---
    #
    # ### 🧪 Your Research Question:
    #
    # > **“Do patients who feel worse actually have more apneas?”**
    # > Or more generally:
    # > **“Is there an association between subjective perception (ordinal) and objective respiratory metrics (continuous)?”**
    #
    # ---
    #
    # ## 🔍 Two Approaches Compared
    #
    # | Method                                                         | Description                                                                                                                                                                                      |
    # | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
    # | **Non-parametric tests** (e.g., Kruskal–Wallis + effect sizes) | Compares distributions of objective metrics across ordered subjective categories, **without assuming a specific model structure**.                                                               |
    # | **Ordinal Logistic Regression**                                | Models the probability of being in higher (worse) subjective categories as a function of objective predictors, **adjusting for confounders** and providing **interpretable effect sizes (ORs)**. |
    #
    # ---
    #
    # ## ✅ Which is “better”? It depends on your analytic goal:
    #
    # ### ✅ **If your goal is *exploratory* and descriptive**:
    #
    # * You're checking **whether** objective variables differ by perceived severity (e.g., more apneas in those who feel more tired):
    # * **→ Non-parametric tests are better.**
    # * Why?
    #
    #   * Minimal assumptions.
    #   * Captures **distributional differences**.
    #   * You can **directly test** each respiratory metric **across subjective categories**.
    #   * You get **rank-biserial correlations** (effect sizes) that are **robust**.
    #
    # **BUT**: they don't adjust for confounders like age, BMI, etc.
    #
    # ---
    #
    # ### ✅ **If your goal is *explanatory* and adjusted**:
    #
    # * You want to **model the likelihood** of feeling worse as a function of apneas, adjusting for age, gender, BMI, race:
    # * **→ Ordinal logistic regression is better.**
    # * Why?
    #
    #   * Models the **effect of each respiratory metric**, **controlling for covariates**.
    #   * Outputs **ORs**, CIs, and p-values.
    #   * Provides a **predictive structure** (log-odds) across ordered outcomes.
    #   * Can handle multiple predictors simultaneously and check for confounding.
    #
    # ---
    #
    # ## 🧩 What about **statistical assumptions**?
    #
    # | Assumption                        | Kruskal–Wallis + Effect Sizes | Ordinal Logistic Regression                  |
    # | --------------------------------- | ----------------------------- | -------------------------------------------- |
    # | Distribution-Free                 | ✅ Yes                         | ❌ No (requires proportional odds assumption) |
    # | Handles non-normality             | ✅ Yes                         | ⚠️ Needs ordinal logistic assumptions        |
    # | Adjusts for confounding           | ❌ No                          | ✅ Yes                                        |
    # | Interaction effects               | ❌ No                          | ✅ Can include                                |
    # | Provides interpretable model (OR) | ❌ No                          | ✅ Yes                                        |
    #
    # ---
    #
    # ### 🧠 So, what’s the **best approach for your question**?
    #
    # > **Use both** in a complementary way.
    #
    # #### Suggested strategy:
    #
    # 1. **Start with Kruskal–Wallis + effect sizes** to see:
    #
    #    * Which respiratory metrics **differ** across subjective categories?
    #    * What’s the **pattern of change**?
    #
    # 2. **Then use ordinal logistic regression** to:
    #
    #    * Quantify the **adjusted association**.
    #    * Estimate whether, e.g., RERAs still predict increased sleepiness after adjusting for age, gender, BMI, etc.
    #    * **Test causality-adjusted hypotheses**.
    #
    # ---
    #
    # ## ✅ TL;DR:
    #
    # * **Effect sizes from non-parametric tests** are **robust, exploratory, and good for descriptive summaries**.
    # * **Ordinal logistic regression** is **more powerful for causal inference, adjustment, and effect estimation**.
    # * Use **effect sizes** to show **distributional differences** and **ordinal regression** to test **adjusted predictive effects**.
    #
    # Let me know if you'd like a visual comparing the results of the two methods side by side!

