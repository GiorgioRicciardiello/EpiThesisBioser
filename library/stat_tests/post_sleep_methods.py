import pathlib
from config.config import config, sections, encoding
from library.helper import get_mappers, pretty_col_name, apply_inverse_transform_betas_to_df
import pandas  as pd
from library.TableOne.table_one import MakeTableOne
import statsmodels.api as sm
from typing import List, Dict, Optional, Tuple
from statsmodels.miscmodels.ordinal_model import OrderedModel
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import statsmodels.formula.api as smf
from typing import List, Optional

all_map, grouped_map = get_mappers()



def plot_ahi_split_by_sex(
    df,
    y_continuous: str,
    x_ordinal: List[str],
    group_col: str = 'gender',
    ncols: int = 2,
output_path:pathlib.Path = None,
):
    """
    Generates box plots of a continuous variable (e.g., AHI) for several
    ordinal predictors, split by a grouping column (e.g., gender), arranged
    in a grid with `ncols` columns.

    :param df: pandas DataFrame with your data
    :param y_continuous: name of the continuous column (e.g., 'ahi')
    :param x_ordinal: list of ordinal column names to plot along x-axes
    :param group_col: column name to split each boxplot (default 'gender')
    :param ncols: number of subplot columns
    """
    n_vars = len(x_ordinal)
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(8 * ncols, 5 * nrows),
        sharey=True
    )

    # flatten axes array for easy iteration
    axes_flat = np.array(axes).reshape(-1)

    legend_handles = None
    genders = sorted(df[group_col].dropna().unique())

    for ax, var in zip(axes_flat, x_ordinal):
        levels = sorted(df[var].dropna().unique())
        ind = np.arange(len(levels))
        width = 0.35

        # draw one boxplot per gender
        boxes = []
        for i, gender in enumerate(genders):
            data = [
                df[(df[var] == lvl) & (df[group_col] == gender)][y_continuous].dropna()
                for lvl in levels
            ]
            positions = ind + (i - 0.5) * width
            bp = ax.boxplot(
                data,
                positions=positions,
                widths=width,
                patch_artist=True,
                boxprops=dict(facecolor=['skyblue', 'lightcoral'][i]),
                medianprops=dict(color='black')
            )
            boxes.append(bp["boxes"][0])

        # capture handles once for figure-level legend
        if legend_handles is None:
            legend_handles = boxes

        # x-tick labels
        lbl_map = {v: k for k, v in all_map[var]['encoding'].items()}
        xticklabels = [
            pretty_col_name(lbl_map.get(lvl, lvl), max_width=10)
            for lvl in levels
        ]
        ax.set_xticks(ind)
        ax.set_xticklabels(xticklabels, rotation=15, ha='right')

        # labels & titles
        ax.set_xlabel(pretty_col_name(var))
        if ax is axes_flat[0]:
            ax.set_ylabel(f"{y_continuous} (events/hr)")
        ax.set_title(pretty_col_name(var))

    # hide any unused subplots
    for ax in axes_flat[n_vars:]:
        ax.set_visible(False)

    # single legend for the whole figure
    fig.legend(
        legend_handles,
        [f"{group_col} = {g}" for g in genders],
        loc='upper right',
        title=pretty_col_name(group_col)
    )
    plt.grid(axis='y', alpha=0.7)
    fig.tight_layout(rect=[0, 0, 0.95, 1])
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

def compute_corr_matrices(
        df: pd.DataFrame,
        vars_bin: List[str],
        vars_ord: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute correlation matrices for binary and ordinal postsleep variables.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the variables.
    vars_bin : list of str
        List of binary variable names.
    vars_ord : list of str
        List of ordinal variable names.

    Returns
    -------
    corr_bin : pd.DataFrame
        Pearson correlation matrix for binary variables.
    corr_ord : pd.DataFrame
        Spearman correlation matrix for ordinal variables.
    """
    corr_bin = df[vars_bin].corr(method='pearson')
    corr_ord = df[vars_ord].corr(method='spearman')
    return corr_bin, corr_ord


def plot_corr_matrix(corr: pd.DataFrame,
                     title: str,
                     ouput_path:pathlib.Path=None) -> None:
    """
    Plot a heatmap of the correlation matrix.

    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix to plot.
    title : str
        Title for the heatmap.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    if ouput_path:
        plt.savefig(ouput_path.joinpath(f'correlation_{title}.png'), dpi=300)
    plt.show()

# %% Regression model to evalaute the association between the sbujectivness in sleep quality and the metrics
def _fit_ols(
        df: pd.DataFrame,
        outcome: str,
        exposure: str,
        covariates: Optional[List[str]] = None,
        cat_vars: Optional[List[str]] = None,
        cov_type: str = 'HC3'
):
    """
    Fit an OLS regression of a continuous outcome on one exposure and optional covariates.
    Categorical variables (exposure or covariates) wrapped as C(var).
    Returns the fitted results object with robust standard errors.
    """
    # Validate input columns exist
    required_cols = [outcome, exposure] + (covariates or [])
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

    clean = df.dropna(subset=required_cols).copy()
    terms = []
    if cat_vars and exposure in cat_vars:
        terms.append(f"C({exposure})")
    else:
        terms.append(exposure)
    for cov in covariates or []:
        if cat_vars and cov in cat_vars:
            terms.append(f"C({cov})")
        else:
            terms.append(cov)
    formula = f"{outcome} ~ " + " + ".join(terms)
    model = smf.ols(formula, data=clean).fit(cov_type=cov_type)
    return model


def _extract_regression_stats(
        model,
        exposure: str,
        cat: bool = False,
        prefix: str = ''
) -> pd.DataFrame:
    """
    Extract coefficients, 95% CI, and p-values for terms involving the exposure.
    If categorical, extracts each level term C(exposure)[T.level].
    """
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues

    if cat:
        mask = params.index.str.startswith(f"C({exposure})")
    else:
        mask = params.index == exposure

    df_stats = pd.DataFrame({
        f'{prefix}coef': params[mask],
        f'{prefix}LCL': conf.loc[mask, 0],
        f'{prefix}UCL': conf.loc[mask, 1],
        f'{prefix}p': pvals[mask]
    })
    df_stats.index = df_stats.index.str.replace(r'.*T\.', '', regex=True)
    return df_stats.round({f'{prefix}coef': 3, f'{prefix}LCL': 3, f'{prefix}UCL': 3, f'{prefix}p': 3})


def _reshape_to_wide_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the regression summary DataFrame to a wide format, merging Unadjusted and Adjusted
    results by exposure and level, and combining LCL/UCL into a single CI column.
    """
    # Split by model type
    unadj = summary[summary['Model'] == 'Unadjusted'].copy()
    adj = summary[summary['Model'] == 'Adjusted'].copy()

    # Rename columns
    unadj = unadj.rename(columns={
        'coef': 'unadj_coef',
        'LCL': 'unadj_LCL',
        'UCL': 'unadj_UCL',
        'p': 'unadj_p',
        'r2': 'unadj_r2',
        'adj_r2': 'unadj_adj_r2',
        'AIC': 'unadj_AIC',
        'BIC': 'unadj_BIC',
    })
    adj = adj.rename(columns={
        'coef': 'adj_coef',
        'LCL': 'adj_LCL',
        'UCL': 'adj_UCL',
        'p': 'adj_p',
        'r2': 'adj_r2',
        'adj_r2': 'adj_adj_r2',
        'AIC': 'adj_AIC',
        'BIC': 'adj_BIC',
    })
    # Drop columns with all NaN values
    unadj = unadj.dropna(axis=1, how='all')
    adj = adj.dropna(axis=1, how='all')
    # Select only needed cols
    # unadj = unadj[['exposure', 'level', 'unadj_coef', 'unadj_LCL', 'unadj_UCL', 'unadj_p']]
    # adj = adj[['exposure', 'level', 'adj_coef', 'adj_LCL', 'adj_UCL', 'adj_p']]

    # Merge on exposure & level
    merged = pd.merge(unadj, adj, on=['exposure', 'level'], how='outer')

    # Round numeric columns to 3 decimals
    for col in ['unadj_coef', 'unadj_LCL', 'unadj_UCL', 'unadj_p',
                'adj_coef', 'adj_LCL', 'adj_UCL', 'adj_p']:
        merged[col] = merged[col].round(3)

    # Combine LCL/UCL into tuple strings
    merged['unadj_CI'] = merged.apply(
        lambda r: f"({r.unadj_LCL:.3f}, {r.unadj_UCL:.3f})", axis=1
    )
    merged['adj_CI'] = merged.apply(
        lambda r: f"({r.adj_LCL:.3f}, {r.adj_UCL:.3f})", axis=1
    )

    # Final column order
    cols = merged.columns.tolist()
    first = [c for c in ['exposure', 'level'] if c in cols]
    unadj = [c for c in cols if c.startswith('unadj_')]
    adj = [c for c in cols if c.startswith('adj_')]
    others = [c for c in cols if c not in first + unadj + adj]
    new_order = first + unadj + adj + others
    # Drop duplciate columns
    merged = merged[new_order]
    dup_bases = {
        col[:-2]
        for col in merged.columns
        if col.endswith('_x') and f"{col[:-2]}_y" in merged.columns
    }

    for base in dup_bases:
        x_col = f"{base}_x"
        y_col = f"{base}_y"
        merged[base] = merged[x_col].combine_first(merged[y_col])

    # 3) drop all the old suffixed columns
    to_drop = [f"{b}_x" for b in dup_bases] + [f"{b}_y" for b in dup_bases]
    merged = merged.drop(columns=to_drop)

    return merged

def build_regression_summary(
        df: pd.DataFrame,
        exposures: List[str],
        outcome: str,
        adjust_vars: Optional[List[str]] = None,
        cat_exposures: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    For each exposure, fit unadjusted and adjusted OLS of outcome ~ exposure (+ covariates),
    extracting coefficients, CIs, and p-values.
    """
    blocks = []
    for model_type, covs in [('Unadjusted', None), ('Adjusted', adjust_vars)]:
        dfs = []
        for exp in exposures:
            m = _fit_ols(df, outcome, exp, covariates=covs, cat_vars=cat_exposures)
            stats = _extract_regression_stats(
                m, exp, cat=(cat_exposures and exp in cat_exposures),
                prefix='' if model_type == 'Unadjusted' else 'adj_'
            )
            stats = stats.reset_index().rename(columns={'index': 'level'})
            stats['exposure'] = exp
            # model‐fit metrics
            stats['r2'] = round(m.rsquared, 3)
            stats['adj_r2'] = round(m.rsquared_adj, 3)
            stats['AIC'] = round(m.aic, 3)
            stats['BIC'] = round(m.bic, 3)
            stats['nobs'] = int(m.nobs)
            stats['Model'] = model_type
            dfs.append(stats)
        combined = pd.concat(dfs, ignore_index=True)
        combined['Model'] = model_type
        blocks.append(combined)

    result = pd.concat(blocks, ignore_index=True)
    # Define column order: exposure, level, stats, Model
    cols = ['exposure', 'level'] + [c for c in result.columns if c not in ['exposure', 'level', 'Model']] + [
        'Model']
    return _reshape_to_wide_summary(result[cols])

