"""

AHI and OSA seveiry is limited measure of sleep breathing disorders. In this script we will explore the best predictor
variable for a sleep questionnaire that can predict  a type of breathing measure from a PSG
"""

import pathlib
from typing import  Optional, List, Union
import numpy as np
import pickle
from config.config import config, metrics_psg, encoding, sections
import pandas  as pd
import statsmodels.api as sm
from library.ml_tabular_data.my_simple_xgb import (plot_xgb_feature_imp,
                                                   create_feature_constraints,
                                                    compute_regression_metrics,
                                                   train_xgb_collect)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer
from typing import List, Tuple
import matplotlib.patches as mpatches
from scipy.stats import shapiro, normaltest, anderson
import scipy.stats as st
from matplotlib.gridspec import GridSpec

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pathlib

def transform_and_plot_ahi_with_severity(df: pd.DataFrame,
                                         column: str = 'ahi',
                                         plot:bool = False,
                                         upper_threshold: int = 160,
                                         output_path: pathlib.Path = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform multiple AHI transformations, classify OSA severity, and plot distributions.

    Applies five different transformations to the specified AHI column and plots
    raw vs. log1p distributions shaded by OSA severity categories:
      1. Winsorization (95th percentile cap)
         - Pros: Limits extreme outliers; preserves data rank
         - Cons: Arbitrary cap threshold; may distort tail
      2. IQR capping (Q3 + 1.5*IQR)
         - Pros: Robust to outliers; data-driven
         - Cons: Can still allow moderate extremes; depends on IQR
      3. Box-Cox transformation (on AHI + 1)
         - Pros: Often yields near-normal distribution; parameter optimized
         - Cons: Requires strictly positive data; shift needed for zeros
      4. Arctan scaling
         - Pros: Smoothly compresses large values; bounded output
         - Cons: Nonlinear distortion; less interpretable scale
      5. Rank-based normalization
         - Pros: Uniform, outlier-robust distribution
         - Cons: Loses original metric meaning; only ranks matter

    Each method's raw and log1p-transformed histograms are shown side by side
    with shared severity legend.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the AHI column.
    column : str, default f'{column}'
        Name of the column to transform.
    upper_threshold : float, default 100
        Scale for the arctan transformation.

    Returns
    -------
    df_t : pandas.DataFrame
        Copy of `df` enriched with new transformed columns:
        - ahi_winsorized, ahi_log1p_winsor
        - ahi_iqr_scaled, ahi_log1p_iqr
        - ahi_boxcox, ahi_log1p_boxcox
        - ahi_arctan_scaled, ahi_log1p_arctan
        - ahi_rank, ahi_log1p_rank
    fig : matplotlib.figure.Figure
        Figure object with the grid of histograms.
    """
    # Work on a copy to preserve original DataFrame
    df_t = df[['id', column]].copy()
    raw_mean, raw_max, raw_min = df_t[column].mean(), df_t[column].max(), df_t[column].min()

    n_samples_raw_target = df_t.shape[0]
    # 0. logp transform
    df_t[f'{column}_log1p'] = np.log1p(df_t[column].copy())

    # 1. Winsorization: cap values above the 95th percentile
    df_t[f'{column}_winsorized'] = winsorize(df_t[column].copy(), limits=[0, 0.05])
    df_t[f'{column}_log1p_winsor'] = np.log1p(df_t[f'{column}_winsorized'])

    # 2. IQR capping: cap values above Q3 + 1.5*IQR
    Q1, Q3 = df_t[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df_t[f'{column}_iqr_scaled'] = df_t[column].copy().clip(upper=upper_bound)
    df_t[f'{column}_log1p_iqr'] = np.log1p(df_t[f'{column}_iqr_scaled'])

    # 3. Box-Cox transform: apply to AHI + 1 for positivity
    pt_box = PowerTransformer(method='box-cox')
    vals = df_t[column] + 1
    df_t[f'{column}_boxcox'] = pt_box.fit_transform(vals.fillna(vals.median()).values.reshape(-1, 1))[:, 0]
    df_t[f'{column}_log1p_boxcox'] = np.log1p(df_t[f'{column}_boxcox'])

    # 4. Arctan scaling: compress tails smoothly
    df_t[f'{column}_arctan_scaled'] = np.arctan(df_t[column] / upper_threshold) * upper_threshold
    df_t[f'{column}_log1p_arctan'] = np.log1p(df_t[f'{column}_arctan_scaled'].copy())

    # 5. Rank normalization: convert to percentile ranks
    df_t[f'{column}_rank'] = df_t[column].copy().rank(pct=True)
    df_t[f'{column}_log1p_rank'] = np.log1p(df_t[f'{column}_rank'].copy())

    assert all([raw_mean == df_t[column].mean(),  raw_max == df_t[column].max(), raw_min == df_t[column].min()])

    # Severity thresholds and colors
    raw_thresh = np.array([5, 15, 30])
    sev_colors = ['#ffe5cc', '#ffcc99', '#ff9999', '#ff6666']
    sev_labels = ['Normal (<5)', 'Mild (5–15)', 'Moderate (15–30)', 'Severe (>=30)']

    # Precompute threshold positions for shading
    thr_map = {
        'raw': {
            'raw': raw_thresh,
            'log': np.log1p(raw_thresh)
        },
        'winsor': {
            'raw': winsorize(raw_thresh.copy(), limits=[0, 0.05]),
            'log': np.log1p(winsorize(raw_thresh.copy(), limits=[0, 0.05]))
        },
        'iqr': {
            'raw': np.clip(raw_thresh, None, upper_bound),
            'log': np.log1p(np.clip(raw_thresh, None, upper_bound))
        },
        'boxcox': {
            'raw': pt_box.transform((raw_thresh + 1).reshape(-1, 1))[:, 0],
            'log': np.log1p(pt_box.transform((raw_thresh + 1).reshape(-1, 1))[:, 0])
        },
        'arctan': {
            'raw': np.arctan(raw_thresh / upper_threshold) * upper_threshold,
            'log': np.log1p(np.arctan(raw_thresh / upper_threshold) * upper_threshold)
        },
        'rank': {
            'raw': pd.Series(raw_thresh).rank(pct=True).values,
            'log': np.log1p(pd.Series(raw_thresh).rank(pct=True).values)
        }
    }

    # List of (raw_col, key, log_col) for plotting
    methods = [
        (column, 'raw', f'{column}_log1p'),
        (f'{column}_winsorized', 'winsor', f'{column}_log1p_winsor'),
        (f'{column}_iqr_scaled', 'iqr', f'{column}_log1p_iqr'),
        (f'{column}_boxcox', 'boxcox', f'{column}_log1p_boxcox'),
        (f'{column}_arctan_scaled', 'arctan', f'{column}_log1p_arctan'),
        (f'{column}_rank', 'rank', f'{column}_log1p_rank')
    ]

    # Arrange 2 methods per row => 3 rows, 4 columns (raw, log) pairs
    n_methods = len(methods)
    methods_per_row = 2
    n_rows = int(np.ceil(n_methods / methods_per_row))
    fig, axes = plt.subplots(n_rows,
                             methods_per_row * 2,
                             figsize=(26, 4 * n_rows),
                             squeeze=False)
    axes = axes.reshape(n_rows, methods_per_row * 2)

    # labels title
    title_map = {
        # raw pair
        column: f'{column}',
        f'{column}_log1p': f'Log₁₊',

        # Winsorization
        f'{column}_winsorized': f'Winsorized (95th percentile cap)',
        f'{column}_log1p_winsor': 'Log₁₊ of Winsorized AHI',

        # IQR capping
        f'{column}_iqr_scaled': f'IQR-Capped (Q₃ + 1.5 · IQR)',
        f'{column}_log1p_iqr': 'Log₁₊ of IQR-Capped',

        # Box-Cox
        f'{column}_boxcox': 'Box-Cox (+1 shift)',
        f'{column}_log1p_boxcox': 'Log₁₊ of Box-Cox',

        # Arctan scaling
        f'{column}_arctan_scaled': 'Arctan-Scaled',
        f'{column}_log1p_arctan': 'Log₁ Arctan-Scaled',

        # Rank
        f'{column}_rank': f'Percentile Rank',
        f'{column}_log1p_rank': 'Log₁₊ Percentile Rank'
    }

    used_axes = []
    # get a color palette with at least as many colors as you have methods
    palette = sns.color_palette("bright")
    for idx, (raw_col, key, log_col) in enumerate(methods):
        # pick a unique color for this pair
        pair_color = palette[idx % len(palette)]

        row = idx // methods_per_row
        col_start = (idx % methods_per_row) * 2
        ax_raw = axes[row, col_start]
        ax_log = axes[row, col_start + 1]
        used_axes.extend([ax_raw, ax_log])

        # 2) (optional) color the histogram bars to match
        ax_raw.hist(df_t[raw_col].dropna(), bins=30, color=pair_color, alpha=0.7)
        ax_log.hist(df_t[log_col].dropna(), bins=30, color=pair_color, alpha=0.7)

        # Shade + plot raw
        bounds_raw = [0, *thr_map[key]['raw'], df_t[raw_col].max()]
        for j in range(4):
            ax_raw.axvspan(bounds_raw[j], bounds_raw[j + 1], color=sev_colors[j], alpha=0.3)
        ax_raw.hist(df_t[raw_col].dropna(), bins=30, color='gray', alpha=0.8)
        ax_raw.set_title(title_map[raw_col])
        # Shade + plot log
        bounds_log = [0, *thr_map[key]['log'], df_t[log_col].max()]
        for j in range(4):
            ax_log.axvspan(bounds_log[j], bounds_log[j + 1], color=sev_colors[j], alpha=0.3)
        ax_log.hist(df_t[log_col].dropna(), bins=30, color='gray', alpha=0.8)
        ax_log.set_title(title_map[log_col])

    fig.text(0.02, 0.98,
             column.upper().replace("_"," ").replace("-", " "),
             transform=fig.transFigure,
             ha='left',
             va='top',
             color='black',
             fontsize=26)

    # Turn off any unused subplots
    for ax in axes.flatten():
        if ax not in used_axes:
            ax.axis('off')

    # Shared legend
    legend_patches = [mpatches.Patch(color=sev_colors[i], alpha=0.3, label=sev_labels[i]) for i in range(4)]
    fig.legend(handles=legend_patches,
               loc='lower center' ,#'upper center',
               ncol=4,
               title='OSA Severity',
               # fontsize=12,
               # title_fontsize=14
               )

    # fig.tight_layout(rect=[0, 0, 0, 0.90])  # rect=[left, bottom, right, top]
    plt.tight_layout(pad=3.0, w_pad=1.5, h_pad=1.0)
    if output_path:
        plt.savefig(output_path.joinpath(f'dist_{column}'), dpi=300)
    if plot:
        plt.show()
    plt.close()

    # ─── statistical summaries (per transformation) ───────────────────────────
    # 1) classify into OSA severity and count samples
    severity_bins = [-np.inf, 5, 15, 30, np.inf]
    severity_labels = ['Normal', 'Mild', 'Moderate', 'Severe']
    df_t['severity'] = pd.cut(df_t[column],
                              bins=severity_bins,
                              labels=severity_labels)
    cnts = df_t['severity'].value_counts().reindex(severity_labels).to_dict()

    # 2) prepare list of all transforms to test
    transforms = {
        'raw': column,
        'winsorized': f'{column}_winsorized',
        'iqr_capped': f'{column}_iqr_scaled',
        'boxcox': f'{column}_boxcox',
        'arctan': f'{column}_arctan_scaled',
        'rank': f'{column}_rank',
        'log_raw': f'{column}_log1p',
        'log_winsorized': f'{column}_log1p_winsor',
        'log_iqr': f'{column}_log1p_iqr',
        'log_boxcox': f'{column}_log1p_boxcox',
        'log_arctan': f'{column}_log1p_arctan',
        'log_rank': f'{column}_log1p_rank',
    }

    dist_candidates = [
        ('normal', st.norm),
        ('lognorm', st.lognorm),
        ('gamma', st.gamma),
    ]

    dist_results = []
    for tname, col_name in transforms.items():
        y = df_t[col_name].dropna().values
        n = len(y)
        # subsample for Shapiro if too large
        if n > 5000:
            y_shap = np.random.choice(y, 5000, replace=False)
        else:
            y_shap = y

        # normality tests
        sw_stat, sw_p = shapiro(y_shap)
        dg_stat, dg_p = normaltest(y)
        ad_res = anderson(y, dist='norm')

        # fit candidate distributions
        for dist_name, dist in dist_candidates:
            if dist_name == 'lognorm' and np.any(y <= 0):
                continue
            data = y
            if len(data) < 3:
                continue
            params = dist.fit(data)
            ll = dist.logpdf(data, *params).sum()
            k = len(params)
            aic = 2 * k - 2 * ll
            bic = np.log(len(data)) * k - 2 * ll

            dist_results.append({
                'transform': tname,
                'column': col_name,
                'distribution': dist_name,
                'n_samples': len(data),
                'AIC': aic,
                'BIC': bic,
                'Shapiro-W': sw_stat,
                'Shapiro-p': sw_p,
                'DAgostino-K²': dg_stat,
                'DAgostino-p': dg_p,
                'Anderson-stat': ad_res.statistic,
                'mean': data.mean(),
                'median': np.median(data),
                'std': data.std(),
                'variance': data.var(),
                'IQR': np.percentile(data, 75) - np.percentile(data, 25),
                'range': data.max() - data.min(),
                'cv': data.std() / data.mean() if data.mean() else np.nan,  # Coefficient of Variation
                'mad_mean': np.mean(np.abs(data - data.mean())),
                'mad_med': np.median(np.abs(data - np.median(data))),
                # 'p90_p10': np.percentile(data, 90) - np.percentile(data, 10),
                # 'gini_md': (np.abs(data[:, None] - data[None, :]).sum()) / (len(data) * (len(data) - 1)),
                'sev_Normal': cnts['Normal'],
                'sev_Mild': cnts['Mild'],
                'sev_Moderate': cnts['Moderate'],
                'sev_Severe': cnts['Severe'],
            })

    df_stats = pd.DataFrame(dist_results)
    # ─── finalize & return two objects ────────────────────────────────────────
    if output_path:
        df_stats.to_csv(output_path.joinpath(f'distribution_stat_test_{target}'), index=False)

    return df_t, df_stats




def fit_poisson_apnea_model(
        df: pd.DataFrame,
        apnea_count_var: str,
        covariates: list = ['age', 'race', 'gender']
):
    """
    Fits a Poisson regression model for obstructive apnea count.

    Parameters:
    - apnea_count_var: Name of column with apnea event count
    - covariates: List of predictor variables

    Returns statsmodels GLM results object
    """
    cols = [apnea_count_var] + covariates
    df_model = df[cols].dropna().copy()

    # Convert outcome to non-negative integer
    df_model[apnea_count_var] = pd.to_numeric(df_model[apnea_count_var],
                                              errors='coerce').astype(int)

    # Filter valid counts (>=0)
    df_model = df_model[df_model[apnea_count_var] >= 0]

    # Separate covariates
    numeric_vars = [var for var in covariates
                    if pd.api.types.is_numeric_dtype(df_model[var])]
    categorical_vars = [var for var in covariates
                        if not pd.api.types.is_numeric_dtype(df_model[var])]

    # Create design matrix
    X = sm.add_constant(df_model[numeric_vars]) if numeric_vars else pd.DataFrame()
    if categorical_vars:
        dummies = pd.get_dummies(df_model[categorical_vars], drop_first=True)
        X = pd.concat([X, dummies], axis=1)

    # Fit model
    poisson_model = sm.GLM(df_model[apnea_count_var],
                           X.astype(float),
                           family=sm.families.Poisson())

    return poisson_model.fit()


def compare_regression_results(results_dict):
    """
    Create comparison DataFrame from collected results.

    Parameters:
    - results_dict: Output from collect_regression_results

    Returns:
    - DataFrame with model comparison metrics
    """
    comparison_data = []

    for target, result in results_dict.items():
        if 'error' in result:
            row = {'target': target, 'error': result['error']}
            comparison_data.append(row)
            continue

        try:
            row = {
                'target': target,
                'n_samples': result['n_samples'],
                # Cross-validation metrics
                'cv_r2_mean': result['cv_metrics']['r2_mean'],
                'cv_r2_std': result['cv_metrics']['r2_std'],
                'cv_rmse_mean': result['cv_metrics']['rmse_mean'],
                'cv_rmse_std': result['cv_metrics']['rmse_std'],
                # Test metrics
                'test_r2': result['test_metrics']['r2'],
                'test_rmse': result['test_metrics']['rmse'],
                # Error metrics
                'mae_mean': result['cv_metrics']['mae_mean'],
                'mae_std': result['cv_metrics']['mae_std'],
                'maxerr_mean': result['cv_metrics']['maxerr_mean']
            }
            comparison_data.append(row)

        except KeyError as e:
            print(f"Missing metric {str(e)} for {target}")
            comparison_data.append({'target': target, 'error': f"Missing metric {str(e)}"})

    # Create and sort DataFrame
    df = pd.DataFrame(comparison_data)
    if not df.empty:
        df = df.sort_values('test_rmse', ascending=True)

    return df



def plot_true_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    textstr: str,
    hue: Optional[np.ndarray] = None,
    output_path:pathlib.Path = None,
):

    df = pd.DataFrame({'True': y_true, 'Pred': y_pred})
    if hue is not None:
        hue_order = ['Normal', 'Mild', 'Moderate', 'Severe']
        df['Hue'] = pd.Categorical(hue, categories=hue_order, ordered=True)

    # compute ±1 STD on the residuals
    residuals = df['True'] - df['Pred']
    std_dev = residuals.std()

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(
        data=df, x='True', y='Pred',
        hue='Hue' if hue is not None else None,
        palette='Reds',
        alpha=0.7,
        s=60,
        ax=ax
    )

    # determine data‐driven limits
    t_min, t_max = df['True'].min(), df['True'].max()
    p_min, p_max = df['Pred'].min(), df['Pred'].max()

    # only span the line & band over the overlap of true & pred ranges
    line_min = max(t_min, p_min)
    line_max = min(t_max, p_max)

    # 1:1 line
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        '--', color='gray', linewidth=2,
        label='Perfect'
    )

    # ±1 STD band
    ax.fill_between(
        [line_min, line_max],
        [line_min - std_dev, line_max - std_dev],
        [line_min + std_dev, line_max + std_dev],
        color='orange', alpha=0.2,
        label='±1 STD'
    )

    # axis limits tied to the **actual** data
    ax.set_xlim(t_min, t_max)
    ax.set_ylim(p_min, p_max)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel('True Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)

    # move legend out of the way
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.3)

    # metrics textbox
    ax.text(
        0.02, 0.98, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )

    sns.despine()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()



def extract_summary_metrics(df_summary: pd.DataFrame, grp: str) -> dict:
    """Return a dict with r2_adj_mean/std and rmse_mean/std for a given group."""
    row = df_summary.loc[df_summary['group'] == grp].iloc[0]
    return {
        'r2_adj_mean': row['r2_adj_mean'],
        'r2_adj_std':  row['r2_adj_std'],
        'rmse_mean':   row['rmse_mean'],
        'rmse_std':    row['rmse_std']
    }



def plot_metrics_by_target(df: pd.DataFrame, output_path: pathlib.Path = None):
    """
    Plots R², Adj. R², and RMSE per target variable from the model comparison DataFrame.

    Parameters:
    - df: DataFrame with columns ['group', 'r2_mean', 'r2_adj_mean', 'rmse_mean', ...] and 'target'.
    - output_path: Directory to save the plots (optional).
    """
    metric_map = {
        'r2_mean': 'R²',
        'r2_adj_mean': 'Adj. R²',
        'rmse_mean': 'RMSE'
    }

    std_map = {
        'r2_mean': 'r2_std',
        'r2_adj_mean': 'r2_adj_std',
        'rmse_mean': 'rmse_std'
    }

    targets = df['target'].unique()
    groups = ['train', 'val', 'test']
    n_targets = len(targets)

    fig = plt.figure(figsize=(12, 3.2 * n_targets))
    gs = GridSpec(n_targets, 3, figure=fig)

    all_handles = []
    all_labels = []

    for row_idx, target in enumerate(targets):
        df_target = df[df['target'] == target]

        for col_idx, metric_key in enumerate(metric_map.keys()):
            ax = fig.add_subplot(gs[row_idx, col_idx])

            for i, group in enumerate(groups):
                group_data = df_target[df_target['group'] == group]
                if not group_data.empty:
                    y = group_data[metric_key].values[0]
                    yerr = group_data[std_map[metric_key]].values[0] if std_map[metric_key] in group_data else None

                    if group == 'train':
                        h = ax.errorbar(i, y, yerr=yerr, fmt='o', capsize=4, color='blue', label='Train')
                    elif group == 'val':
                        h = ax.errorbar(i, y, yerr=yerr, fmt='o', capsize=4, color='orange', label='Validation')
                    else:
                        h = ax.scatter(i, y, color='black', marker='X', s=60, label='Test')

                    if row_idx == 0 and col_idx == 0:  # only collect handles once
                        handle, label = ax.get_legend_handles_labels()
                        all_handles.extend(handle)
                        all_labels.extend(label)

            ax.set_title(f"{metric_map[metric_key]} – {target}", fontsize=10)
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels(groups, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)

    # Add single legend at the bottom center
    unique_legend = dict(zip(all_labels, all_handles))
    fig.legend(unique_legend.values(), unique_legend.keys(),
               loc='lower center', ncol=len(unique_legend), fontsize=10, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()

if __name__ == '__main__':
    # %% Input data
    PCA_reduced_dim = True

    # %% Data to read
    if PCA_reduced_dim:
        df = pd.read_csv(config.get('data')['pp_data']['pca_reduced'], low_memory=False)
    else:
        df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    print(f'Dataet dimensions {df.shape}')
    # %% output directory
    path_dir = config.get('results')['best_target']

    path_output = path_dir.joinpath('models')

    # %% Create output directory if it doesn't exist'
    path_output.mkdir(parents=True, exist_ok=True)

    # %% respiratory events
    target = 'ahi'  # same as resp-ahi_no_reras-total
    # events = metrics_psg.get('resp_events')['raw_events']
    # event = events[0]
    # pos = 'total'
    t_col = 'resp-position-total'  # metrics_psg.get('resp_events')['position_keywords']
    # cnt_col = f"resp-{event}-{pos}"  # 'resp-oa-total'
    # target_candidates = [f"resp-{event}-{pos}" for event in events]
    # target_candidates = [tar for tar in target_candidates if tar in df.columns]
    #
    df['sleep_hours'] = df[t_col] / 60.0
    print('Target candidates:)')
    # use only the respiratory metrics that compose the AHI = (OA+CA+MA+HYP)/T
    target_candidates = ['ahi',
                        'resp-oa-total',
                         'resp-ca-total',
                         'resp-ma-total',
                         'resp-hi_hypopneas_only-total',
                         'resp-ri_rera_only-total'
                         'ahi']

    # %% Evaluate teh distribution of each target
    if not path_dir.joinpath('dist_targets_stats.csv').exists():
        df_stats = pd.DataFrame()
        df_t_all = pd.DataFrame()
        for tar_can in target_candidates:
            df_t, df_stat = transform_and_plot_ahi_with_severity(df=df,
                                    column=tar_can,
                                    plot=False,
                                    upper_threshold=160,
                                    output_path=path_dir)
            df_stat['target'] = tar_can
            df_t.drop(columns=['severity'], inplace=True)
            df_stats = pd.concat([df_stats, df_stat],
                                 axis=0,
                                 ignore_index=True)
            df_t_all = pd.concat([df_t_all, df_t],
                                 axis=1,
                                 ignore_index=False)
        df_stats.to_csv(path_dir.joinpath('dist_targets_stats.csv'), index=False)
        # Keep only the first 'id' column and drop duplicates
        df_t_all = df_t_all.loc[:, ~df_t_all.columns.duplicated()]
        df_t_all.to_csv(path_dir.joinpath('dist_targets_all_transformations.csv'), index=False)
    else:
        df_stats = pd.read_csv(path_dir.joinpath('dist_targets_stats.csv'))
        df_t_all = pd.read_csv(path_dir.joinpath('dist_targets_all_transformations.csv'))

    if not all(col in df.columns for col in df_t_all.columns):
        print(f'Not all targets transforms are in the data, we need to append them')
        # Remove overlapping columns from df_t_all (excluding 'id')
        columns_to_drop = [col for col in df_t_all.columns if col in df.columns and col != 'id']
        df_t_all = df_t_all.drop(columns=columns_to_drop)

        # Now perform the merge
        df = pd.merge(df,
                      df_t_all,
                      on='id',
                      how='left')
        if PCA_reduced_dim:
            df.to_csv(config.get('data')['pp_data']['pca_reduced_transf'], index=False)
        else:
            df.to_csv(config.get('data')['pp_data']['q_resp_trasnf'], index=False)

    # define your recommended transforms based on the distribution tests  ---
    transform_recs = {
        'ahi': 'log1p',
        "resp-oa-total": "log1p",  # best Shapiro-p was on log1p of raw
        "resp-ca-total": "log1p",  # best Shapiro-p was on log1p of Box–Cox
        "resp-ma-total": "rank",  # best Shapiro-p was on rank/percentile
        "resp-hi_hypopneas_only-total": "log1p",  # best Shapiro-p was on log1p of raw
        # if you ever go back to AHI itself, you could add "ahi": "log1p"
    }
     # %%
    section_features = [sect for sect in sections if not sect in ['resp']]
    # 1. Define your sections and columns
    features = [col for col in df.columns
                    if any(col.startswith(alias) for alias in section_features)]
    features.append('sleep_hours')

    # 2) To get a dict mapping each alias → its matching columns:
    cols_by_alias = {
        alias: [col for col in df.columns if col.startswith(alias)]
        for alias in sections
    }
    cols_count_by_sect = {key: len(val) for key, val in cols_by_alias.items()}
    print(f'columns per section: {cols_count_by_sect} \n')

    # extract each column’s “first prefix” (including the underscore)
    prefix = df.columns.to_series().str.extract(r'^([^_]+_)', expand=False)
    col_dict = {
        p: df.columns[prefix == p].tolist()
        for p in prefix.dropna().unique()
    }
    # 2. Create the constraint groups
    interaction_constraints, missing = create_feature_constraints(sections=section_features,
                                                         df=df)
    stratify_col = 'osa_four'
    # %%
    # we want also the post sleep, an ml that collects before and afet the sleep
    # features = [feat for feat in [*encoding.keys()] if not (feat.startswith('postsleep')) and (feat in df.columns)]
    if path_output.joinpath('results_dict_all_targets.pkl').is_file():
        results_dict = pickle.load(open(path_output.joinpath('results_dict_all_targets.pkl'), 'rb'))
    else:
        results_dict = {}
        df_model_comparisons = pd.DataFrame()
        for idx, target in enumerate(target_candidates):
            target = target + '_' +transform_recs.get(target)
            # df[target]
            print(f"********Fitting {target}   {idx/len(target_candidates)}********\n")
            df_model = df.loc[~df[target].isna(), features+[target] + [stratify_col]].copy()
            X = df_model[features].copy()
            y = df_model[target].copy()
            output_dir_current = path_output.joinpath(f'{target}')
            fi_df, preds_df, best_params, final_bst = train_xgb_collect(
                data=df_model,
                # in_params={'interaction_constraints': interaction_constraints},
                in_params={
                            'objective': 'reg:squarederror',
                           # 'tweedie_variance_power': 1.5,
                           'eval_metric': "rmse",

                           # 'objective': 'reg:tweedie',
                           # 'tweedie_variance_power': 1.3,
                           # 'eval_metric': "tweedie-nloglik@1.3",
                           # 'gamma': 0.1,
                           'max_bin': 256,
                           'num_parallel_tree': 10,
                           'early_stopping_rounds':100,
                            'num_boost_round':1000
                            # 'updater': 'coord_descent',
                           # 'feature_selector': 'greedy'
                           },
                feature_cols=features,
                target_col=target,
                optimization=True,
                val_size=0.3,
                test_size=0.1,
                n_trials=15,
                cv_folds=False,
                use_gpu=True,
                stratify_col='osa_four',
                model_path=output_dir_current,
                show_loss_curve=True,
                resample=False,
                random_state=42,
            )
            df_metrics, df_metrics_summary = compute_regression_metrics(preds_df=preds_df,
                                                                        n_feats=X.shape[1],
                                                                        output_dir=output_dir_current)
            plot_xgb_feature_imp(fi_df=fi_df,
                                 top_n=20,
                                 ncol=2,
                                 output_path=output_dir_current)

            results_dict[target] = {
                'model': final_bst,
                'best_params': best_params,
                'preds_df': preds_df,
                'metrics': df_metrics,
                'metrics_summary': df_metrics_summary,
                'n_samples': len(df_model)
            }

            # 2. Generate comparison DataFrame
            comparison_df = df_metrics_summary.copy()
            comparison_df['target'] = target
            df_model_comparisons = pd.concat([df_model_comparisons, comparison_df], axis=0)

        #  save the collection
        with open(path_output.joinpath('results_dict_all_targets.pkl'), 'wb') as f:
            pickle.dump(results_dict, f)
        df_model_comparisons.to_csv(path_output.joinpath('model_comparisons.csv'), index=False)

    # %% plot the data summmary metrics

    if path_output.joinpath('model_comparisons.csv').is_file():
        df_model_comparisons = pd.read_csv(path_output.joinpath('model_comparisons.csv'))

        plot_metrics_by_target(df=df_model_comparisons.copy(),
                               output_path=path_output.joinpath('metrics_by_target_log1p.png'))


    #%% 4. Generate plots for each target
    for target, res in results_dict.items():
        # target = [*results_dict.keys()][0]
        # res = results_dict[target]
        output_dir_current = path_output.joinpath(f'{target}')

        preds = res['preds_df']
        # only keep rows where both true & pred exist
        mask = preds['test_true'].notna() & preds['test_pred'].notna()
        y_true = preds.loc[mask, 'test_true']
        y_pred = preds.loc[mask, 'test_pred']

        # hue from original df via the test indices
        hue = df.loc[y_true.index, stratify_col]

        # build train‐summary textbox
        train_metrics = extract_summary_metrics(res['metrics_summary'], 'train')

        if train_metrics['r2_adj_std'] or train_metrics['rmse_std']:
            textstr = (
                f"Train:\n"
                f"R² (adj.): {train_metrics['r2_adj_mean']:.3f} ± {train_metrics['r2_adj_std']:.3f}\n"
                f"RMSE:      {train_metrics['rmse_mean']:.3f} ± {train_metrics['rmse_std']:.3f}"
            )
        else:
            textstr = (
                f"Train:\n"
                f"R² (adj.): {train_metrics['r2_adj_mean']:.3f}\n"
                f"RMSE:      {train_metrics['rmse_mean']:.3f}"
            )

        # build title with test metrics
        test_row = res['metrics'].query("group=='test'").iloc[0]
        title = (
            f"{target.replace('_', ' ').replace('_', ' ').title()}\n"
            f"Test R² (adj.): {test_row['r2_adj']:.3f} | RMSE: {test_row['rmse']:.3f}"
        )

        # call your plotting helper
        plot_true_vs_predicted(
            y_true=np.expm1(y_true.values),
            y_pred=np.expm1(y_pred.values),
            title=title,
            textstr=textstr,
            hue=hue.values,
            output_path=output_dir_current.joinpath(f'{target}_true_vs_pred.png'),
        )



    # %%
    model_results = fit_poisson_apnea_model(
        df=df,
        apnea_count_var='resp-oa-total',
        covariates=['age', 'bmi', 'gender', ]
    )
    model_results.summary()






