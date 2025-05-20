"""
Multicollinearity and Dimensionality Reduction Analysis Toolkit

This script provides a comprehensive set of tools for analyzing multicollinearity,
performing dimensionality reduction (PCA and MCA), and evaluating their impact on
predictive modeling using XGBoost. It includes functions for:

1. **Correlation Analysis**:
   - Visualizes Spearman correlation matrices as heatmaps.
   - Generates histograms of correlation coefficients with statistical summaries.
   - Outputs sorted correlation tables for feature pairs.

2. **Variance Inflation Factor (VIF)**:
   - Computes VIF to quantify multicollinearity.
   - Plots VIF values with thresholds for negligible, moderate, and high multicollinearity.

3. **Multiple Correspondence Analysis (MCA)**:
   - Fits MCA models for categorical data.
   - Returns scores, category coordinates, and inertia explained.
   - Visualizes MCA results with scatter plots.

4. **Principal Component Analysis (PCA) with Whitening**:
   - Performs PCA followed by whitening to produce uncorrelated components.
   - Injects whitened scores into the input DataFrame.
   - Returns explained variance ratios and component loadings.

5. **Visualization Tools**:
   - Generates scree plots for explained variance.
   - Creates heatmaps for component loadings and coordinates.
   - Plots VIF bar charts with wrapped labels.

6. **Model Evaluation**:
   - Trains XGBoost models on raw features, PCA components, or both.
   - Compares predictive performance (R², Adjusted R², RMSE) across models.
   - Visualizes performance metrics with error bars for training and test sets.

The script is designed to handle tabular data, assess feature relationships, reduce
dimensionality, and evaluate the impact of these techniques on regression tasks,
with a focus on interpretability and robust visualization.

Dependencies:
- numpy, pandas, matplotlib, seaborn, sklearn, xgboost, statsmodels, prince, textwrap, pathlib

Usage:
- Configure input data and parameters via `config.config`.
- Specify output directories for saving plots and tables.
- Call individual functions for specific analyses or combine them for a full pipeline.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from config.config import config, encoding
from typing import Sequence, Optional, Tuple, Union
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import List, Optional, Dict
from statsmodels.stats.outliers_influence import variance_inflation_factor
import textwrap
from library.ml_tabular_data.my_simple_xgb import train_xgb_collect, compute_regression_metrics
import pathlib
import pickle
import seaborn as sns
from matplotlib.gridspec import GridSpec


# %% Correlations
def plot_correlation_matrix(
        df: pd.DataFrame,
        title: str,
        figsize: Tuple[int, int] = (8, 6),
        cmap: str = 'coolwarm',
        wrap_width: int = 15,
        absolute: bool = True,
        output_dir:pathlib.Path = None
) -> pd.DataFrame:
    """
    plot_correlation_matrix(df, title, figsize=(8,6), cmap='coolwarm', wrap_width=15) -> None
    Plots a Pearson correlation heatmap for multicollinearity assessment,
    wrapping long feature names on both axes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of numeric features.
    title : str
        Title for the heatmap.
    figsize : Tuple[int,int], default=(8,6)
        Figure size in inches.
    cmap : str, default='coolwarm'
        Matplotlib colormap name.
    wrap_width : int, default=15
        Maximum line width for wrapped labels.
    """

    corr = df.corr(method='spearman')

    # Auto‐size if not specified and you have a lot of features
    n = corr.shape[0]
    if figsize is None:
        side = max(8, n * 0.25)
        figsize = (side, side)

    # Wrap long labels
    wrapped_cols = [textwrap.fill(col, width=wrap_width) for col in corr.columns]
    wrapped_idx  = [textwrap.fill(idx, width=wrap_width) for idx in corr.index]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values,
                   aspect='auto',
                   interpolation='nearest',
                   cmap=cmap,
                   # vmin=threshold,  # start gradient at threshold
                   # vmax=1.0  # assume maximum possible correlation is 1
                   )
    # Add annotations to each cell
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f'{corr.values[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if abs(corr.values[i, j]) > 0.5 else 'black',  # Adjust text color for readability
                    fontsize=11)
    ax.set_xticks(np.arange(len(wrapped_cols)))
    ax.set_xticklabels(wrapped_cols, rotation=90, ha='center')
    ax.set_yticks(np.arange(len(wrapped_idx)))
    ax.set_yticklabels(wrapped_idx, va='center')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='Pearson correlation')
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / f'{title}.png', dpi=300)
    plt.show()
    plt.close()

    # Table of the correlations sorted by absolute value, igorind the diagonal
    # 2) Take only upper triangle, k=1 to exclude diagonal
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    corr_entries = corr.where(mask).stack().reset_index()
    corr_entries.columns = ['feature_1', 'feature_2', 'correlation']
    # 3) Sort
    if absolute:
        corr_entries['abs_corr'] = corr_entries['correlation'].abs()
        corr_entries = corr_entries.sort_values('abs_corr', ascending=False).drop(columns='abs_corr')
    else:
        corr_entries = corr_entries.sort_values('correlation', ascending=False)
    df_corr = corr_entries.reset_index(drop=True)
    df_corr['feature_1'] = df_corr['feature_1'].apply(lambda x: x.replace('_', ' ').title())
    df_corr['feature_2'] = df_corr['feature_2'].apply(lambda x: x.replace('_', ' ').title())
    df_corr = df_corr[df_corr['correlation'] > 0]
    if output_dir:
        df_corr.to_csv(output_dir / f'{title}_table.csv', index=False)

    # histogram plot fo the correlation distribution
    corr_values = df_corr['correlation']
    print(f'Histogram total number of values: {corr_values.shape} \n')
    mu = corr_values.mean()
    sigma = corr_values.std()
    plt.figure(figsize=(8, 5))
    plt.hist(corr_values, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mu, color='red', linestyle='--', linewidth=2, label='Mean')
    plt.axvline(mu + sigma, color='orange', linestyle='--', linewidth=2, label='+1 Std Dev')
    plt.axvline(mu - sigma, color='orange', linestyle='--', linewidth=2, label='-1 Std Dev')
    plt.axvline(mu + 2 * sigma, color='goldenrod', linestyle='--', linewidth=2, label='+2 Std Dev')
    plt.axvline(mu - 2 * sigma, color='goldenrod', linestyle='--', linewidth=2, label='-2 Std Dev')
    plt.xlim([0, np.max(corr_values)+0.2])
    plt.title('Histogram of Correlation Coefficients')
    plt.xlabel('Correlation')
    plt.grid(alpha=0.7, linestyle='--')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / f'{title}_histogram.png', dpi=300)
    plt.show()
    print(f'Histogram values: \n'
          f'\tmu:{mu:.2f} 1SD:{mu + sigma:.2f} 2SD:{mu + 2* sigma:.2f}')
    # proportion of variable‐pairs with at least medium (|φ| ≥ 0.3)
    prop_medium = (df_corr['correlation'].abs() >= 0.3).mean()
    # proportion with large (|φ| ≥ 0.5)
    prop_large = (df_corr['correlation'].abs() >= 0.5).mean()
    print(f"{prop_medium:.1%} of pairs have |φ| ≥ 0.3 (medium+)")
    print(f"{prop_large:.1%} of pairs have |φ| ≥ 0.5 (large)")

    return df_corr

# %% VIF analysis
def compute_vif(
        df: pd.DataFrame,
        output_dir:pathlib.Path = None
) -> pd.Series:
    """
    compute_vif(df) -> pd.Series
    Computes Variance Inflation Factor (VIF) for each column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of numeric features.

    Returns
    -------
    pd.Series
        VIF values indexed by column name.
    """
    X = df.values
    vif_values = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif_df = pd.DataFrame({
        "variable": df.columns,
        "VIF": vif_values
    }).set_index("variable")
    if output_dir:
        vif_df.to_csv(output_dir / 'vif.csv', index=True)
    return pd.Series(vif_values, index=df.columns)

#%% MCA model
def run_mca(X_df,
            n_components=3,
            n_iter=3,
            random_state=42):
    """
    Fit an MCA model, inject per-sample scores into `df`, and return:
      - mca_scores: DataFrame (n_samples × n_components) of row coordinates
      - col_coords: DataFrame (n_categories × n_components) of category coords
      - perc_inertia:  ndarray (% inertia per component)

    Parameters
    ----------
    X_df : pd.DataFrame
        The binary input data for MCA (after filtering).
    df : pd.DataFrame
        The original DataFrame into which we’ll inject the MCA scores.
    prefix : str
        Column-name prefix for injected score columns.
    n_components : int
        Number of MCA dimensions.
    n_iter : int
        Number of iterative passes for MCA.
    random_state : int
        RNG seed for reproducibility.

    Returns
    -------
    mca_scores : pd.DataFrame
        Row coordinates (scores) in MCA space.
    col_coords : pd.DataFrame
        Coordinates of each one‑hot category.
    perc_inertia : np.ndarray
        Percentage of inertia explained by each dimension.
    """
    from prince.mca import MCA

    mca = MCA(
        n_components=n_components,
        n_iter=n_iter,
        copy=True,
        check_input=True,
        engine='sklearn',
        random_state=random_state
    ).fit(X_df)

    mca_scores = mca.row_coordinates(X_df)      # observations in MCA space :contentReference[oaicite:0]{index=0}
    mca_scores.columns = [f"mca_comp_{i + 1}" for i in range(n_components)]
    # category-level coords
    col_coords = mca.column_coordinates(X_df)    # category‐level loadings :contentReference[oaicite:1]{index=1}
    # inertia explained
    perc_inertia = mca.percentage_of_variance_   # % inertia per component :contentReference[oaicite:2]{index=2}

    return mca_scores, col_coords, perc_inertia

# %% PCA + Whiten
def run_pca_and_whiten(
    X: np.ndarray,
    prefix: str,
    df: pd.DataFrame,
    max_components: int = 6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Performs PCA → U → whitening → Uw, injects Uw into `df`, and returns:
      - whitened_var: explained_variance_ratio_ of the *whitened* PCA
      - orig_components: the original pca0.components_ (k × p)
      - whiten_components: pw.components_ (k × k, usually less useful for feature loadings)
      - Uw_df: DataFrame of shape (n_samples × k) with your Uw scores
    """
    # 1) Original PCA on X
    pca0 = PCA(n_components=min(max_components, X.shape[1]), svd_solver='full', random_state=0)
    U = pca0.fit_transform(X)                  # (n_samples × k)
    orig_comps = pca0.components_              # (k × p)

    # 2) Whitening PCA on U
    pw = PCA(whiten=True, random_state=0)
    Uw = pw.fit_transform(U)                   # (n_samples × k)
    white_comps = pw.components_               # (k × k)

    # 3) Build Uw DataFrame & inject
    k = Uw.shape[1]
    cols = [f"{prefix}fact_{i+1}" for i in range(k)]
    Uw_df = pd.DataFrame(Uw, columns=cols, index=df.index)
    df[cols] = Uw_df

    return pw.explained_variance_ratio_, orig_comps, white_comps, Uw_df


# %% Visualizations
# mca
def plot_mca_scatter(
    rows: pd.DataFrame,
    cols: pd.DataFrame,
    title: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 6)
) -> None:
    """
    plot_mca_scatter(rows, cols, title, save_path=None, figsize=(6,6)) -> None
    2D scatter of MCA rows vs categories (first two dimensions).

    Parameters
    ----------
    rows : pd.DataFrame
        DataFrame of shape (n_samples, 2) containing row coordinates.
    cols : pd.DataFrame
        DataFrame of shape (n_categories, 2) containing category coordinates.
    title : str
        Plot title.
    save_path : Optional[str], default=None
        File path to save the figure (PNG). If None, figure is not saved.
    figsize : Tuple[int, int], default=(6, 6)
        Figure size in inches (width, height).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(rows.iloc[:, 0], rows.iloc[:, 1],
               s=20, alpha=0.6, label="Rows")
    ax.scatter(cols.iloc[:, 0], cols.iloc[:, 1],
               s=50, marker='D', label="Categories", c='C3')
    ax.axhline(0, color="gray", lw=1)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("MCA Dim 1")
    ax.set_ylabel("MCA Dim 2")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()


# PCA
def plot_scree(
    values: Sequence[float],
    labels: Sequence[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (6, 4)
) -> None:
    """
    plot_scree(values, labels, title, xlabel, ylabel, save_path=None, figsize=(6,4)) -> None
    Generic bar-plot with annotations atop each bar and a line plot overlay for elbow rule.

    Parameters
    ----------
    values : Sequence[float]
        Sequence of numerical proportions (e.g., explained variance ratios).
    labels : Sequence[str]
        Tick labels for each bar.
    title : str
        Plot title.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    save_path : Optional[str], default=None
        File path to save the figure (PNG). If None, figure is not saved.
    figsize : Tuple[int, int], default=(6, 4)
        Figure size in inches (width, height).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(np.arange(len(values)), values, edgecolor='k')
    ax.plot(np.arange(len(values)), values, marker='o', linestyle='-', color='r', label='Elbow')  # Add line plot
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend()  # Show legend for the line plot

    for i, v in enumerate(values):
        ax.text(i, v + 0.01 * max(values), f"{v*100:.1f}%",
                ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300)
    plt.show()


# heatmaps
def plot_heatmap(
    coords: pd.DataFrame,
    title: str,
    figsize: Tuple[int, int] = (8, 4),
    cmap: str = "viridis",
    save_path: pathlib.Path = None,
) -> None:
    """
    plot_heatmap(coords, title, figsize=(8,4), cmap='viridis') -> None
    Heat-map of any coords DataFrame (index × components).

    Parameters
    ----------
    coords : pd.DataFrame
        DataFrame of shape (n_items, n_components) to visualize.
    title : str
        Plot title.
    figsize : Tuple[int, int], default=(8, 4)
        Figure size in inches (width, height).
    cmap : str, default='viridis'
        Matplotlib colormap name.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(coords.values, aspect="auto",
                   interpolation="nearest", cmap=cmap)
    ax.set_ylabel("Index")
    ax.set_xlabel("Component")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Coordinate value")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_component_heatmap(
    components: np.ndarray,
    feature_names: Sequence[str],
    comp_labels: Sequence[str],
    title: str,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = "coolwarm",
    xtick_rotation: float = 45,
    annot_fontsize: float = 8,
    horizontal: bool = False ,
    save_path: pathlib.Path = None,
) -> None:
    """
    Heatmap of feature loadings with optional horizontal orientation.

    Parameters
    ----------
    components : array (n_components, n_features)
    feature_names : list of str
    comp_labels   : list of str
    title         : str
    figsize       : tuple
    cmap          : str
    xtick_rotation: float, degrees for x‐tick labels (0=horizontal)
    annot_fontsize: float, inside‐cell font size
    horizontal    : bool, if True plot components on y and features on x
    """
    # build DataFrame
    df = pd.DataFrame(components.T,
                      index=feature_names,
                      columns=comp_labels)
    if horizontal:
        df = df.T

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(df.values,
                   aspect="auto",
                   interpolation="nearest",
                   cmap=cmap)

    # annotations
    for (i, j), val in np.ndenumerate(df.values):
        ax.text(j, i,
                f"{val:.2f}",
                ha="center", va="center",
                fontsize=annot_fontsize,
                color="white" if abs(val) > 0.5 else "black")

    # ticks
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_xticklabels(df.columns, rotation=xtick_rotation,
                       ha="right" if xtick_rotation else "center")
    ax.set_yticks(np.arange(df.shape[0]))
    ax.set_yticklabels(df.index)

    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# VIF
def plot_vif(
        vif: pd.Series,
        title: str,
        figsize: Tuple[int, int] = (8, 4),
        wrap_width: int = 20,
        output_dir:pathlib.Path = None,
        xticks_rotation: Optional[int] = 90,
) -> None:
    """
    plot_vif(vif, title, figsize=(8,4), wrap_width=20) -> None
    Improved bar-plot of Variance Inflation Factor (VIF) for each feature:
     - Sorted descending
     - Wrapped tick labels
     - Labels centered

    Parameters
    ----------
    vif : pd.Series
        VIF values indexed by feature names.
    title : str
        Plot title.
    figsize : Tuple[int,int], default=(8,4)
        Figure size in inches.
    wrap_width : int, default=20
        Maximum characters per line for tick labels.

    Returns
    -------
    None
    """
    thresholds = {'Negligible': 1,
                  'Moderate': 5,
                  'High': 10}

    # Sort descending & compute upper y-limit
    vif_sorted = vif.sort_values(ascending=False)
    max_val    = vif_sorted.max() + 1.05
    max_thresh = max(thresholds.values())
    upper      = max(max_val, max_thresh) * 1.05  # 5% headroom

    # Wrap labels
    wrapped = [textwrap.fill(lbl, wrap_width) for lbl in vif_sorted.index]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(vif_sorted)), vif_sorted.values, edgecolor='k')

    # add a tiny x-margin so bars at the ends aren’t right up against the axes
    ax.margins(x=0.01)

    ax.set_xticks(range(len(vif_sorted)))
    if xticks_rotation != 90:
        ax.set_xticklabels(wrapped,
                           rotation=xticks_rotation,  # prefered is 45
                           ha='right',
                           va='center',
                           rotation_mode='anchor')
    else:
        ax.set_xticklabels(wrapped, rotation=xticks_rotation, ha='center')

    ax.set_title(title, pad=12)
    ax.set_ylabel('VIF')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Draw threshold lines with legend entries
    colors = {'Negligible': 'green', 'Moderate': 'orange', 'High': 'red'}
    for name, thresh in thresholds.items():
        # Skip drawing “High” if your data never reaches 10
        if thresh == 10 and max_val < 10:
            continue
        ax.axhline(thresh,
                   color=colors[name],
                   linestyle='--',
                   linewidth=1,
                   label=f'{name} (VIF={thresh})')

    ax.set_ylim(0, max_val)
    ax.legend(title='Thresholds', loc='upper right')

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir / f'{title}_vif.png', dpi=300)
    plt.show()


# %% Model call to evalaute decompostion on prediction model
def evalaute_pca_in_Model(
        prefix_raw: str,
        bin_cols:List[str],
        df_components: pd.DataFrame,
        df: pd.DataFrame,
        target: str = "ahi",
        stratify_col: str = "osa_four",
        n_trials: int = 50,
        cv_folds: Union[int, bool] = False,
        use_gpu: bool = True,
        in_params: Optional[dict] = None,
        show_loss_curve:Optional[bool] = False,
        output_dir:pathlib.Path = None,
):
    """
    For a given prefix:
      - trains an XGB on PCA components + demographics
      - trains an XGB on raw features      + demographics
      - returns two (metrics_df, summary_df) tuples with an extra column 'model_type'
    """
    metric_dfs = []
    summary_dfs = []

    # --- Model A: PCA components + demographics ------------------------------
    # 1) assemble df for PCA model
    df_model = (
        pd.merge(
            df_components,
            df[["dem_age", "dem_bmi", stratify_col, target]],
            left_index=True,
            right_index=True
        )
        .loc[lambda d: d[target].notna()]
    )
    features_pca = [
        c for c in df_model.columns
        if c not in [target, stratify_col]
    ]
    # 2) fit & collect
    fi_a, preds_a, params_a, _ = train_xgb_collect(
        data=df_model,
        in_params=in_params,
        feature_cols=features_pca,
        target_col=target,
        optimization=True,
        val_size=0.2,
        test_size=0.1,
        n_trials=n_trials,
        cv_folds=cv_folds,
        use_gpu=use_gpu,
        show_loss_curve=show_loss_curve,
        stratify_col=stratify_col,
        model_path=output_dir.joinpath(f"'PCA_dem_{target}")
    )
    metrics_a, summary_a = compute_regression_metrics(preds_a, n_feats=len(features_pca))
    metrics_a['model_type'] = 'PCA'
    summary_a['model_type'] = 'PCA'
    metric_dfs.append(metrics_a)
    summary_dfs.append(summary_a)

    # --- Model B: raw features + demographics -------------------------------
    df_raw = df[
        # [col for col in df.columns if col.startswith(prefix_raw)]
        bin_cols +
        ["dem_age", "dem_bmi", stratify_col, target]
        ].copy().loc[lambda d: d[target].notna()]

    features_raw = [
        c for c in df_raw.columns
        if c not in [target, stratify_col]
    ]

    fi_b, preds_b, params_b, _ = train_xgb_collect(
        data=df_raw,
        in_params=in_params,
        feature_cols=features_raw,
        target_col=target,
        optimization=True,
        val_size=0.2,
        test_size=0.1,
        n_trials=n_trials,
        cv_folds=cv_folds,
        use_gpu=use_gpu,
        show_loss_curve=show_loss_curve,
        stratify_col=stratify_col,
        model_path=output_dir.joinpath(f"'raw_demo_{target}"), #out_current_section.joinpath(f"{prefix}_RAW_{target}")
    )
    metrics_b, summary_b = compute_regression_metrics(preds_b, n_feats=len(features_raw))
    metrics_b['model_type'] = 'RAW'
    summary_b['model_type'] = 'RAW'
    metric_dfs.append(metrics_b)
    summary_dfs.append(summary_b)

    # --- Model C: raw features + demographics + PCA -------------------------------
    df_model = (
        pd.merge(
            df_components,
            df[bin_cols + ["dem_age", "dem_bmi", stratify_col, target]],
            left_index=True,
            right_index=True
        )
        .loc[lambda d: d[target].notna()]
    )

    fi_c, preds_c, params_c, _ = train_xgb_collect(
        data=df_model,
        in_params=in_params,
        feature_cols=[col for col in df_model if not col in [target, stratify_col]],
        target_col=target,
        optimization=True,
        val_size=0.4,
        test_size=0.1,
        n_trials=n_trials,
        cv_folds=cv_folds,
        use_gpu=use_gpu,
        stratify_col=stratify_col,
        model_path=output_dir.joinpath(f"'PCA_demo_raw{target}"), #out_current_section.joinpath(f"{prefix}_RAW_{target}")
    )
    metrics_b, summary_b = compute_regression_metrics(preds_b, n_feats=len(features_raw))
    metrics_b['model_type'] = 'RAW_PCA'
    summary_b['model_type'] = 'RAW_PCA'
    metric_dfs.append(metrics_b)
    summary_dfs.append(summary_b)

    # return concatenated for this prefix
    return pd.concat(metric_dfs, ignore_index=True), pd.concat(summary_dfs, ignore_index=True)


# %% Evaluation from metrics collection
def plot_metrics_by_prefix(df: pd.DataFrame,
                           output_dir: pathlib.Path = None):
    """
    Creates one figure per prefix (questionnaire section). Each figure includes three subplots (R², Adj. R², RMSE),
    showing error bars for the training set and points for the test set, grouped by model type.

    Parameters:
    - df: DataFrame with model performance metrics.
    - output_dir: Directory where plots will be saved.
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
    model_type_map = {
        'PCA': 'PCA',
        'RAW': 'Raw',
        'RAW_PCA': 'Raw + PCA'
    }
    df['model_type'] = df['model_type'].map(model_type_map)
    df['group'] = df['group'].str.capitalize()
    prefixes = df['prefix'].unique()

    for prefix in prefixes:
        df_prefix = df[df['prefix'] == prefix]
        model_types = df_prefix['model_type'].unique()

        fig = plt.figure(figsize=(16, 8))
        gs = GridSpec(2, 3, height_ratios=[0.9, 0.1], figure=fig)
        axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
        fig.suptitle(f"Performance Metrics – {prefix}", fontsize=16)

        for ax, metric_key in zip(axes, metric_map.keys()):
            for model in model_types:
                train = df_prefix[(df_prefix['group'] == 'Train') & (df_prefix['model_type'] == model)]
                test = df_prefix[(df_prefix['group'] == 'Test') & (df_prefix['model_type'] == model)]
                x_pos = list(model_types).index(model)

                if not train.empty:
                    ax.errorbar(
                        x=x_pos,
                        y=train[metric_key].values[0],
                        yerr=train[std_map[metric_key]].values[0],
                        fmt='o',
                        capsize=5,
                        color=f"C{x_pos}",
                        label="Train" if model == model_types[0] and metric_key == 'r2_mean' else None
                    )
                if not test.empty:
                    ax.scatter(
                        x=x_pos,
                        y=test[metric_key].values[0],
                        color='black',
                        marker='X',
                        s=100,
                        label="Test" if model == model_types[0] and metric_key == 'r2_mean' else None
                    )

            ax.set_title(metric_map[metric_key])
            ax.set_xticks(range(len(model_types)))
            ax.set_xticklabels(model_types, rotation=30, ha='right')
            ax.grid(True, linestyle='--', alpha=0.6)

        legend_ax = fig.add_subplot(gs[1, :])
        legend_ax.axis('off')
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        legend_ax.legend(unique.values(), unique.keys(), loc='center', ncol=2, fontsize='medium', frameon=False)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if output_dir:
            plt.savefig(output_dir.joinpath(f'{prefix.replace(" ", "_")}_metrics.png'), dpi=300)
        plt.show()

