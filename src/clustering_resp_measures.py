import pathlib

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import shapiro, f_oneway, kruskal, chi2, chi2_contingency
from config.config import config,encoding, sections, metrics_psg
from itertools import combinations
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from tabulate import tabulate
from library.TableOne.table_one import MakeTableOne
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
import gower
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP
from sklearn.preprocessing import StandardScaler


def epsilon_squared(H, n):
    """Effect size for Kruskal–Wallis"""
    return H / (n - 1)

def evaluate_resp_by_cluster(df,
                             resp_vars:List[str],
                             cluster_col='cluster',
                             method='fdr_bh'):
    """Run Kruskal–Wallis and epsilon² per respiratory variable with p-value correction"""
    results = []

    # Collect raw p-values first
    raw_pvals = []
    temp = []

    for var in resp_vars:
        groups = [df[df[cluster_col] == c][var].dropna() for c in df[cluster_col].unique()]
        if all(len(g) > 0 for g in groups):
            H, p = kruskal(*groups)
            eps2 = epsilon_squared(H, df.shape[0])
            raw_pvals.append(p)
            temp.append({'resp_variable': var, 'H': H, 'p_uncorrected': p, 'epsilon_squared': eps2})

    # Apply correction
    reject, pvals_corrected, _, _ = multipletests(raw_pvals, alpha=0.05, method=method)

    # Combine corrected p-values with results
    for i, res in enumerate(temp):
        res['p_corrected'] = pvals_corrected[i]
        res['reject'] = reject[i]
        results.append(res)

    return pd.DataFrame(results).sort_values('epsilon_squared', ascending=False)


def visualize_clusters_umap(
        X,
        cluster_labels,
        title='UMAP Projection',
        metric='precomputed',
        figsize=(8, 6),
        point_size=60,
        alpha=0.8,
        palette='Set2',
        legend_loc='best',
        save_path=None,
        show=True,
):
    """
    Visualize clustering results using UMAP in 2D.

    Parameters:
    - X: numpy array or distance matrix (if metric='precomputed')
    - cluster_labels: list or array-like of cluster assignments
    - title: plot title
    - metric: 'precomputed' or any valid UMAP metric
    - figsize: tuple for plot size
    - point_size: size of points in the scatter plot
    - alpha: transparency
    - palette: seaborn color palette
    - legend_loc: location of legend
    - save_path: if provided, saves the plot to this path
    - show: whether to display the plot (useful for saving without showing)

    Returns:
    - X_2d: the 2D UMAP projection
    """
    reducer = UMAP(metric=metric, random_state=42)
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=figsize)
    sns.scatterplot(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        hue=cluster_labels,
        palette=palette,
        s=point_size,
        alpha=alpha,
        edgecolor='white',
        linewidth=0.3
    )
    plt.title(title, fontsize=14)
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.legend(title='Cluster', loc=legend_loc)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    return X_2d


if __name__ == '__main__':
    osa_mapper = {
        0: 'Normal',
        1: 'Mild',
        2: 'Moderate',
        3: 'Severe',
    }
    path_output = config.get('results')['dir'].joinpath('post_vs_pre')
    path_output.mkdir(exist_ok=True, parents=True)

    df_data = pd.read_csv(
        config.get('data')['pp_data']['pca_reduced_transf'],
        low_memory=False
    )

    # %%
    section_features = [sect for sect in sections if not sect in ['presleep_', 'resp']]
    # 1. Define your sections and columns
    features = [col for col in df_data.columns
                    if any(col.startswith(alias) for alias in section_features)]
    features.append('sleep_hours')

    resp_measures = ['ahi',
                        'resp-oa-total',
                         'resp-ca-total',
                         'resp-ma-total',
                         'resp-hi_hypopneas_only-total',
                         'resp-ri_rera_only-total']
    features.extend(resp_measures)

    # resp_measures_idx = metrics_psg.get('resp_events')['indices']
    df_data.loc[df_data['sleep_hours'] < 2, 'sleep_hours'] = 2

    # Compute index values
    resp_measures_idx = []
    for resp in resp_measures:
        if resp == 'ahi':
            # ahi is already an index so we do not need to compute it
            continue
        col_idx = f'{resp}_idx'
        df_data[col_idx] = df_data[resp] / df_data['sleep_hours']
        resp_measures_idx.append(col_idx)

    features.extend(resp_measures_idx)

    features = list(set(features))
    # drp rows with any mising data in the resp columns
    df = df_data.dropna(subset=resp_measures, how='any', axis=0).copy()

    # replace the nans of teh contunous with zero
    df = df[features + ['osa_four_numeric']].fillna(0)


    for col in features: print(col)

    # %% Plot for each respiratory measure and its index
    PLOT = False
    if PLOT:
        for resp in resp_measures[0:1]:
            if resp == 'ahi': continue
            idx_name = f'{resp}_idx'

            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=False)

            for ax, col, label in zip(axes, [resp, idx_name], ['Raw', 'Indexed']):
                data = df[col].dropna()

                sns.histplot(data, stat='probability', kde=True, ax=ax)

                mean_val = data.mean()
                median_val = data.median()
                std_val = data.std()

                ax.axvline(mean_val,
                           color='blue',
                           linestyle='--',
                           label=f'Mean: {mean_val:.2f}',
                           alpha=.4)
                ax.axvline(median_val,
                           color='green',
                           linestyle='-',
                           label=f'Median: {median_val:.2f}',
                           alpha=.4)
                ax.axvline(mean_val + std_val,
                           color='red',
                           linestyle=':',
                           label=f'+1 STD: {mean_val + std_val:.2f}',
                           alpha=.4)
                ax.axvline(mean_val - std_val,
                           color='red',
                           linestyle=':',
                           label=f'-1 STD: {mean_val - std_val:.2f}',
                           alpha=.4)
                ax.set_title(f'{label} Distribution of {col}')
                ax.set_xlabel(col)
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.7)
                ax.legend()
                ax.set_xlim(data.min() - 1, data.max() + 1)
            plt.tight_layout()
            plt.show()


    # %% Data transformation for clusterting
    # Scale only continuous columns (important!)
    pca_cols = [col for col in features if col.startswith('pca_')]
    continuous_cols = [col for col, meta in encoding.items() if len(meta['encoding']) == 0]
    continuous_cols = continuous_cols + pca_cols
    df_scaled = df.copy()

    if continuous_cols:
        df_scaled[continuous_cols] = StandardScaler().fit_transform(df_scaled[continuous_cols])


    # %% random sample for debugging
    from sklearn.model_selection import train_test_split

    # Define your stratified sample size
    sample_size = 5000

    # Ensure the dataset is large enough
    assert df_scaled.shape[0] >= sample_size, "Dataset too small for this sample size"

    # Stratified sampling
    df_sampled, _ = train_test_split(
        df_scaled,
        train_size=sample_size,
        stratify=df['osa_four_numeric'],  # from the original, not dropping rows before so ok
        random_state=42
    )

    # Now subset features for Gower/UMAP/etc.
    X = df_sampled[features].copy()


    # %% Unsupervised clustering
    # X = df_scaled[features].copy()
    # Compute Gower distance matrix
    gower_dist = gower.gower_matrix(X)

    # Cluster with Agglomerative on Gower distance
    model = AgglomerativeClustering(n_clusters=4,
                                    linkage='average')
    X['cluster'] = model.fit_predict(gower_dist)
    df['cluster'] =  X['cluster'].copy()


    results_df = evaluate_resp_by_cluster(X, resp_measures)
    print(results_df)

    strata = 'cluster'
    def get_var_dtypes(columns:List[str]) -> Dict[str, List[str]]:
        """
        Determines the data types of variables based on their encoding, categorizing them into
        'continuous', 'binary', or 'ordinal_categorical'.

        :param columns: A list of variable names to be classified into different data types.
        :type columns: List[str]

        :return: A dictionary where keys are data type categories ('continuous',
            'binary', 'ordinal_categorical') and values are lists of variables
            belonging to each category.
        :rtype: Dict[str, List[str]]
        """
        var_types = {
            'continuous': [],
            'binary': [],
            'ordinal_categorical': [],
        }

        for var in columns:
            if not var in encoding:
                continue
            if len(encoding[var]['encoding']) == 0:
                var_types.get('continuous').append(var)
            if len(encoding[var]['encoding']) == 2:
                var_types.get('binary').append(var)
            if len(encoding[var]['encoding']) > 2:
                var_types.get('ordinal_categorical').append(var)

        return var_types

    vars_continuous = get_var_dtypes(features)['continuous']
    vars_continuous = vars_continuous + resp_measures + resp_measures_idx
    vars_continuous = list(np.sort(vars_continuous))
    vars_binary = get_var_dtypes(features)['binary']
    vars_ordinal_categorical = get_var_dtypes(features)['ordinal_categorical']
    vars_categorical = vars_binary + vars_ordinal_categorical
    vars_all = vars_continuous + vars_categorical



    tab_post_sleep = MakeTableOne(df=df[~df['cluster'].isna()],
                                  continuous_var=vars_continuous,
                                  categorical_var=vars_categorical,
                                  strata=strata)
    df_clusters = tab_post_sleep.create_table()


    # Save to CSV
    df_clusters.to_csv(path_output / 'df_clusters.csv', index=False)


    # Visualize clusters in 2D
    visualize_clusters_umap(gower_dist, X['cluster'], title='Questionnaire-Derived Clusters (UMAP)')



    # %%
    import numpy as np
    from gower import gower_matrix
    from sklearn.cluster import DBSCAN

    resp_measures_weights = {var: 1.0 for var in resp_measures_idx + resp_measures}
    resp_measures_weights = {1: 'ahi'}
    # Assign weights
    uniform_weight = 0.2
    weights = np.array([resp_measures_weights.get(f, uniform_weight) for f in features])

    # Prepare data
    X = df_sampled[features].copy()


    # Step 3: Compute weighted Gower distance
    gower_dist = gower.gower_matrix(data_x=X )  # weight=weights)
    # Step 4: Run DBSCAN on the distance matrix
    model = DBSCAN(eps=0.45, min_samples=5, metric='precomputed')  # Tune eps as needed
    X['cluster'] = model.fit_predict(gower_dist)
    X['cluster'].value_counts()






























