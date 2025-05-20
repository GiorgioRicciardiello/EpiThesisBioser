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
from config.config import config, encoding
from typing import List, Optional, Dict
import pickle
from library.variance_decomposition.var_decomp import (plot_correlation_matrix,
                                                       compute_vif,
                                                       run_mca,
                                                       run_pca_and_whiten,
                                                        plot_vif,
                                                       plot_mca_scatter,
                                                       plot_heatmap,
                                                       plot_scree,
                                                       plot_component_heatmap,
                                                       plot_metrics_by_prefix,
                                                       evalaute_pca_in_Model,
                                                       train_xgb_collect,
                                                       compute_regression_metrics,)

def filter_columns_binary(cols, encoding):
    """Keep only the columns whose encoding dict has exactly 2 levels."""
    return [
        col for col in cols
        if col in encoding and len(encoding[col]['encoding']) == 2
    ]


if __name__ == "__main__":
    # Load your data
    # sections = ["mh_", "sa_"] # , "presleep_", "postsleep_"]
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    path_out_pca_data = config.get('data')['pp_data']['pca_reduced']
    # %% define output path
    out_dir = config.get("results")['dir'].joinpath('variance_decomposition')
    out_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = out_dir / "mh_mca_pca_results.pkl"

    # %% count how many binary variables we have
    sections = ["dem_", "ep_", "mh_", "sa_", "presleep_", "postsleep_", "resp"]
    count_var_types = {sec: {'binary': 0, 'ordinal': 0, 'continuous': 0}
                       for sec in sections}
    for key, meta in encoding.items():
        # find which section this variable belongs to
        for sec in sections:
            if key.startswith(sec):
                n_levels = len(meta.get('encoding', {}))
                if n_levels == 2:
                    count_var_types[sec]['binary'] += 1
                elif n_levels > 2:
                    count_var_types[sec]['ordinal'] += 1
                else:  # n_levels == 0
                    count_var_types[sec]['continuous'] += 1
                break  # once assigned, don’t check other sections

    total_questions = len(encoding)
    for section in sections:
        length = len([col for col in encoding.keys() if col.startswith(section)])
        print(f'{section}: {length} ({((length/total_questions) * 100):.2f}%) variables')

    # %% mapper for formal names on plots
    def get_formal_names(columns:List[str]) -> Dict[str, str]:
        bin_var_formal_names = {}
        for bin_var in columns:
            if bin_var in encoding:
                bin_var_formal_names[bin_var] = encoding[bin_var]['definition'].replace('_', ' ').title()
            else:
                bin_var_formal_names[bin_var] = bin_var.replace('_', ' ').title()
        return bin_var_formal_names


    prefix_formal = {
        "mh_": 'Medical History',
        "sa_": 'Sleep Assessment',
        'presleep_': 'Presleep',
        'postsleep_': 'Postsleep',
    }
    # %% multicollinearity pipeline and variance decomposition
    if pickle_path.exists():
        print(f'Results already exist in {pickle_path}. Loading from file.')
        # Load pre‐computed results
        with open(pickle_path, "rb") as f:
            results = pickle.load(f)
    else:
        print(f'Computing multicollinearity results and saving to {pickle_path}.')
        results = {}
        for raw_prefix in sections:

            if raw_prefix == 'presleep_':
                pass

            print(f'\n**{raw_prefix.title()} decomposition:')
            out_current_section = out_dir.joinpath(raw_prefix)
            out_current_section.mkdir(parents=True, exist_ok=True)
            putput_name_alias = f'{raw_prefix.replace("_", "")}'
            # 2) Identify binary columns
            all_cols = [c for c in df.columns if c.startswith(raw_prefix)]
            bin_cols = filter_columns_binary(all_cols, encoding)
            if not bin_cols:
                continue
            if len(bin_cols) < 3:
                continue
            perc_count = len(bin_cols) / len(all_cols) * 100
            print(f'Binary columns: {len(bin_cols)} ({perc_count:.2f}%))')
            X_df = df[bin_cols].fillna(0).copy()
            mapper = get_formal_names(bin_cols)
            X_df.rename(mapper, axis=1, inplace=True)

            X = X_df.values

            prefix = prefix_formal[raw_prefix]
            # Multicollinearity plots:
            df_corr = plot_correlation_matrix(
                df=X_df,
                title=f"Correlation matrix – {prefix}",
                output_dir=out_current_section,
                figsize=(10, 8) if raw_prefix in ['presleep_', 'postsleep_'] else (8, 6)
            )
            vif_series = compute_vif(X_df,
                                     output_dir=out_current_section)
            plot_vif(
                vif_series,
                figsize=(22, 8) if raw_prefix == 'mh_' else (12, 8),
                title=f"VIF – {prefix}",
                output_dir=out_current_section,
                xticks_rotation=90 if raw_prefix == 'mh_' else 45
            )

            # === MCA workflow ===
            mca_scores, col_coords, perc_inertia = run_mca(X_df)
            plot_mca_scatter(
                mca_scores, col_coords,
                title=f"MCA Biplot – {prefix}",
                save_path=out_current_section.joinpath(f"{prefix}mca_scatter.png")
            )
            plot_heatmap(
                col_coords,
                title=f"Heatmap of Row Coordinates – {prefix}",
                figsize=(8, 4),
                save_path=out_current_section.joinpath(f"{prefix}_heatmap_row_coord.png")
            )

            plot_heatmap(
                col_coords,
                title=f"Heatmap of Category Coordinates – {prefix}",
                figsize=(6, 20),
                save_path=out_current_section.joinpath(f"{prefix}_heatmap_category_coord.png")
            )
            plot_scree(
                perc_inertia,
                labels=[f"Dim{i+1}" for i in range(len(perc_inertia))],
                title=f"MCA Scree Plot – {prefix}",
                xlabel="Dimension",
                ylabel="% Inertia Explained",
                save_path=out_current_section.joinpath(f"{prefix}mca_scree.png")
            )

            # extract the MCA “components” as loadings of each one-hot variable:
            # shape = (n_components × n_categories)
            mca_components = col_coords.values.T
            mca_feature_names = col_coords.index.tolist()

            # === PCA + whitening workflow ===
            pca_var, orig_comps, white_comps, Uw_df = run_pca_and_whiten(X, prefix, df)

            plot_scree(
                pca_var,
                labels=[f"Comp{i+1}" for i in range(len(pca_var))],
                title=f"PCA (whitened) Scree – {prefix}",
                xlabel="Component",
                ylabel="% Variance Ratio",
                save_path=out_current_section.joinpath(f"{prefix}pca_whiten_scree.png"),
                figsize=(8, 6)
            )

            # 2) Heatmap of original PCA loadings (k components × p features)
            features = X_df.columns.tolist()
            comp_labels = [f"Comp{i + 1}" for i in range(orig_comps.shape[0])]
            plot_component_heatmap(
                orig_comps,
                feature_names=features,
                comp_labels=comp_labels,
                title=f"PCA Loadings Heatmap – \n{prefix}",
                figsize=(10, len(features) * 0.2 + 1) if raw_prefix == 'mh_' else (8, 6),
                horizontal=False,
                save_path=out_current_section.joinpath(f"{prefix}_pca_orig_compo.png"),

            )
            plot_component_heatmap(
                white_comps,
                feature_names=[f"PC{i+1}" for i in range(white_comps.shape[1])],
                comp_labels=comp_labels,
                title=f"Whitened PCA Loadings – \n{prefix}",
                save_path=out_current_section.joinpath(f"{prefix}_pca_white_comps.png"),
            )

            # 5) Store everything in results
            results[prefix] = {
                # MCA
                'mca_inertia_pct': perc_inertia,  # % inertia per MCA dim
                'mca_components': mca_components,  # array (n_dims × n_categories)
                'mca_component_names': mca_feature_names,  # list of category labels
                'mca_scores': mca_scores,  # DataFrame (n_samples × n_dims)

                # PCA + whitening
                'pca_whiten_variance': pca_var,  # whitened explained‐variance ratios
                'pca_components_orig': orig_comps,  # original PCA.components_ (n_dims × n_features)
                'pca_component_names': X_df.columns.tolist(),  # feature names for orig_comps
                'pca_whiten_components': white_comps,  # whitened PCA.components_ (n_dims × n_dims)
                'pca_whiten_scores': Uw_df,  # DataFrame (n_samples × n_dims)

                'bin_cols': bin_cols,
            }

        # Save for next time
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)


    # %%
    # prefix_raw = 'mh_'
    # out_current_section = out_dir.joinpath(prefix_raw)
    # assert out_current_section.exists()
    # prefix = prefix_formal[prefix_raw]
    #
    # df_components = pd.DataFrame(results[prefix]['pca_whiten_scores'])
    # target = 'ahi'  #'resp-oa-total'
    # stratify_col = 'osa_four'
    # df_model = pd.merge(df_components,
    #                     df[["dem_age", "dem_bmi", 'osa_four', target]],
    #                      left_index=True,
    #                      right_index=True)
    # df_model = df_model.loc[df_model[target].notna(), :]
    #
    # features = [col for col in df_model if not col in [target, stratify_col]]
    # fi_df, preds_df, best_params, final_bst = train_xgb_collect(
    #     data=df_model,
    #     in_params={'n_estimators': 1000},
    #     feature_cols=features,
    #     target_col=target,
    #     optimization=True,
    #     val_size=0.2,
    #     test_size=0.1,
    #     n_trials=2,
    #     cv_folds=5,
    #     use_gpu=True,
    #     stratify_col=stratify_col,
    #     model_path=out_current_section.joinpath(f'{prefix}_{target}'),
    # )
    # metrics_df, summary_df = compute_regression_metrics(preds_df, n_feats=len(features))
    #
    #
    # # the mode without the pca and teh sparse columns
    # base_columns = [col for col in df.columns if col.startswith(prefix_raw)]
    # base_columns = base_columns + ["dem_age", "dem_bmi", 'osa_four', target]
    # df_model = df[base_columns].copy()
    #
    # fi_df, preds_df, best_params, final_bst = train_xgb_collect(
    #     data=df_model,
    #     in_params={'n_estimators': 1000},
    #     feature_cols=features,
    #     target_col=target,
    #     optimization=True,
    #     val_size=0.2,
    #     test_size=0.1,
    #     n_trials=2,
    #     cv_folds=5,
    #     use_gpu=True,
    #     stratify_col=stratify_col,
    #     model_path=out_current_section.joinpath(f'{prefix}_{target}'),
    # )
    # metrics_df, summary_df = compute_regression_metrics(preds_df, n_feats=len(features))

    # %%

    if not (out_dir.joinpath('summary_pca_test.csv').is_file()) or not (out_dir.joinpath('summary_pca_test.csv').is_file()):
        path_out_pca_models = out_dir.joinpath('pca_models')
        path_out_pca_models.mkdir(parents=True, exist_ok=True)
        all_metrics = []
        all_summaries = []
        for prefix, res_meta in results.items():
            df_components = pd.DataFrame(results[prefix]['pca_whiten_scores'])

            m_df, s_df = evalaute_pca_in_Model(
                prefix_raw=prefix,
                df=df,
                bin_cols=res_meta['bin_cols'],
                df_components=df_components,
                target='ahi_log1p',
                in_params={'num_boost_round': 1000,
                           'early_stopping_rounds': 50},
                stratify_col='osa_four',
                n_trials=15,
                cv_folds=False,
                use_gpu=True,
                show_loss_curve=True,
                output_dir=path_out_pca_models,
            )
            # tag which prefix these rows come from
            m_df['prefix'] = prefix
            s_df['prefix'] = prefix

            all_metrics.append(m_df)
            all_summaries.append(s_df)

        # concatenate across prefixes
        df_metrics_pca_test = pd.concat(all_metrics, ignore_index=True)
        df_summary_pca_test = pd.concat(all_summaries, ignore_index=True)

        df_metrics_pca_test.to_csv(out_dir.joinpath('metrics_pca_test.csv'), index=False)
        df_summary_pca_test.to_csv(out_dir.joinpath('summary_pca_test.csv'), index=False)
    else:
        df_metrics_pca_test = pd.read_csv(out_dir.joinpath('metrics_pca_test.csv'))
        df_summary_pca_test = pd.read_csv(out_dir.joinpath('summary_pca_test.csv'))

    PLOT_METRICS = False
    if PLOT_METRICS:
        plot_metrics_by_prefix(df=df_summary_pca_test.copy(),
                              output_dir=out_dir)
    # %% merge the pca into the data
    prefix_formal_inv = {val:key for key, val in prefix_formal.items()}
    df_merged_pca = df.copy()
    for formal_prefix, res_meta in results.items():
        # Get alias (short name) for the formal section name
        prefix_alias = prefix_formal_inv[formal_prefix]
        bin_cols = res_meta['bin_cols']
        # Get PCA scores and rename columns
        df_pca_score = res_meta['pca_whiten_scores']
        columns = [f'{prefix_alias}pca_{idx}' for idx in range(1, df_pca_score.shape[1] + 1)]
        df_pca_score.columns = columns

        # Merge with main dataframe
        df_merged_pca = pd.concat([df_merged_pca, df_pca_score], axis=1)
        df_merged_pca.drop(columns=bin_cols, inplace=True)
        assert df.shape[0] == df_merged_pca.shape[0]


    # sot the columns
    priority_cols = [
        'id', 'study_year', 'md_identifyer', 'studytype',
        'dem_age', 'dem_gender', 'dem_race', 'dem_bmi', 'ess',
        'rdi','lowsat', 'tib', 'tst', 'sme', 'so', 'rol', 'ai', 'plms',
        'di', 'sen', 'sao2_per', 'lps', 'ahi', 'isl', 'usl','wasorace'
    ]
    priority_cols = [col for col in priority_cols if col in df_merged_pca.columns]
    ahi_cols = [col for col in df_merged_pca.columns if col.startswith('ahi') and col not in priority_cols]

    remaining_cols = sorted([
        col for col in df_merged_pca.columns
        if col not in priority_cols and col not in ahi_cols
    ])
    final_cols = priority_cols + remaining_cols + ahi_cols
    df_merged_pca = df_merged_pca[final_cols]

    # %% Save the pre-process version with pca
    df_merged_pca.to_csv(path_out_pca_data, index=False)
    print(f'PCA data saved to {path_out_pca_data}')



