import pathlib
from typing import  Optional, List, Union, Tuple
import numpy as np
import json
from config.config import config, metrics_psg, encoding, sections

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib
import os
from textwrap import wrap


def compare_model_metrics(root_dir: pathlib.Path,
                          plot: bool = True,
                        annotation_font_size: int = 12,
                          output_path:pathlib.Path=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collects classifier and regressor metrics from the specified results directory.
    Optionally generates plots with connected line plots.

    Parameters:
    - root_dir: Path to the folder with metrics
    - plot: Whether to visualize results

    Returns:
    - df_classifiers: Cleaned classifier metrics
    - df_regressors: Cleaned regressor metrics
    """
    classifier_dfs = []
    regressor_dfs = []

    # Mapping for prettier model names
    def rename_model(model: str) -> str:
        replacements = {
            "base_classifier_binary_logistic": "Base Binary",
            "base_classifier_fifteenth_logistic": "Base Fifteen",
            "base_classifier_four_softprob": "Base 4-Class",
            "final_classifier_binary_logistic": "Final Binary",
            "final_classifier_fifteenth_logistic": "Final Fifteen",
            "final_classifier_four_softprob": "Final 4-Class",
            "base_regressor": "Base Regressor",
            "final_regressor": "Final Regressor",
            "resp-ca-total_idx_log1p": "CA/h",
            "resp-hi_hypopneas_only-total_idx_log1p": "Hypopneas/h",
            "resp-ma-total_idx_log1p": "MA/h",
            "resp-oa-total_idx_log1p": "OA/h",
            "resp-ri_rera_only-total_idx_log1p": "RERA/h",
        }
        return replacements.get(model, model)

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = pathlib.Path(subdir) / file

            if file == "classifier_metrics_df.csv":
                df = pd.read_csv(file_path)
                df["model"] = rename_model(file_path.parent.name)
                df = df[df["class_label"] == "global"]
                classifier_dfs.append(df)

            elif file == "metrics_df.csv":
                df = pd.read_csv(file_path)
                df["model"] = rename_model(file_path.parent.name)
                regressor_dfs.append(df)

    df_classifiers = pd.concat(classifier_dfs, ignore_index=True) if classifier_dfs else pd.DataFrame()
    df_regressors = pd.concat(regressor_dfs, ignore_index=True) if regressor_dfs else pd.DataFrame()
    metrics_reg = ['rmse', 'medae', 'explained_variance', 'r2_adj']
    df_regressors[metrics_reg] = df_regressors[metrics_reg].round(3)

    if 'recall' in df_classifiers.columns:
        df_classifiers.rename(columns={'recall': 'sensitivity'}, inplace=True)

    df_classifiers['model'] = df_classifiers.model.str.replace('Base', 'Single')
    df_classifiers['model'] = df_classifiers.model.str.replace('Final', 'Stacked')

    df_regressors['model'] = df_regressors['model'].replace({'Base Regressor': 'Single Regressor (AHI)',
                                                            'Final Regressor': 'Stacked Regressor (AHI)'})

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        df_regressors.to_csv(output_path.joinpath('metrics_df.csv'), index=False)
        df_classifiers.to_csv(output_path.joinpath('classifier_metrics_df.csv'), index=False)

    if plot:
        # Plot Classifiers
        if not df_classifiers.empty:
            metrics = ['accuracy', 'precision', 'sensitivity', 'f1_score']
            df_clf_plot = df_classifiers[df_classifiers['split'] == 'test_true'].copy()
            df_clf_plot['model_wrapped'] = df_clf_plot['model'].apply(lambda x: '\n'.join(wrap(x, 12)))

            unique_models = df_clf_plot['model_wrapped'].unique()
            color_map = dict(zip(unique_models, sns.color_palette("pastel", n_colors=len(unique_models))))
            df_clf_plot['color'] = df_clf_plot['model_wrapped'].map(color_map)
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, metric in zip(axes.flat, metrics):
                # Sort data for current metric
                df_sorted = df_clf_plot.sort_values(by=metric, ascending=False)
                sns.barplot(data=df_sorted,
                            x='model_wrapped',
                            y=metric,
                            ax=ax,
                            # color='lightblue',
                            palette=df_sorted['color'].tolist(),
                            edgecolor='black')
                sns.lineplot(data=df_clf_plot,
                             x='model_wrapped',
                             y=metric,
                             ax=ax,
                             color='black',
                             marker='o')


                for bar in ax.patches:
                    h = bar.get_height()
                    ax.annotate(f"{h:.3f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=annotation_font_size)

                ax.set_title(f"Classifier Test {metric.capitalize().replace('_', ' ')}")
                ax.set_ylabel(metric.capitalize())
                ax.set_xlabel('')
                ax.tick_params(axis='x', rotation=0, labelsize=9)
                ax.set_xticklabels(ax.get_xticklabels(), ha='center')
                ax.grid(alpha=0.4)

            plt.tight_layout()
            if output_path:
                plt.savefig(output_path.joinpath('classifier_metrics.png'), dpi=300)
            plt.show()

        # Plot Regressors
        if not df_regressors.empty:
            metrics_reg = ['rmse', 'medae', 'explained_variance', 'r2_adj']
            df_reg_plot = df_regressors[df_regressors['split'] == 'test_true'].copy()
            df_reg_plot['model_wrapped'] = df_reg_plot['model'].apply(lambda x: '\n'.join(wrap(x, 12)))

            unique_models = df_reg_plot['model_wrapped'].unique()
            color_map = dict(zip(unique_models, sns.color_palette("pastel", n_colors=len(unique_models))))
            df_reg_plot['color'] = df_reg_plot['model_wrapped'].map(color_map)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, metric in zip(axes.flat, metrics_reg):
                if metric in ['r2_adj', 'explained_variance']:
                    df_sorted = df_reg_plot.sort_values(by=metric, ascending=False)
                else:
                    df_sorted = df_reg_plot.sort_values(by=metric, ascending=True)

                sns.barplot(data=df_sorted,
                            x='model_wrapped',
                            y=metric,
                            ax=ax,
                            # color='lightgray',
                            palette=df_sorted['color'].tolist(),
                            edgecolor='black')
                sns.lineplot(data=df_reg_plot,
                             x='model_wrapped',
                             y=metric,
                             ax=ax,
                             color='darkblue',
                             marker='o')

                for bar in ax.patches:
                    h = bar.get_height()
                    ax.annotate(f"{h:.3f}",
                                xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                fontsize=annotation_font_size)

                ax.set_title(f"Regressor Test {metric.replace('_', ' ').title()}")
                ax.set_ylabel(metric.replace('_', ' ').capitalize())
                ax.set_xlabel('')
                y_values = df_sorted[metric]
                y_min, y_max = y_values.min(), y_values.max()
                pad = (y_max - y_min) * 0.10  # 5% padding above and below
                ax.set_ylim(y_min - pad, y_max + pad)
                ax.tick_params(axis='x', rotation=0, labelsize=9)
                ax.set_xticklabels(ax.get_xticklabels(), ha='center')
                ax.grid(alpha=0.4)

            plt.tight_layout()
            if output_path:
                plt.savefig(output_path.joinpath('regression_metrics.png'), dpi=300)
            plt.show()

    return df_classifiers, df_regressors


# %%
if __name__ == '__main__':
    # %% Input data

    # %% output directory
    path_dir = config.get('results')['dir'].joinpath('regres_classif_xgboost')
    output_path =  config.get('results')['dir'].joinpath('regres_classif_xgboost_summary')
    # %%
    df_classifiers, df_regressors = compare_model_metrics(root_dir=path_dir,
                                                          output_path=output_path,
                                                          plot=True)








