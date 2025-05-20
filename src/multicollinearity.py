from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from config.config import config, sections, encoding, metrics_psg
from library.helper import get_mappers, classify_osa
import pandas  as pd
from typing import List, Optional, Tuple
from typing import List
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


if __name__ == '__main__':
    # %% Input data
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    output_path = config.get('results')['multicollinearity_check']
    # %%  Explore Variable Categories
    epworth_vars = [col for col in df.columns if col.startswith("ep_")]
    sa_vars = [col for col in df.columns if col.startswith("sa_")]
    pre_vars = [col for col in df.columns if col.startswith("presleep")]
    post_vars = [col for col in df.columns if col.startswith("postsleep")]
    mh_vars = [col for col in df.columns if col.startswith("mh_")]
    ph2_vars = [col for col in df.columns if col.startswith("ph2_")]
    metrics_resp = [col for col in df.columns if col.startswith("resp")]


    def get_var_dtype(variable: str) -> str:
        for type_, values in grouped_map.items():
            if variable in values:  # Check if variable is in the values list directly
                return type_
        return 'continuous'  # Moved outside the loop to return only if not found


    sections_dict = {
        section: [(col, get_var_dtype(col)) for col in df.columns if col.startswith(section)]
        for section in sections
    }

    def compute_vif(data: pd.DataFrame, section_columns: List[tuple], section_name: str):
        """
        Compute VIF and correlation heatmaps for a section, with subplots for binary, ordinal, and continuous variables.

        Args:
            data: Input DataFrame.
            section_columns: List of (column_name, dtype) tuples for the section.
            section_name: Name of the section (e.g., 'ep', 'mh_').
        """
        # Group columns by data type (binary, ordinal, continuous)
        type_to_columns = {'binary': [], 'ordinal': [], 'continuous': []}
        for col_name, dtype in section_columns:
            if col_name in data.columns:  # Ensure column exists in DataFrame
                type_to_columns[dtype].append(col_name)

        # Create subplots for each data type
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=False)
        axes = axes.flatten()
        titles = [
            f'{section_name}: Binary Variables',
            f'{section_name}: Ordinal Variables',
            f'{section_name}: Continuous Variables'
        ]

        for idx, (dtype, dtype_cols) in enumerate(type_to_columns.items()):
            if not dtype_cols:  # Skip if no columns for this data type
                axes[idx].set_visible(False)
                continue

            # Select and clean data for the specific data type
            selected_df = data[dtype_cols].copy()
            selected_df_numeric = selected_df.select_dtypes(include=["int64", "float64"])
            selected_df_numeric = selected_df_numeric.loc[:, selected_df_numeric.std() > 0]
            selected_df_clean = selected_df_numeric.dropna()

            if selected_df_clean.empty or selected_df_clean.shape[1] == 0:
                axes[idx].set_visible(False)
                continue

            # Compute correlation matrix (Spearman for ordinal, Pearson for continuous/binary)
            corr_method = 'spearman' if dtype == 'ordinal' else 'pearson'
            corr_matrix = selected_df_clean.corr(method=corr_method)

            # Plot heatmap
            sns.heatmap(corr_matrix, ax=axes[idx], cmap="coolwarm", center=0,
                        xticklabels=False, yticklabels=False, cbar=True)
            axes[idx].set_title(titles[idx])

            # VIF Calculation
            X_scaled = StandardScaler().fit_transform(selected_df_clean)
            vif_data = pd.DataFrame()
            vif_data["feature"] = selected_df_clean.columns
            vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
            vif_data_sorted = vif_data.sort_values(by="VIF", ascending=False).round(2)
            print(f"VIF for {section_name} ({dtype} variables):\n{vif_data_sorted}\n")

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"correlation_heatmaps_{section_name}.png")
        plt.close()


    # Loop through sections_dict to process each section
    for section_, value_ in sections_dict.items():
        compute_vif(data=df, section_columns=value_, section_name=section_)