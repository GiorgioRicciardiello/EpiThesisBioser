import pathlib
from typing import  Optional, List, Union
from config.config import config, encoding
from library.helper import get_mappers, classify_osa
import pandas  as pd
from library.stat_tests.my_within_stats_tests import (fit_osa_severity_mixedlm,
                                                      qqplot_pre_post_pairs_combined,
                                                      apply_within_pairs_from_dict)


if __name__ == '__main__':
    # %% Input data
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    path_output = config.get('results')['within_stats']

    # %% Post sleep perception

    from statsmodels.miscmodels.ordinal_model import OrderedModel

    cols_post_sleep = ['postsleep_feeling_sleepy',
                       'postsleep_feeling_alert',
                       'postsleep_feeling_tired']


    def plot_post_sleep_stacked_by_gender(
            df: pd.DataFrame,
            outcome: str,
            group: str,
            gender_col: str,
            encoding_map: dict
    ):
        """
        Plots a stacked barplot of post-sleep feeling by OSA severity and gender with proportion annotations.

        Parameters
        ----------
        df : pd.DataFrame
        outcome : str
            Column like 'postsleep_feeling_sleepy'
        group : str
            OSA severity column (e.g., 'osa_four')
        gender_col : str
            Gender column name
        encoding_map : dict
            Mapping like {0: 'None', 1: 'A little', 2: 'Quite a bit', 3: 'Extreme'}
        """
        # Drop NAs and apply label mapping
        df_plot = df[[gender_col, group, outcome]].dropna()
        df_plot[outcome] = df_plot[outcome].map(encoding_map)

        # Create grouped data: gender x OSA severity → counts of each outcome level
        grouped = df_plot.groupby([gender_col, group, outcome]).size().unstack(fill_value=0)
        proportions = grouped.div(grouped.sum(axis=1), axis=0)

        # Flatten the index for better x-axis labels
        x_labels = [f"{g}\n{sev}" for g, sev in proportions.index]

        # Plot
        ax = proportions.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='viridis')

        for i, (g, sev) in enumerate(proportions.index):
            for j, cat in enumerate(proportions.columns):
                val = proportions.loc[(g, sev), cat]
                if val > 0.02:
                    ax.text(
                        i,
                        proportions.iloc[i, :j].sum() + val / 2,
                        f"{val * 100:.0f}%",
                        ha='center', va='center',
                        fontsize=9, color='white', weight='bold'
                    )

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=0)
        ax.set_xlabel('Gender + OSA Severity')
        ax.set_ylabel('Proportion')
        ax.set_title(f'Distribution of {outcome.replace("_", " ").title()} by OSA Severity and Gender')
        ax.legend(title='Feeling Level', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


    def fit_ordered_logit_model(
            df: pd.DataFrame,
            outcome: str,
            group: str = 'osa_four',
            covariates: list = ['age', 'race', 'gender'],
            encoding_order: list = ['Normal', 'Mild', 'Moderate', 'Severe']
    ):
        """
        Fits an ordered logistic regression with outcome ~ OSA severity (+ covariates).
        Returns statsmodels result.
        """
        cols = [outcome, group] + covariates
        df_model = df[cols].dropna().copy()

        # Convert outcome to numeric
        df_model[outcome] = pd.to_numeric(df_model[outcome], errors='coerce').astype(int)

        # Convert group to ordered categorical with numeric codes
        df_model[group] = pd.Categorical(df_model[group],
                                         categories=encoding_order,
                                         ordered=True).codes + 1  # +1 to avoid 0-based coding

        # Separate covariates
        cat_covs = [col for col in covariates
                    if not pd.api.types.is_numeric_dtype(df_model[col])]
        num_covs = [col for col in covariates
                    if pd.api.types.is_numeric_dtype(df_model[col])]

        # Process covariates
        X_group = df_model[[group]]  # Use numeric codes directly
        X_num = df_model[num_covs].astype(float) if num_covs else pd.DataFrame()
        X_cat = pd.get_dummies(df_model[cat_covs], drop_first=True) if cat_covs else pd.DataFrame()

        # Combine features
        X = pd.concat([X_group, X_num, X_cat], axis=1).dropna(axis=1)

        # Remove constant columns
        X = X.loc[:, X.nunique() > 1]

        # Align indices
        y = df_model[outcome].reset_index(drop=True)
        X = X.reset_index(drop=True)

        # Fit model
        model = OrderedModel(y, X, distr='logit')
        m =  model.fit(method='bfgs', disp=False)
        return model.fit(method='bfgs', disp=False)

    encoding_map = {val: key for key, val in encoding.get(cols_post_sleep[0])['encoding'].items()}

    # Plot proportions with annotations
    plot_post_sleep_stacked_by_gender(
        df=df,
        outcome='postsleep_feeling_sleepy',
        group='osa_four',
        gender_col='gender',
        encoding_map=encoding_map,
        encoding_order=['Normal', 'Mild', 'Moderate', 'Severe']
    )

    # Fit ordered logistic regression (adjusted)
    res = fit_ordered_logit_model(
        df=df,
        outcome='postsleep_feeling_sleepy',
        group='osa_four',
        covariates=['age', 'bmi', 'gender'],
    )

    # %% apnea model
    import statsmodels.api as sm
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


    model_results = fit_poisson_apnea_model(
        df=df,
        apnea_count_var='resp-oa-total',
        covariates=['age', 'bmi', 'gender',]
    )
    model_results.summary()