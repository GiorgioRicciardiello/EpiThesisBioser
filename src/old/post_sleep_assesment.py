"""
Background: Subjective sleep quality (postsleep perceptions) is widely used in clinical research, but its relationship to objective respiratory‐event burden (e.g., total apneas+hypopneas) remains unclear.

Aim: Quantify how well postsleep measures capture total respiratory events, beyond just AHI severity.
"""
import pathlib
from config.config import config, encoding
import pandas  as pd
from library.TableOne.table_one import MakeTableOne
from library.stat_tests.post_sleep_methods import (plot_ahi_split_by_sex,
                                                   plot_corr_matrix,
                                                    compute_corr_matrices,
                                                   build_regression_summary)
# from library.ml_tabular_data.my_simple_xgb import train_xgb_regressor_with_optuna_and_cv, plot_xgb_feature_imp


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



if __name__ == '__main__':
    # %% Input data
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    path_output = config.get('results')['post_sleep_stats']
    # %%  post sleep questions
    cols_adjusted = ['age', 'gender', 'bmi', 'race']
    cols_post_sleep_all = [col for col in df.columns if col.startswith("postsleep")]
    cols_post_sleep_all_adjusted = cols_post_sleep_all + cols_adjusted

    cols_post_sleep_feeling = ['postsleep_feeling_sleepy',
                               'postsleep_feeling_alert',
                               'postsleep_feeling_tired']

    metrics_resp = [col for col in df.columns if col.startswith("resp")]
    cols_post_sleep_feeling_adjusted = cols_post_sleep_feeling + cols_adjusted

    # %% respiratory events
    target  = 'ahi'  # same as resp-ahi_no_reras-total
    events = metrics_psg.get('raw_events')
    event = events[0]
    pos = 'total'
    t_col = 'resp-position-total'  #  metrics_psg.get('resp_events')['position_keywords']
    cnt_col = f"resp-{event}-{pos}"  #  'resp-oa-total'
    target_candidates = [f"resp-{event}-{pos}" for event in events]
    df['sleep_hours'] = df[t_col] / 60.0


    # %% Table One of the post sleep events clustered by ahi severity
    vars_bin = [var for var in cols_post_sleep_all if df[var].max() <= 1]
    vars_ordinal = [var for var in cols_post_sleep_all if df[var].nunique() <= 10 and not var in vars_bin]
    vars_continuous = [var for var in cols_post_sleep_all if not (var in vars_ordinal or var in vars_bin)]
    vars_categorical = vars_ordinal + vars_bin
    tab_post_sleep = MakeTableOne(df=df[~df['osa_four'].isna()],
                                  continuous_var=vars_continuous,
                                  categorical_var=vars_categorical,
                                  strata='osa_four')
    df_post_sleep = tab_post_sleep.create_table()
    df_post_sleep.to_csv(path_output / 'table_one_post_sleep.csv', index=False)

    # %% Descriptive & Distribution Check
    encoding_map = {val: key for key, val in encoding.get(cols_post_sleep[0])['encoding'].items()}
    plot_post_sleep_stacked_by_gender(
        df=df,
        outcome='postsleep_feeling_sleepy',
        group='osa_four',
        gender_col='gender',
        encoding_map=encoding_map,
        encoding_order=['Normal', 'Mild', 'Moderate', 'Severe']
    )
    # Histograms of each rate
    # for event in events:
    #     plt.figure()
    #     df[f"{event}_rate_per_hr"].dropna().hist(bins=30)
    #     plt.title(f"{event.upper()} rate (events/hr)")
    #     plt.xlabel("Events per hour")
    #     plt.ylabel("Frequency")
    #     plt.tight_layout()
    #     plt.show()
    #
    # for event in events:
    #     rate_col = f"{event}_rate_per_hr"
    #     data = df[rate_col].dropna()
    #     x_max = data.quantile(0.99)  # trim outliers for display
    #
    #     plt.figure(figsize=(6, 4))
    #     plt.hist(data, bins=100, density=True)  # density=True normalizes
    #     plt.xlim(0, x_max)
    #     plt.xlabel("Events per hour")
    #     plt.ylabel("Density")
    #     plt.title(f"{event.upper()} rate (events/hr)")
    #     plt.tight_layout()
    #     plt.show()
    # 3) Loop through each ordinal variable and plot subplots nobsplots in relation to the target
    plot_ahi_split_by_sex(df=df,
                          y_continuous=target,
                          x_ordinal=vars_categorical,
                          group_col='gender',
                          output_path=path_output / 'post_sleep_ahi_age_split.png')

    # %% Descriptive with the distribution of postsleep measures
    # 1) Descriptive
    # for var in post_vars:
    #     plt.figure()
    #     df[var].dropna().hist(bins=30)
    #     plt.title(f"{var.upper()} distribution")
    #     plt.xlabel(var.upper())
    #     plt.ylabel("Frequency")
    #     plt.tight_layout()
    #     plt.show()
    # %% Correlation of Event-Rates with Postsleep Measures


    corr_bin, corr_ord = compute_corr_matrices(df, vars_bin, vars_ordinal)
    plot_corr_matrix(corr=corr_bin, title="Binary Correlations", ouput_path=path_output)
    plot_corr_matrix(corr=corr_ord, title="Ordinal Correlations",ouput_path=path_output)


    #%% Using a regression model
    # Define vars_bin, vars_ord, vars_cont
    exposures = vars_bin + vars_ordinal + vars_continuous
    # df['sleep_hours'] = df['resp-position-total'] / 60.0
    # df['ahi'] = (df['resp-ai_apneas_only-total'] + df['resp-hi_hypopneas_only-total']) / df['sleep_hours']
    df_post_sleep_reg = build_regression_summary(df=df,
                                       exposures=exposures,
                                       outcome='ahi_logp1_boxcox',
                                       adjust_vars=['age','race','gender'],
                                       cat_exposures=vars_ordinal)
    print(df_post_sleep_reg)

    # treat the ordinal as continous
    df_post_sleep_reg_simplified = build_regression_summary(df=df,
                                       exposures=[var for var in exposures if var != 'race'],
                                       outcome='ahi', # 'ahi_logp1_boxcox',
                                       adjust_vars=['age','race','gender'],
                                       cat_exposures=vars_bin + ['race'] )
    print(df_post_sleep_reg_simplified)
    # rename exposure and the levels
    def rename_exposures_levels(
            df: pd.DataFrame,
            all_map: dict,
            exposure_col: str = 'exposure',
            level_col: str = 'level'
    ) -> pd.DataFrame:
        """
        Rename exposure variable names to their definitions and map numeric levels
        back to their original string labels based on `all_map` encoding.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with columns for exposure names and level codes.
        all_map : dict
            Dictionary mapping exposure keys to metadata, including:
              - 'definition': human-readable variable name
              - 'encoding': dict mapping label->code
        exposure_col : str
            Name of the column in df containing exposure variable keys.
        level_col : str
            Name of the column in df containing the numeric level codes.

        Returns
        -------
        pd.DataFrame
            A copy of df with:
              - `exposure_col` replaced by human-friendly definitions
              - `level_col` replaced by the original string labels
        """
        # Copy to avoid modifying original
        df_ren = df.copy()

        # # Map each exposure key to its 'definition'; fallback to the key if missing
        # df_ren[exposure_col] = df_ren[exposure_col].map(
        #     lambda key: all_map.get(key, {}).get('definition', key)
        # )

        # Build reverse-encoding: for each key, map code->label
        rev_encoding = {
            key: {code: label for label, code in info.get('encoding', {}).items()}
            for key, info in all_map.items()
        }

        # Function to map level code back to its label string
        def _map_level(row):
            definition = row[exposure_col]
            lvl = row[level_col]
            lvl = lvl.replace(']', "")
            # Find the original key corresponding to this definition
            key = next(
                (k for k, info in all_map.items() if info.get('definition') == definition),
                None
            )
            if key and key in rev_encoding:
                # Attempt integer conversion for lookup
                try:
                    code = int(lvl)
                except (ValueError, TypeError):
                    return lvl
                # Return mapped label or original lvl
                return rev_encoding[key].get(code, code)
            return lvl

        # Apply level mapping
        df_ren[level_col] = df_ren.apply(_map_level, axis=1)

        return df_ren

    df_simplified = rename_exposures_levels(df_post_sleep_reg_simplified, all_map)

    # %% Machine Learning Models
    # 2) Select postsleep predictors and target
    postsleep_cols = [c for c in df.columns if c.startswith('postsleep')]
    df_model = df.loc[~df['ahi'].isna(), postsleep_cols + ['ahi'] + ['osa_four']].copy()
    X = df_model[postsleep_cols].copy()
    y = df_model['ahi'].copy()

    result = train_xgb_regressor_with_optuna_and_cv(
        df=df_model,
        feature_cols=postsleep_cols,
        target_col='ahi',
        optimization=False,
        n_trials=30,
        cv_folds=5,
        use_gpu=True,
        stratify_col='osa_four',
    )


    import matplotlib.pyplot as plt


    def create_alias(name):
        # Split the name by underscores
        parts = name.split('_')
        # Ignore the first substring (e.g., 'postsleep')
        if len(parts) > 1:
            parts = parts[1:]
        # Filter out words like 'have' and join first letters of remaining words
        filtered_parts = [part[0].upper() for part in parts if part.lower() not in ['have']]
        return ''.join(filtered_parts)


    result['feature_importance_df']['alias'] = result['feature_importance_df']['feature'].apply(create_alias)


    plot_xgb_feature_imp(fi_df=result['feature_importance_df'],
                                         height_prop=0.8,
                                         output_path=None)

    # now we want to see the association with the output
    import shap

    # assuming your trained model is `m` and your DataFrame is `df`
    m = result['model']
    explainer = shap.TreeExplainer(m)
    shap_vals = explainer.shap_values(df[feature_cols])

    # dependence plot for feature 'X'
    shap.dependence_plot('X', shap_vals, df[feature_cols])

    # Partial dependicy plots
    from sklearn.inspection import PartialDependenceDisplay

    # Assuming final_model, X, and postsleep_cols are defined in your environment

    # Function to create more readable feature names
    def format_feature_name(name):
        # Remove 'postsleep_' prefix and replace underscores with spaces
        if name.startswith('postsleep_'):
            name = name[10:]
        return name.replace('_', ' ').title()


    # Calculate grid layout
    n_features = len(postsleep_cols)
    cols = 4  # Number of columns for subplots
    rows = (n_features + cols - 1) // cols  # Calculate needed rows

    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(24, 6 * rows))
    axes = axes.flatten()  # Flatten for easier indexing
    # palette = plt.cm.get_cmap("Accent", len(postsleep_cols))
    # Plot each feature in its own subplot
    for i, feature in enumerate(postsleep_cols):
        # colour = palette(i)
        PartialDependenceDisplay.from_estimator(
            estimator=m,
            X=X,
            kind='average',
            features=[feature],  # One feature per subplot
            subsample=0.9,
            is_categorical=True,
            grid_resolution=10,  # Better resolution without being excessive
            ax=axes[i],
            line_kw={"color": "blue", "linewidth": 2},
            # # ICE lines in a faint version of the same colour
            # ice_lines_kw = {"color": colour, "alpha": 0.2, "linewidth": 1},
            # # PD line in bold
            # pd_line_kw  = {"color": colour, "linewidth": 3, "label": None},
        )
        # Set better title with formatted feature name
        axes[i].set_title(f"{format_feature_name(feature)}")
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)
        # Add grid for better readability
        axes[i].grid(True, linestyle='--', alpha=0.7)

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Add overall title
    plt.suptitle("Partial Dependence Plots for AHI", fontsize=16)
    # Use tight_layout with rect to ensure the title isn't cut off
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    # These plots are showing you exactly how your model’s predicted AHI (apnea–hypopnea index) changes as you vary each individual “postsleep” questionnaire response, while holding all other inputs fixed. In more detail:
    #
    # Individual vs. average effects
    #
    # The many faint blue lines in each panel are ICE curves (“individual conditional expectation”) for a random subsample of subjects. They show how the prediction for each subject would move if you change just that one feature.
    #
    # The solid, darker blue line is the average of those curves—the true partial dependence—which tells you the marginal effect of that feature on the predicted AHI, averaged over your data.
    #
    # Interpreting the slope
    #
    # Wherever the average curve slopes upward, increasing the feature value tends to raise the predicted AHI.
    #
    # Wherever it slopes downward, increasing the feature value tends to lower the predicted AHI.
    #
    # A nearly flat curve means that feature has very little influence on the model’s AHI prediction.
    #
    # Examples from your plots
    #
    # “Have difficulty falling asleep” (binary 0→1): the average curve jumps up by a few points when you go from “no difficulty” to “yes,” indicating that, all else equal, reporting difficulty falling asleep is associated with a higher predicted AHI.
    #
    # “Slept longer” (binary 0→1): you see a slight downward shift, suggesting that saying you slept longer is modestly predictive of a lower AHI.
    #
    # “Feeling refreshing” (ordinal 0→3): there’s a noticeable drop around level 2, telling you that subjects who report feeling more refreshed tend to have substantially lower predicted AHI.
    #
    # “Dreaming last night”, “Feeling sick”, etc., with nearly flat curves, show that those questions didn’t move the model’s AHI prediction much at all.
    #
    # Heterogeneity
    #
    # The spread of the ICE curves around the average line gives you a sense of how consistent that effect is across individuals. A tight bundle means the effect is uniform; a wide spread means some people’s predictions respond more strongly than others’.

