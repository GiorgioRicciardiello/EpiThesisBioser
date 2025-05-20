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
     # %% define the test we want toe valaute

    paired_tests_dict = {
        'Pre-Post Sleepiness': {
            'variable': ('presleep_feel_sleepy_now', 'postsleep_feeling_sleepy'),
            'interpretation': 'Change in subjective sleepiness after overnight PSG',
            'type_test': 'Wilcoxon signed-rank test'
        },
        'Pre-Post Alertness': {
            'variable': ('presleep_feel_alert_now', 'postsleep_feeling_alert'),
            'interpretation': 'Change in subjective alertness following PSG night',
            'type_test': 'Wilcoxon signed-rank test'
        },
        'Pre-Post Fatigue': {
            'variable': ('presleep_feel_tired_now', 'postsleep_feeling_tired'),
            'interpretation': 'Change in self-reported fatigue from pre- to post-PSG',
            'type_test': 'Wilcoxon signed-rank test'
        },
        # 'Pre-Post Sleep Quality': {
        #     'variable': ('presleep_physical_complaints_today', 'postsleep_last_night_sleep_quality'),
        #     'interpretation': 'Is self-rated sleep quality better?',
        #     'type_test': 'Wilcoxon signed-rank test'  # Non-normal, stepped Q-Q plot
        # },
    }
    osa_group_col = 'osa_four'
    # %% Visualization
    qqplot_pre_post_pairs_combined(df,
                          paired_tests_dict,
                          save_dir=path_output.joinpath('qqplot_pre_post_pairs_combined.png')
                                   )
    # Based on the Q-Q plots evaluating the normality of within-subject differences between pre- and post-sleep responses, we can draw several conclusions regarding the appropriate statistical tests for each symptom domain.
    #
    # The difference in subjective sleepiness before and after the PSG night appears to follow a near-normal distribution. The Q-Q plot shows the data points closely aligned with the theoretical quantile line, suggesting that the assumption of normality holds. Therefore, it would be statistically appropriate to apply a paired t-test to evaluate changes in sleepiness. However, a Wilcoxon signed-rank test could also be used as a non-parametric alternative with minimal power loss in this case.
    #
    # Similarly, the alertness comparison exhibits only minor deviations from the diagonal line, indicating approximate normality. Although slight asymmetry is noted in the distribution tails, the overall pattern supports the use of a paired t-test. As with sleepiness, the Wilcoxon test remains a valid fallback if a more conservative approach is desired.
    #
    # The fatigue comparison demonstrates an almost perfect alignment of the data with the theoretical quantiles. This strong evidence of normality supports the use of a paired t-test as the optimal method for evaluating within-subject changes in fatigue.
    #
    # In contrast, the comparison of self-reported sleep quality shows clear deviations from normality. The Q-Q plot reveals a stepped distribution and noticeable divergence from the diagonal line, suggesting that the data are not normally distributed. As such, a Wilcoxon signed-rank test is the more appropriate choice for this variable, providing a robust and non-parametric alternative to the t-test.
    #
    # In summary, while most symptom domains show evidence supporting the use of parametric methods, subjective sleep quality exhibits non-normal characteristics that warrant a non-parametric approach. These insights allow for a tailored statistical strategy that respects the underlying distribution of the data and aligns with best practices in sleep medicine research.

    # %% Perform teh test
    results_stratified = apply_within_pairs_from_dict(
        df=df,
        paired_tests_dict=paired_tests_dict,
        pair_id='id_subject',
        group_col=osa_group_col  # e.g., None, Mild, Moderate, Severe
    )
    results_stratified.to_csv(path_output.joinpath('within_stats_results.csv'))

    # In the paired analysis of subjective sleepiness (n = 26 796), we observed a statistically significant reduction in post‐sleep sleepiness compared with pre‐sleep levels (t = 8.25, p ≈ 1.6 × 10⁻¹⁶, FDR-adjusted p ≈ 2.15 × 10⁻¹⁶). However, the Cohen’s d for this change was only 0.05—an effect size so small that, despite the enormous sample, it is unlikely to reflect any meaningful clinical improvement in how sleepy participants felt after the PSG.
    #
    # For alertness (n = 27 088), there was also a tiny but statistically significant decrease after sleep (t = 3.53, p = 4.2 × 10⁻⁴, adjusted p = 4.2 × 10⁻⁴), with Cohen’s d around 0.02. In practical terms, this negligible effect size indicates essentially no real change in self‐reported alertness resulting from the sleep study.
    #
    # When examining fatigue (n = 27 243), participants actually reported a small but significant increase in tiredness following the PSG (t = –18.56, p ≈ 2 × 10⁻⁷⁶, adjusted p ≈ 4 × 10⁻⁷⁶). The effect size here (d ≈ 0.11) remains in the “small” range, suggesting the PSG environment may have induced a modest uptick in fatigue, but again not a large-scale shift.
    #
    # In contrast, the change in self‐rated sleep quality (n = 27 099) was both highly significant and substantively large: the Wilcoxon signed-rank test yielded W = 12 606 148.5 (p < 10⁻³⁰⁰, adjusted p ≈ 0) with an effect size r ≈ 0.81. This indicates a pronounced decline in perceived sleep quality during the PSG compared to participants’ usual expectations—by far the most meaningful pre→post difference in our set of measures.


    # %%
    mixedlm, model_result = fit_osa_severity_mixedlm(
        df,
        pre_col='presleep_feel_sleepy_now',
        post_col='postsleep_feeling_sleepy',
        subject_col='id',
        severity_col='osa_four',
        output_path=path_output.joinpath('mixedlm_summary.csv')
    )
    model_result.reset_index(inplace=True, drop=False, names=['variable'])

    rename_dict = {
        "Intercept": "Baseline (Post, Severe OSA)",
        "time[T.pre]": "Pre vs Post (Severe OSA)",
        "osa_four[T.Moderate]": "Moderate vs Severe OSA (Post)",
        "osa_four[T.Normal]": "Normal vs Severe OSA (Post)",
        "osa_four[T.Severe]": "Severe vs Severe OSA (Post)",  # redundant but included for completeness
        "time[T.pre]:osa_four[T.Moderate]": "Interaction: Pre vs Post × Moderate OSA",
        "time[T.pre]:osa_four[T.Normal]": "Interaction: Pre vs Post × Normal OSA",
        "time[T.pre]:osa_four[T.Severe]": "Interaction: Pre vs Post × Severe OSA",
        "Group Var": "Random Intercept Variance",
        "Group x time[T.pre] Cov": "Random Slope-Intercept Covariance",
        "time[T.pre] Var": "Random Slope Variance",
        "LogLikelihood": "Log-Likelihood"
    }
    model_result['variable'] = model_result['variable'].map(rename_dict)

    # The mixed‐effects model tests whether the change in sleepiness from pre→post differs by OSA severity, while accounting for within‐subject correlation. Here’s how to read the key rows:
    #
    # **1. Main “time” effect (time[T.presleep_feel_sleepy_now]):**
    # - **OR = 1.099 (95% CI: 1.077–1.121), p ≈ 8.3 × 10⁻²⁰**
    # - Interpretation: Overall, there is a modest increase in the odds of higher “score” (i.e. greater sleepiness) post-sleep versus pre-sleep. This aligns with the small but significant increase in fatigue we saw previously.
    #
    # **2. Main OSA severity effects:**
    # - **Moderate vs. Mild:** OR = 0.943 (CI 0.916–0.971), p = 9.0 × 10⁻⁵
    # - **Severe vs. Mild:**   OR = 0.952 (CI 0.922–0.982), p = 2.1 × 10⁻³
    # - Interpretation: At baseline (“pre”), patients with Moderate or Severe OSA report slightly lower sleepiness scores than those with Mild OSA. These are small but statistically significant differences.
    #
    # **3. Time × Severity interactions:**
    # - **Moderate × Time:** OR = 1.031 (CI 0.995–1.067), p = 0.089
    # - **Severe × Time:**   (not shown; presumably similar/missing)
    # - Interpretation: The p‐value for the interaction term is ~0.09, which does *not* reach conventional significance. In other words, there is no strong evidence that the pre→post increase in sleepiness differs between Mild and Moderate (or Severe) groups. All severity strata appear to exhibit a similar post-sleep shift.
    #
    # **4. Log‐likelihood:**
    # - Although not row‐aligned, the model’s log‐likelihood (≈ –…) indicates overall fit; you’d normally compare it to a reduced model if you wanted a formal likelihood‐ratio test of the interaction.
    #
    # ---
    #
    # **Conclusion for your hypothesis (“does the pre→post change vary by OSA severity?”):**
    # The non-significant interaction (p ≈ 0.09) suggests that, while OSA severity influences baseline sleepiness, it does *not* meaningfully alter the within‐subject increase in sleepiness after PSG. Thus, the modest post‐sleep effect on sleepiness appears consistent across Mild, Moderate, and Severe OSA groups.
    # scoreᵢⱼ = β₀ + β₁·postᵢⱼ + β₂·Moderateᵢ + β₃·Severeᵢ + β₄·(postᵢⱼ×Moderateᵢ) + u₀ᵢ + u₁ᵢ·postᵢⱼ + εᵢⱼ
    # In this linear mixed‐effects specification—
    # ```
    # scoreᵢⱼ = β₀ + β₁·postᵢⱼ + β₂·Moderateᵢ + β₃·Severeᵢ + β₄·(postᵢⱼ×Moderateᵢ) + u₀ᵢ + u₁ᵢ·postᵢⱼ + εᵢⱼ
    # ```
    # (with random intercepts *u₀ᵢ* and slopes *u₁ᵢ* for each subject), the fixed‐effects tell us: the baseline (pre‐sleep, Mild‐OSA reference) odds of reporting higher sleepiness correspond to an OR of **2.948** (SE = 0.0087; 95 % CI 2.898–2.999; *p* < 1×10⁻³⁰⁰). The main effect of time (post vs. pre) yields **OR = 1.099** (SE = 0.0103; 95 % CI 1.077–1.121; *p* ≈ 8.3×10⁻²⁰), indicating a significant 9.9 % increase in the odds of greater sleepiness after the PSG. Compared with Mild, Moderate OSA has a slightly lower baseline sleepiness odds (**OR = 0.943**, SE = 0.0149; 95 % CI 0.916–0.971; *p* ≈ 9.0×10⁻⁵) and Severe likewise (**OR = 0.952**, SE = 0.0160; 95 % CI 0.922–0.982; *p* ≈ 2.1×10⁻³). Crucially, the time×Moderate interaction (**OR = 1.031**, SE = 0.0177; 95 % CI 0.995–1.067; *p* = 0.089) fails to reach conventional significance, implying that the post‐sleep increase in sleepiness does *not* differ meaningfully between Mild and Moderate (or Severe) groups. In sum, while OSA severity shifts baseline sleepiness, the within‐subject post‐sleep effect is statistically homogeneous across all severity strata.

    # %% shankey data
    import plotly.graph_objects as go

    paired_tests_dict = {
        'Sleepiness': ('presleep_feel_sleepy_now', 'postsleep_feeling_sleepy'),
        'Alertness': ('presleep_feel_alert_now', 'postsleep_feeling_alert'),
        'Fatigue': ('presleep_feel_tired_now', 'postsleep_feeling_tired'),
    }

    encoding.get('presleep_feel_sleepy_now')['encoding']


    def save_sankey_plot_by_osa(
            df: pd.DataFrame,
            pre_col: str,
            post_col: str,
            label: str,
            osa_group_col: str,
            encoding_map: dict,
            output_path: Optional[pathlib.Path] = None
    ):
        from plotly.subplots import make_subplots

        unique_osa_groups = df[osa_group_col].dropna().unique()
        unique_osa_groups = sorted(unique_osa_groups, key=lambda x: str(x))  # optional: sort categories

        fig = make_subplots(
            rows=1,
            cols=len(unique_osa_groups),
            subplot_titles=[f"OSA: {g}" for g in unique_osa_groups],
            specs=[[{"type": "domain"}] * len(unique_osa_groups)]
        )

        for i, osa_group in enumerate(unique_osa_groups):
            df_sub = df[df[osa_group_col] == osa_group][[pre_col, post_col]].dropna().copy()
            df_sub.columns = ['pre', 'post']
            total = len(df_sub)
            if total == 0:
                continue

            # transition matrix
            transitions = df_sub.groupby(['pre', 'post']).size().reset_index(name='count')
            transitions['percentage'] = transitions['count'] / total * 100

            # map to labels
            transitions['pre_lbl'] = transitions['pre'].map(encoding_map)
            transitions['post_lbl'] = transitions['post'].map(encoding_map)

            sources, targets, values, labels = [], [], [], []
            label_map = {}

            def add_node(name):
                if name not in label_map:
                    label_map[name] = len(label_map)
                    labels.append(name)
                return label_map[name]

            for _, row in transitions.iterrows():
                s_name = f"Pre: {row['pre_lbl']}"
                t_name = f"Post: {row['post_lbl']}"
                s_idx = add_node(s_name)
                t_idx = add_node(t_name)
                sources.append(s_idx)
                targets.append(t_idx)
                values.append(row['count'])

            sankey = go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=labels
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    customdata=[f"{v / total * 100:.1f}%" for v in values],
                    hovertemplate='%{source.label} → %{target.label}<br>%{customdata} of {total} pts<extra></extra>'
                )
            )

            fig.add_trace(sankey, row=1, col=i + 1)

        fig.update_layout(
            title_text=f"{label}: Pre vs Post Response Transitions by OSA Severity",
            font_size=10,
            height=400,
            width=300 * len(unique_osa_groups)
        )

        if output_path:
            fig.write_image(output_path.joinpath(f"sankey_{label.lower().replace(' ', '_')}_by_osa.png"))

        fig.show()

    def get_transition_table(data:pd.DataFrame,
                             pre_col:str,
                             post_col:str,
                             osa_group_col:str,
                             encoding_map):
        data = data[[pre_col, post_col, osa_group_col]].dropna().copy()
        data['pre_lbl'] = data[pre_col].map(encoding_map)
        data['post_lbl'] = data[post_col].map(encoding_map)

        table = data.groupby([osa_group_col, 'pre_lbl', 'post_lbl']).size().reset_index(name='count')
        table['total_group'] = table.groupby(osa_group_col)['count'].transform('sum')
        table['percent'] = table['count'] / table['total_group'] * 100
        return table


    def get_transition_matrix(
            df: pd.DataFrame,
            pre_col: str,
            post_col: str,
            osa_group_col: str,
            encoding_map: dict,
            percent: bool = False
    ) -> dict:
        """
        Returns a dict of confusion-matrix-style DataFrames for each OSA group,
        showing transitions from pre to post with either counts or counts + percentages.

        Parameters
        ----------
        df : pd.DataFrame
        pre_col : str
        post_col : str
        osa_group_col : str
        encoding_map : dict (e.g., {0: "None", 1: "A little", ...})
        percent : bool
            If True, each cell shows 'count (xx.x%)' format. If False, only count.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys are OSA group names; values are transition matrices.
        """
        df = df[[pre_col, post_col, osa_group_col]].dropna().copy()
        df['pre_lbl'] = df[pre_col].map(encoding_map)
        df['post_lbl'] = df[post_col].map(encoding_map)

        pre_levels = list(encoding_map.values())
        post_levels = list(encoding_map.values())

        matrices = {}
        for group, group_df in df.groupby(osa_group_col):
            counts = pd.crosstab(
                index=group_df['pre_lbl'],
                columns=group_df['post_lbl'],
                rownames=['Pre'],
                colnames=['Post'],
                dropna=False
            ).reindex(index=pre_levels, columns=post_levels, fill_value=0)

            if percent:
                row_totals = counts.sum(axis=1)
                percent_matrix = counts.div(row_totals, axis=0).fillna(0) * 100
                combined = counts.astype(str) + " (" + percent_matrix.round(1).astype(str) + "%)"
                matrices[group] = combined
            else:
                matrices[group] = counts

        return matrices


    import matplotlib.pyplot as plt
    import seaborn as sns

    import math
    import matplotlib.pyplot as plt
    import seaborn as sns


    def plot_transition_matrix_grid(
            tabs: dict[str, dict[str, pd.DataFrame]],
            encoding_map: dict,
            osa_order: list = ['Normal', 'Mild', 'Moderate', 'Severe'],
            cmap: str = 'Blues',
            font_size: int = 9,
            n_cols: int = 2,
            figsize_per_plot: tuple = (5, 4)
    ):
        """
        Plot transition matrix heatmaps for each test label separately.
        Each figure has a fixed number of columns and automatic row adjustment.

        Parameters
        ----------
        tabs : dict[label -> osa_group -> pd.DataFrame]
        encoding_map : dict[int -> str]
        osa_order : list[str]
            The order of OSA severity groups to be shown in columns.
        cmap : str
            Colormap to use for the heatmaps.
        font_size : int
        n_cols : int
            Number of columns per figure (subplot grid).
        figsize_per_plot : tuple
            Size of each individual subplot (width, height).
        """

        def wrap_ticklabels(labels, width=10):
            return ['\n'.join(label.split(' ', 1)) if len(label) > width else label for label in labels]

        # Sort encoding values
        ordered_labels = [k for k, _ in sorted(encoding_map.items(), key=lambda x: x[1])]

        for label, matrices_dict in tabs.items():
            n_plots = len(osa_order)
            n_rows = math.ceil(n_plots / n_cols)

            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(figsize_per_plot))
            axes = axes.flatten()

            for i, osa_group in enumerate(osa_order):
                ax = axes[i]
                matrix = matrices_dict.get(osa_group)

                if matrix is None or matrix.empty:
                    ax.axis('off')
                    continue

                # Ensure consistent order
                matrix = matrix.reindex(index=ordered_labels, columns=ordered_labels, fill_value=0)

                # Detect formatting type
                if isinstance(matrix.iloc[0, 0], str):
                    heat_vals = matrix.replace(r'\s*\(.*?\)', '', regex=True).astype(float)
                    annot_vals = matrix
                else:
                    heat_vals = matrix
                    annot_vals = matrix.replace(' ', '\n')

                sns.heatmap(
                    heat_vals,
                    annot=annot_vals,
                    fmt='s',
                    cmap=cmap,
                    cbar=False,
                    ax=ax,
                    annot_kws={"size": font_size - 1, "weight": "bold"},
                    linewidths=0.5,
                    linecolor='gray'
                )

                ax.set_title(f"OSA: {osa_group}", fontsize=font_size + 1)
                ax.set_xlabel("Post", fontsize=font_size)
                ax.set_ylabel("Pre", fontsize=font_size)
                xticks_wrapped = wrap_ticklabels([t.get_text() for t in ax.get_xticklabels()])
                yticks_wrapped = wrap_ticklabels([t.get_text() for t in ax.get_yticklabels()])

                ax.set_xticklabels(xticks_wrapped, rotation=30, ha='right', fontsize=font_size)
                ax.set_yticklabels(yticks_wrapped, rotation=0, fontsize=font_size)

            # Hide unused subplots
            for j in range(len(osa_order), len(axes)):
                axes[j].axis("off")

            fig.suptitle(f"{label}: Pre vs Post Transitions by OSA Severity", fontsize=font_size + 3)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


    # Generate and save plots
    tabs = {}
    for label, (pre_col, post_col) in paired_tests_dict.items():
        encoding_map = {val: key for key, val in encoding.get(pre_col)['encoding'].items()}
        # save_sankey_plot_by_osa(
        #     df=df,
        #     pre_col=pre_col,
        #     post_col=post_col,
        #     label=label,
        #     osa_group_col=osa_group_col,
        #     encoding_map=encoding.get(pre_col)['encoding'],  # assumes same for pre/post
        #     output_path=None
        # )
        # tabs[label] = get_transition_table(data=df,
        #                                    pre_col=pre_col,
        #                                    post_col=post_col,
        #                                    osa_group_col=osa_group_col,
        #                                    encoding_map=encoding_map)

        tabs[label] = get_transition_matrix(df=df,
                                            percent=True,
                                           pre_col=pre_col,
                                           post_col=post_col,
                                           osa_group_col=osa_group_col,
                                           encoding_map=encoding_map)

    plot_transition_matrix_grid(
        tabs=tabs,
        encoding_map=encoding.get('presleep_feel_sleepy_now')['encoding'],
        n_cols=2,  # try 2 or 3 for better layout
        font_size=9,
        figsize_per_plot=(10, 9)
    )