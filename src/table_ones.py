#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title: Adjacent-Group Effect Size Analysis for OSA Questionnaire Data

Description:
    This script performs nonparametric adjacent-group comparisons on continuous
    and ordinal questionnaire variables stratified by OSA severity. It computes
    Mann–Whitney U tests, Cliff’s Δ (rank-biserial correlation), and 95% bootstrap
    confidence intervals (Holm–Bonferroni–adjusted p-values), produces “Table 1”
    with descriptive statistics, and generates forest plots of effect sizes for:
      1) All continuous/ordinal variables (effect_measure_tab_one.png)
      2) Sleep-assessment subsection only (effect_measure_sa.png)

Usage:
    python forest_adjacency_plot.py

Inputs (via config):
    - Questionnaire responses CSV (config.data.pp_data.q_resp)
    - Encoding and PSG metrics definitions (config.config, metrics_psg)

Outputs:
    - table_one_all.xlsx
    - effect_measure_tab_one.csv, effect_measure_tab_one.png
    - effect_measure_sa.csv, effect_measure_sa.png

Author: Giorgio Ricciardiello
Date: 2025-05-08
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.config import config, encoding, metrics_psg
from library.TableOne.table_one import MakeTableOne
import textwrap
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import pathlib
from typing import Dict
import seaborn as sns

class ForestAdjacencyPlot:
    def __init__(
            self,
            df: pd.DataFrame,
            variable_col: str = "variable",
            effect_col: str = "cliffs_delta",
            lcl_col: str = "ci_95%_lower",
            ucl_col: str = "ci_95%_upper",
            comp1_col: str = "group1",
            comp2_col: str = "group2",
            cmap_name: str = "tab10",
            output_path: pathlib.Path = None ,
    ):
        """
        df must contain one row per adjacent‐group comparison with:
          – variable (str)
          – cliffs_delta (float)
          – ci_95%_lower (float)
          – ci_95%_upper (float)
          – group1, group2 (labels for the comparison)
        """
        self.df = df.copy()
        self.vcol, self.ecol, self.lcol, self.ucol = (
            variable_col, effect_col, lcl_col, ucl_col
        )
        self.c1, self.c2 = comp1_col, comp2_col

        # Build a single “comparison” label and compute error‐bar widths
        self.df["comparison"] = self.df[self.c1] + " vs " + self.df[self.c2]
        self.df["err_low"] = (self.df[self.ecol] - self.df[self.lcol]).abs()
        self.df["err_high"] = (self.df[self.ucol] - self.df[self.ecol]).abs()

        # y-positions for each variable
        uniques = list(self.df[self.vcol].unique())
        self.y_pos = {v: i for i, v in enumerate(uniques)}
        self.variables = uniques

        # Color‐map: one distinct color per comparison label
        comparisons = self.df["comparison"].unique()
        cmap = plt.get_cmap(cmap_name)
        self.color_map = {cmp_label: cmap(i) for i, cmp_label in enumerate(comparisons)}
        self.output_path = output_path

    def _check_significance(self):
        pass

    # def plot(
    #         self,
    #         figsize: tuple = (8, None),
    #         marker_size: int = 60,
    #         ci_linewidth: float = 1.2,
    #         zero_line: bool = True,
    #         xlabel: str = "Cliff’s Δ (rank-biserial)",
    #         wrap_width: int = 25,
    # ):
    #     """
    #     Draws a forest-plot with one horizontal error bar + dot per adjacent
    #     comparison, one y-tick per variable, wrapped at wrap_width.
    #     """
    #     # Auto-scale height if not fixed
    #     if figsize[1] is None:
    #         figsize = (figsize[0], len(self.variables) * 0.6 + 1)
    #
    #     fig, ax = plt.subplots(figsize=figsize)
    #
    #     # vertical zero line
    #     if zero_line:
    #         ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    #
    #     # plot each row with its assigned color
    #     for _, row in self.df.iterrows():
    #         y = self.y_pos[row[self.vcol]]
    #         x = row[self.ecol]
    #         err = [[row["err_low"]], [row["err_high"]]]
    #         cmp = row["comparison"]
    #         col = self.color_map[cmp]
    #
    #         ax.errorbar(
    #             x, y,
    #             xerr=err,
    #             fmt="none",
    #             ecolor=col,
    #             elinewidth=ci_linewidth,
    #             capsize=3,
    #             zorder=1,
    #             alpha=0.7
    #         )
    #         ax.scatter(
    #             x, y,
    #             s=marker_size,
    #             color=col,
    #             edgecolors="none",
    #             zorder=2,
    #             label=cmp,
    #             alpha=0.7,
    #         )
    #
    #     # wrap long variable names
    #     wrapped = [textwrap.fill(v, wrap_width) for v in self.variables]
    #
    #     # y-axis formatting
    #     ax.set_yticks(list(self.y_pos.values()))
    #     ax.set_yticklabels(wrapped)
    #     ax.invert_yaxis()
    #     ax.set_xlabel(xlabel)
    #     ax.set_ylabel("Variable")
    #
    #     # legend (one entry per comparison)
    #     handles, labels = ax.get_legend_handles_labels()
    #     unique = dict(zip(labels, handles))
    #     ax.legend(
    #         unique.values(),
    #         unique.keys(),
    #         title="Comparisons",
    #         bbox_to_anchor=(1.02, 1),
    #         loc="upper left",
    #         frameon=False
    #     )
    #     plt.grid(alpha=0.7)
    #     plt.tight_layout()
    #     if self.output_path:
    #         plt.savefig(self.output_path, dpi=300)
    #     # return fig, ax
    #     plt.show()

    def plot(
            self,
            figsize: tuple = (8, None),
            marker_size: int = 60,
            ci_linewidth: float = 1.2,
            zero_line: bool = True,
            xlabel: str = "Cliff’s Δ (rank-biserial)",
            ylabel: str = "Variable",
            wrap_width: int = 25,
            horizontal: bool = False,  # New parameter for horizontal plotting
    ):
        """
        Draws a forest-plot with one error bar + dot per adjacent comparison,
        one tick per variable, wrapped at wrap_width. If horizontal=True,
        swaps x and y axes (variables on x-axis, effect sizes on y-axis).
        """
        # Auto-scale height (or width if horizontal) if not fixed
        if figsize[1] is None:
            figsize = (figsize[0], len(self.variables) * 0.6 + 1) if not horizontal else (len(self.variables) * 0.6 + 1,
                                                                                          figsize[0])

        fig, ax = plt.subplots(figsize=figsize)

        # Zero line (vertical for vertical plot, horizontal for horizontal plot)
        if zero_line:
            if horizontal:
                ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            else:
                ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)

        # Plot each row with its assigned color
        for _, row in self.df.iterrows():
            y = self.y_pos[row[self.vcol]]
            x = row[self.ecol]
            err = [[row["err_low"]], [row["err_high"]]]
            cmp = row["comparison"]
            col = self.color_map[cmp]

            if horizontal:
                # Swap x and y for horizontal plotting
                ax.errorbar(
                    y, x,
                    yerr=err,
                    fmt="none",
                    ecolor=col,
                    elinewidth=ci_linewidth,
                    capsize=3,
                    zorder=1,
                    alpha=0.7
                )
                ax.scatter(
                    y, x,
                    s=marker_size,
                    color=col,
                    edgecolors="none",
                    zorder=2,
                    label=cmp,
                    alpha=0.7,
                )
            else:
                # Original vertical plotting
                ax.errorbar(
                    x, y,
                    xerr=err,
                    fmt="none",
                    ecolor=col,
                    elinewidth=ci_linewidth,
                    capsize=3,
                    zorder=1,
                    alpha=0.7
                )
                ax.scatter(
                    x, y,
                    s=marker_size,
                    color=col,
                    edgecolors="none",
                    zorder=2,
                    label=cmp,
                    alpha=0.7,
                )

        # Wrap long variable names
        wrapped = [textwrap.fill(v, wrap_width) for v in self.variables]

        if horizontal:
            # x-axis: variables, y-axis: effect sizes
            ax.set_xticks(list(self.y_pos.values()))
            ax.set_xticklabels(wrapped, rotation=45, ha="right")
            ax.set_ylabel(xlabel)  # Effect size label on y-axis
            ax.set_xlabel(ylabel)  # Variable label on x-axis
            ax.invert_yaxis()  # Invert y-axis to have positive effect sizes upward
        else:
            # Original y-axis: variables, x-axis: effect sizes
            ax.set_yticks(list(self.y_pos.values()))
            ax.set_yticklabels(wrapped)
            ax.invert_yaxis()
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)

        # Legend (one entry per comparison)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(
            unique.values(),
            unique.keys(),
            title="Comparisons",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            frameon=False
        )
        plt.grid(alpha=0.7)
        plt.tight_layout()
        if self.output_path:
            plt.savefig(self.output_path, dpi=300)
        plt.show()

def compute_adjacent_nonparam_pairwise(
        df: pd.DataFrame,
        cont_vars: list[str],
        group_col: str,
        group_labels: dict[int, str] = None,
        alpha: float = 0.05,
        n_boot: int = 3
) -> pd.DataFrame:
    """
    For each continuous variable in cont_vars, compare only adjacent
    numeric levels in group_col (e.g. 0 vs 1, 1 vs 2, 2 vs 3) using:
      - Mann–Whitney U (two-sided)
      - Cliff's delta (Δ)
      - Rank–biserial correlation (r_rb == Δ)
      - 95% bootstrap CI for Δ
      - Holm–Bonferroni–adjusted p-values
    """
    # 1) Determine truly adjacent numeric levels
    levels = sorted(df[group_col].dropna().unique())
    adjacent = [(levels[i], levels[i + 1]) for i in range(len(levels) - 1)]

    results = []
    for var in cont_vars:
        stats_for_var = []
        for g1, g2 in adjacent:
            x = df.loc[df[group_col] == g1, var].dropna().to_numpy()
            y = df.loc[df[group_col] == g2, var].dropna().to_numpy()
            n1, n2 = len(x), len(y)
            if n1 < 2 or n2 < 2:
                continue

            # Mann–Whitney U
            U, p = mannwhitneyu(x, y, alternative='two-sided')

            # Cliff's delta & rank–biserial (they coincide)
            more = np.sum(x[:, None] > y[None, :])
            less = np.sum(x[:, None] < y[None, :])
            delta = (more - less) / (n1 * n2)
            r_rb = delta

            # Bootstrap CI for Δ
            boot_deltas = []
            for _ in tqdm(range(n_boot), desc=f"Bootstrapping {var}"):
                xb = np.random.choice(x, size=n1, replace=True)
                yb = np.random.choice(y, size=n2, replace=True)
                m = np.sum(xb[:, None] > yb[None, :])
                l = np.sum(xb[:, None] < yb[None, :])
                boot_deltas.append((m - l) / (n1 * n2))
            ci_low, ci_high = np.percentile(boot_deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])

            stats_for_var.append({
                "variable": var,
                "group1_code": g1,
                "group2_code": g2,
                "group1": group_labels.get(g1, str(g1)) if group_labels else str(g1),
                "group2": group_labels.get(g2, str(g2)) if group_labels else str(g2),
                "n1": n1,
                "n2": n2,
                "U_stat": U,
                "cliffs_delta": delta,
                "rank_biserial": r_rb,
                f"ci_{int(100 * (1 - alpha))}%_lower": ci_low,
                f"ci_{int(100 * (1 - alpha))}%_upper": ci_high,
                f'ci': [float(ci_low), float(ci_high)],
                "p_value": p,

            })

        # 2) Holm–Bonferroni p-value correction _within_ this variable
        if stats_for_var:
            p_vals = [d["p_value"] for d in stats_for_var]
            _, p_adj, _, _ = multipletests(p_vals, method="holm")
            for d, pad in zip(stats_for_var, p_adj):
                d["p_adj"] = pad

        results.extend(stats_for_var)

    return pd.DataFrame(results)

def count_data_types_per_section(
        encoding: Dict[str, dict],
        df: pd.DataFrame,
        sections: list[str]) -> Dict[str, Dict[str, int]]:
    """
    Count the number of binary, ordinal, and continuous variables in each section.
    Print a summary of the counts, including the total number of variables and percetanges per section.
    :param encoding:
    :param df:
    :param sections:
    :return:
    """
    # initialize counts
    section_counts = {
        sect: {'binary': 0, 'ordinal': 0, 'continuous': 0}
        for sect in sections
    }

    # count per‐variable types
    for var, meta in encoding.items():
        if var not in df.columns:
            continue

        sect = next((s for s in sections if var.startswith(s)), None)
        if not sect:
            continue

        n_levels = len(meta.get('encoding', {}))
        if n_levels == 2:
            section_counts[sect]['binary'] += 1
        elif n_levels > 2:
            section_counts[sect]['ordinal'] += 1
        else:  # 0 or 1 levels → continuous
            section_counts[sect]['continuous'] += 1

    # compute grand total of all counted features
    grand_total = sum(
        sum(counts.values())
        for counts in section_counts.values()
    )

    # print summary
    for sect, counts in section_counts.items():
        total = sum(counts.values())
        if total == 0:
            print(f"{sect:12s}: no variables found")
            continue

        b = counts['binary']
        o = counts['ordinal']
        c = counts['continuous']

        # within‐section percentages
        pb = b / total * 100
        po = o / total * 100
        pc = c / total * 100

        # percent of all features
        p_all = total / grand_total * 100

        print(
            f"{sect:3s}: {total} vars ({p_all:5.1f}% of all) |\n "
            f"binary={b} ({pb:5.1f}%) |\n "
            f"ordinal={o} ({po:5.1f}%) |\n"
            f"continuous={c} ({pc:5.1f}%)\n\n"
        )

    return section_counts


# plot of the sections count
df_demo_updated = pd.DataFrame({
    "Category": [
        "Demographics", "ESS", "Medical history",
        "Sleep assessment", "Pre sleep", "Post sleep"
    ],
    "Count": [4, 8, 57, 71, 12, 12],
    "Percentage": [2.41, 5.5, 34.5, 43.0, 43.0, 43.0]
})

# Create the updated plot with annotations
plt.figure(figsize=(10, 6))
barplot = sns.barplot(data=df_demo_updated, y="Category", x="Count", palette="crest")

# Add text annotations
for index, row in df_demo_updated.iterrows():
    barplot.text(
        row["Count"] + 1,  # x position
        index,  # y position
        f'{row["Count"]} ({row["Percentage"]:.1f}%)',
        color='black',
        va='center'
    )

plt.title("Distribution of Features by Section", fontsize=14)
plt.xlabel("Percentage of Total Features")
plt.ylabel("")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()



if __name__ == "__main__":
    # Load your data
    sections = ["mh_", "sa_"] # , "presleep_", "postsleep_"]
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    df = df[~df['osa_four'].isna()]
    # %% define output path
    out_dir = config.get("results")['dir'].joinpath('table_ones')
    out_dir.mkdir(parents=True, exist_ok=True)

    path_table = out_dir / 'table_one_all.xlsx'
    path_effect_measure_tab_one = out_dir.joinpath('effect_measure_tab_one.csv')

    path_effect_measure_tab_sa = out_dir.joinpath('effect_measure_sa.csv')
    # %% Data description report
    from config.config import sections as all_sections
    count_data_types_per_section(encoding, df, all_sections)

    # %% list of all columns
    variables = []
    for key, meta in encoding.items():
        # find which section this variable belongs to
        n_levels = len(meta.get('encoding', {}))
        if n_levels == 2:
            vartype = 'binary'
        elif n_levels > 2:
            vartype = 'ordinal'
        else:  # n_levels == 0
            vartype = 'continuous'
        variables.append((key, vartype))

    total_questions = len(encoding)
    for section in sections:
        length = len([col for col in encoding.keys() if col.startswith(section)])
        print(f'{section}: {length} ({((length/total_questions) * 100):.2f}%) variables')


    #%% different respiratory events
    events = metrics_psg.get('resp_events')['raw_events']
    pos = 'total'
    t_col = 'resp-position-total'  #  metrics_psg.get('resp_events')['position_keywords']

    cols_events = []
    for event in events:
        resp_event = f'resp-{event}-{pos}'
        if resp_event in df.columns:
            cols_events.append((f'resp-{event}-{pos}', 'continuous'))

    events_mapper_formal = {
        'resp-oa-total': 'Obstructive Apnea',
        'resp-ca-total': 'Central Apnea',
        'resp-ma-total': 'Mixed Apnea',
        'resp-hyp_4_30_airflow_reduction_desaturation-total': 'Hypopnea with 4% desat. 30% airflow reduction',
        'resp-rera_5_hyp_with_arousal_without_desaturation-total': 'RERA without 4% desat.',
        'resp-apnea-total': 'Apnea',
        'resp-apnea+hyp-total': 'Apnea + Hypopnea',
        'resp-apnea+hyp+rera-total': 'Apnea + Hypopnea + RERA',
        'resp-ri_rera_only-total': 'RERA only',
        'resp-hi_hypopneas_only-total': 'HYP only',
    }


    # %% sections to include in the table
    columns_tab_one = [
        ('dem_age', 'continuous'),
        ('dem_bmi', 'continuous'),
        ('dem_gender', 'binary'),
        ('dem_race', 'ordinal'),

        ('ep_reading', 'ordinal'),
        ('ep_watchingtv', 'ordinal'),
        ('ep_sittingquietlyinpublicplace', 'ordinal'),
        ('ep_carpassenger', 'ordinal'),
        ('ep_lyingdown', 'ordinal'),
        ('ep_talking', 'ordinal'),
        ('ep_sittingquietlyafterlunch', 'ordinal'),
        ('ep_incar', 'ordinal'),
        ('ep_score', 'continuous'),

        ('mh_high_blood_pressure', 'binary'),
        ('mh_liver_disease', 'binary'),
        ('mh_diabetes', 'binary'),
        ('mh_heart_attack', 'binary'),
        ('mh_tobacco_habit', 'binary'),
        ('mh_alcohol', 'binary'),
        ('mh_dentures', 'binary'),
        ('mh_chest_pain', 'binary'),

        ('sa_problem_wakingup_during_night', 'ordinal'),
        ('sa_problem_not_feeling_rested', 'ordinal'),
        ('sa_falling_asleep_depressed', 'ordinal'),
        ('sa_sleep_breath_problem', 'ordinal'),
        ('sa_sleep_choke', 'ordinal'),
        ('sa_sleep_gasp', 'ordinal'),
        ('sa_sleep_stop_breath', 'ordinal'),
        ('sa_sleep_thirst', 'ordinal'),

        ('presleep_feel_sleepy_today', 'binary'),
        ('presleep_feel_tired_now', 'ordinal'),

        ('postsleep_have_difficulty_falling_asleep', 'binary'),
        ('postsleep_slept_longer', 'binary'),
        ('postsleep_wake_during_night', 'binary'),
        ('postsleep_feeling_refreshing', 'ordinal'),
    ]
    columns_tab_one.extend(cols_events)
    # Re map the variables to the formal names
    index_mapper = {}
    variable_name_mapper = {}
    for col, dtype in columns_tab_one:
        if dtype == 'continuous':
            continue
        # mapping = {cal: key for key, cal in encoding[col]['encoding'].items()}
        # df[col] = df[col].map(mapping)
        mapping = {key: val for key, val in encoding[col]['encoding'].items()}
        index_mapper[col] = mapping
        variable_name_mapper.update({col: encoding[col]['definition'].replace('_', ' ').capitalize()})

    vars_categorical = [var for var, type_ in columns_tab_one if type_ in ['categorical','ordinal', 'binary']]
    vars_continuous = [var for var, type_ in columns_tab_one if type_ == 'continuous']
    columns = vars_categorical + vars_continuous

    df[columns].head(10)
    # %%
    if not path_table.exists():
        tab_one_all = MakeTableOne(df=df[~df['osa_four_numeric'].isna()],
                                      continuous_var=vars_continuous,
                                      categorical_var=vars_categorical,
                                      strata='osa_four',
                                   index_mapper=index_mapper,
                                   )
        df_tab_one_all = tab_one_all.create_table()
        df_tab_one_all['variable'].replace(variable_name_mapper, inplace=True)

        # rename all with formal names
        variable_name_mapper = {}
        for col in df_tab_one_all['variable'].unique():
            if col in encoding:
                variable_name_mapper.update({col: encoding[col]['definition'].replace('_', ' ').capitalize()})
        variable_name_mapper.update(events_mapper_formal)
        df_tab_one_all['variable'] = df_tab_one_all['variable'].replace(variable_name_mapper)

        df_tab_one_all.to_excel(out_dir / 'table_one_all.xlsx', index=False)

    # %% statistical test of effect sizes between groups with Confidence intervals

    pairs = df[['osa_four_numeric','osa_four']].drop_duplicates()
    s = pairs.set_index('osa_four_numeric')['osa_four'].to_dict()

    # %% Effect measures for table one continuous and ordinal variables
    if not path_effect_measure_tab_one.is_file():
        vars_continuous_stat_test = [var for var, type_ in columns_tab_one if type_ in ['continuous','ordinal']]
        # remove sa as we do it separately
        vars_continuous_stat_test = [var for var in vars_continuous_stat_test if not var.startswith('sa_')]
        df_effect_tab_one = compute_adjacent_nonparam_pairwise(df=df,
                                               cont_vars=vars_continuous_stat_test,
                                               group_col='osa_four_numeric',
                                                group_labels=s,
                                               n_boot=100)
        variable_name_mapper = {}
        for col in df_effect_tab_one['variable'].unique():
            if col in encoding:
                variable_name_mapper.update({col: encoding[col]['definition'].replace('_', ' ').capitalize()})
        variable_name_mapper.update(events_mapper_formal)
        df_effect_tab_one['variable'] = df_effect_tab_one['variable'].replace(variable_name_mapper)
        df_effect_tab_one.to_csv(path_effect_measure_tab_one, index=False)
    else:
        df_effect_tab_one = pd.read_csv(path_effect_measure_tab_one)
    # plot of the statistical measure
    plotter = ForestAdjacencyPlot(df_effect_tab_one,
                                  variable_col="variable",
                                  effect_col="cliffs_delta",
                                  lcl_col="ci_95%_lower",
                                  ucl_col="ci_95%_upper",
                                  comp1_col="group1",
                                  comp2_col="group2",
                                  output_path=out_dir / 'effect_measure_tab_one.png'
                                  )
    plotter.plot(
        figsize=(22, 8),
        marker_size=60,
        ci_linewidth=1.5,
        zero_line=True,
        xlabel="Cliff’s Δ (Rank-biserial)",
        horizontal=True
    )

    # %% Effect measure plot of sleep asessment that are non=binary
    col_sa = []  # collect the sa variables we will use for the plot
    col_sa_mapper = {}  # mapper to rename with formal names
    for key, meta in encoding.items():
        if key.startswith('sa_'):
            meta_sa_def = meta.get('definition').replace('_', ' ').capitalize()
            meta_sa_encoding = meta.get('encoding')
            if len(meta_sa_encoding) == 2:
                continue
            else:
                col_sa.append(key)
                col_sa_mapper[key] = meta_sa_def

    if not path_effect_measure_tab_sa.is_file():
        df_effect_sa = compute_adjacent_nonparam_pairwise(df=df,
                                               cont_vars=col_sa,
                                               group_col='osa_four_numeric',
                                                group_labels=s,
                                               n_boot=100)
        df_effect_sa['variable'] = df_effect_sa['variable'].replace(col_sa_mapper)
        df_effect_sa.to_csv(path_effect_measure_tab_sa, index=False)
    else:
        df_effect_sa = pd.read_csv(path_effect_measure_tab_sa)
        plotter_sa = ForestAdjacencyPlot(df_effect_sa,
                                      variable_col="variable",
                                      effect_col="cliffs_delta",
                                      lcl_col="ci_95%_lower",
                                      ucl_col="ci_95%_upper",
                                      comp1_col="group1",
                                      comp2_col="group2",
                                         output_path=out_dir / 'effect_measure_sa.png'
                                      )
        plotter_sa.plot(
            figsize=(18, 20),
            marker_size=60,
            ci_linewidth=1.5,
            zero_line=True,
            xlabel="Cliff’s Δ (Rank-biserial)",
            wrap_width=39
        )
