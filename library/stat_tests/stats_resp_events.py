#!/usr/bin/env python3
"""
respiratory_position_pipeline.py

A class that:
  1) computes position-normalized event-rates (with artifact filtering on short epochs),
  2) plots boxplots,
  3) runs within-subject stats (ANOVA or Friedman + pairwise tests),
  4) summarizes rates by position,
  5) labels “positional OSA” cases.
"""

from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    shapiro,
    friedmanchisquare,
    wilcoxon,
    ttest_rel
)
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from itertools import combinations


class RespiratoryPositionPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        metrics: dict,
        id_col: str           = 'id',
        pos_time_prefix: str  = 'resp-position',
        min_time: float       = 10.0,
        output_path: Path     = None,
    ):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Raw wide-format data containing:
              • 'resp-<raw_event>-<position>'  (raw counts)
              • 'resp-<index>-<position>'       (#/hr already)
              • 'resp-position-<position>'      (minutes in each position)
        metrics : dict
            metrics_psg['resp_events'] must contain:
              • 'raw_events':   list of event names to normalize by time
              • 'indices':      list of event names already expressed as #/hr
              • 'position_keywords': list of position suffixes
              • 'sleep_stage':       list of stage suffixes (e.g. 'rem','nrem')
        id_col : str
            Column name for the subject identifier.
        pos_time_prefix : str
            Prefix for the time‐in‐position columns (in minutes).
        min_time : float
            Epochs with time_in_position < min_time are dropped as artifacts.
        output_path : Path
            If given, saves boxplots there.
        """
        self.df              = df.copy()
        self.id_col          = id_col
        self.pos_time_prefix = pos_time_prefix
        self.min_time        = min_time
        self.output_path     = Path(output_path) if output_path else None

        resp = metrics['resp_events']
        self.raw_events  = resp['raw_events']
        self.indices     = resp['indices']
        self.positions   = resp['position_keywords'] + resp['sleep_stage']

        # unified list of all events to iterate downstream
        self.events = self.raw_events + self.indices

        self.df_long = None


    def compute_event_rates(self) -> pd.DataFrame:
        """
        Build self.df_long with columns:
        [id_col, event, position, count, time_min, rate].

        - raw_events are divided by (time_min/60) → events/hour
        - indices are taken as-is (already #/hr)
        - epochs with time_min < min_time are dropped
        """
        records: List[pd.DataFrame] = []

        # 1) Normalize raw‐count events
        for ev in self.raw_events:
            for pos in self.positions:
                cnt_col = f"resp-{ev}-{pos}"
                t_col   = f"{self.pos_time_prefix}-{pos}"
                if cnt_col in self.df and t_col in self.df:
                    cnt  = self.df[cnt_col]
                    tmin = self.df[t_col]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rate = cnt / (tmin / 60)
                    rate = rate.replace([np.inf, -np.inf], np.nan)

                    df_tmp = pd.DataFrame({
                        self.id_col: self.df[self.id_col],
                        'event':    ev,
                        'position': pos,
                        'count':    cnt,
                        'time_min': tmin,
                        'rate':     rate
                    })
                    # artifact filter
                    df_tmp = df_tmp[df_tmp['time_min'] >= self.min_time]
                    records.append(df_tmp)

        # 2) Directly use indices (#/hr)
        for ev in self.indices:
            for pos in self.positions:
                col = f"resp-{ev}-{pos}"
                if col in self.df:
                    rate = self.df[col]
                    df_tmp = pd.DataFrame({
                        self.id_col: self.df[self.id_col],
                        'event':    ev,
                        'position': pos,
                        'count':    np.nan,   # not raw count
                        'time_min': np.nan,
                        'rate':     rate
                    })
                    records.append(df_tmp)

        if not records:
            raise ValueError("No matching columns found for any event/position.")

        self.df_long = pd.concat(records, ignore_index=True)
        return self.df_long


    def _plot_position_rates(self, event: str, figsize: Tuple[int,int]=(8,4)):
        """
        Draw a boxplot of `rate` by `position` for one event.
        """
        if self.df_long is None:
            raise RuntimeError("Call compute_event_rates() first.")

        df_e = (
            self.df_long[self.df_long['event'] == event]
                .dropna(subset=['rate'])
        )
        if df_e.empty:
            print(f"[plot] no data for '{event}'")
            return

        plt.figure(figsize=figsize)
        sns.boxplot(x='position', y='rate', data=df_e)
        plt.title(f"{event} – events/hour by position")
        plt.ylabel("Events per hour")
        plt.xlabel("Position")
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        if self.output_path:
            plt.savefig(self.output_path / f"{event}_boxplot.png", dpi=300)
        plt.tight_layout()
        plt.show()


    def _run_stat_tests(self, event: str, alpha: float = 0.05) -> Dict:
        """
        Within‐subject tests on rates across positions.

        1) Shapiro–Wilk on (supine−prone) diffs
           H₀: differences ∼ normal.
        2a) If normal → Repeated‐measures ANOVA (H₀: means equal)
            + paired t‐tests (Holm).
        2b) Else → Friedman (H₀: distributions equal)
            + Kendall’s W + Wilcoxon (Holm).
        """
        if self.df_long is None:
            raise RuntimeError("Call compute_event_rates() first.")

        df_w = (
            self.df_long[self.df_long['event'] == event]
                .pivot(index=self.id_col, columns='position', values='rate')
                .dropna()
        )
        n_subj, n_pos = df_w.shape

        diffs = df_w['supine'] - df_w['prone']
        W, p_sw = shapiro(diffs)
        print(f"Shapiro–Wilk W={W:.3f}, p={p_sw:.3f}")

        results: Dict = {}

        if p_sw > alpha:
            print("→ Parametric: Repeated‐measures ANOVA")
            anova = AnovaRM(
                self.df_long[self.df_long['event']==event].dropna(subset=['rate']),
                depvar='rate',
                subject=self.id_col,
                within=['position']
            ).fit()
            print(anova)
            results['anova'] = anova

            # pairwise t‐tests
            pairs = list(combinations(df_w.columns, 2))
            pvals = [ttest_rel(df_w[a], df_w[b]).pvalue for a,b in pairs]
            rej, p_holm, *_ = multipletests(pvals, method='holm')
            results['pairwise'] = pd.DataFrame({
                'comparison':   [f"{a} vs {b}" for a,b in pairs],
                'p_uncorrected':pvals,
                'p_holm':       p_holm,
                'significant':  rej
            })

        else:
            print("→ Nonparametric: Friedman test")
            stat, p_f = friedmanchisquare(*[df_w[pos] for pos in df_w.columns])
            Wk = stat / (n_subj * (n_pos - 1))
            print(f"χ²={stat:.3f}, p={p_f:.3f}; Kendall’s W={Wk:.3f}")
            results['friedman']  = (stat, p_f)
            results['kendall_w'] = Wk

            # pairwise Wilcoxon
            pairs = list(combinations(df_w.columns, 2))
            pvals = [wilcoxon(df_w[a], df_w[b]).pvalue for a,b in pairs]
            rej, p_holm, *_ = multipletests(pvals, method='holm')
            results['pairwise'] = pd.DataFrame({
                'comparison':   [f"{a} vs {b}" for a,b in pairs],
                'p_uncorrected':pvals,
                'p_holm':       p_holm,
                'significant':  rej
            })

        return results


    def _summarize_rates(self, event: str) -> pd.DataFrame:
        """
        Return median (IQR) and mean±SD of rate by position.
        """
        if self.df_long is None:
            raise RuntimeError("Call compute_event_rates() first.")

        df_e = (
            self.df_long[self.df_long['event'] == event]
                .dropna(subset=['rate'])
        )
        grp = df_e.groupby('position')['rate']
        summary = grp.agg([
            ('median','median'),
            ('q1',   lambda x: np.percentile(x,25)),
            ('q3',   lambda x: np.percentile(x,75)),
            ('mean','mean'),
            ('sd',   'std'),
        ]).reset_index()
        summary['IQR'] = summary['q3'] - summary['q1']
        return summary[['position','median','IQR','mean','sd']]


    def label_positional_osa(
        self,
        event: str,
        lateral_positions: List[str] = ('left','right'),
        require_rem_supine: bool     = False
    ) -> pd.Series:
        """
        Flag subjects whose supine rate exceeds both prone & any lateral rate.
        Optionally also require REM_supine > REM.
        """
        if self.df_long is None:
            raise RuntimeError("Call compute_event_rates() first.")

        df_w = (
            self.df_long[self.df_long['event'] == event]
                .pivot(index=self.id_col, columns='position', values='rate')
        )
        lateral_max = df_w[lateral_positions].max(axis=1)
        mask = (df_w['supine'] > df_w['prone']) & (df_w['supine'] > lateral_max)
        if require_rem_supine:
            mask &= (df_w['rem_supine'] > df_w['rem'])
        return mask


    def run(self, event: str = 'ahi_no_reras'):
        """
        Convenience: compute rates, plot, stats, summary.
        """
        self.compute_event_rates()
        self._plot_position_rates(event)
        stats   = self._run_stat_tests(event)
        summary = self._summarize_rates(event)
        return stats, summary


# %%
def summarize_event_counts(df,
                           events,
                           group_cols=None):
    """
    Compute mean, IQR and std for total respiratory‐event counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Your input dataframe, must contain columns like "resp-oa-total", etc.
    events : list of str
        List of event codes, e.g. ['oa','ca','ma'].
    group_cols : list of str, optional
        Columns to group by (e.g. ['gender','age_group','bmi_group']).
        If None, computes global summaries.

    Returns
    -------
    pandas.DataFrame
        Summary table with columns = group_cols + ['event','mean','IQR','std'].
    """
    summary_list = []

    for e in events:
        col = f"resp-{e}-total"
        if group_cols:
            # group and compute
            stats = (
                df
                .groupby(group_cols)[col]
                .agg(
                    mean='mean',
                    std='std',
                    IQR=lambda x: x.quantile(0.75) - x.quantile(0.25)
                )
                .reset_index()
            )
            stats['event'] = e
        else:
            # overall
            series = df[col].dropna()
            stats = pd.DataFrame({
                'event': [e],
                'mean': [series.mean()],
                'std': [series.std()],
                'IQR': [series.quantile(0.75) - series.quantile(0.25)]
            })

        summary_list.append(stats)

    summary_df = pd.concat(summary_list, ignore_index=True)
    # ensure column order
    cols = (group_cols or []) + ['event', 'mean', 'IQR', 'std']
    return summary_df[cols]


def plot_event_trends_by_group(df, events,
                               age_group_col='age_group',
                               bmi_group_col='bmi_group',
                               count_fmt='resp-{}-total',
                               figsize_per_plot=(4, 4)):
    """
    Plots mean ±1 SD of event counts across age and BMI groups.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns for age_group, bmi_group and resp-<event>-total.
    events : list of str
        Event codes, e.g. ['oa','ca','ma'].
    age_group_col : str
        Name of the pre-made age‐bin column.
    bmi_group_col : str
        Name of the pre-made BMI‐bin column.
    count_fmt : str
        Format string for the count column; '{}' will be replaced by each event.
    figsize_per_plot : tuple
        (width, height) per subplot.
    """
    n = len(events)
    fig, axes = plt.subplots(2, n,
                             figsize=(figsize_per_plot[0] * n,
                                      figsize_per_plot[1] * 2),
                             sharey=False)

    for i, e in enumerate(events):
        col = count_fmt.format(e)

        # --- AGE ROW ---
        grp = df.groupby(age_group_col)[col]
        means = grp.mean()
        sds = grp.std()
        x = means.index.astype(str)

        ax = axes[0, i]
        ax.plot(x, means, marker='o')
        ax.fill_between(x, means - sds, means + sds, alpha=0.3)
        ax.set_title(e.upper())
        if i == 0:
            ax.set_ylabel('Mean count ± SD')
        ax.set_xlabel('Age group')
        ax.tick_params(axis='x', rotation=30)

        # --- BMI ROW ---
        grp = df.groupby(bmi_group_col)[col]
        means = grp.mean()
        sds = grp.std()
        x = means.index.astype(str)

        ax = axes[1, i]
        ax.plot(x, means, marker='o')
        ax.fill_between(x, means - sds, means + sds, alpha=0.3)
        if i == 0:
            ax.set_ylabel('Mean count ± SD')
        ax.set_xlabel('BMI group')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout()
    plt.show()
