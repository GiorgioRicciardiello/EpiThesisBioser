"""
Perform linear regression to assess how a one unit increase in the features corresponds to a change in the outcome.
We will use the raw outcome AHI to have a better clinical interpretation.

https://stats.oarc.ucla.edu/other/mult-pkg/faq/ologit/
"""
import pathlib
from typing import  Optional, List, Union, Dict
import statsmodels.formula.api as smf
import numpy as np
import pickle
from pandas import CategoricalDtype
from config.config import config, metrics_psg, encoding, sections
from library.helper import get_mappers, classify_osa
import pandas  as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ForestPlot:
    def __init__(
        self,
        df,
        feature_col: str = 'feature',
        estimate_col: str = 'coef',
        ci_lower_col: str = 'CI_lower',
        ci_upper_col: str = 'CI_upper',
        pval_col: str = 'p_value',
        alpha: float = 0.05,
        null_value: float = 1.0
    ):
        """
        df: DataFrame with at least [feature_col, estimate_col, ci_lower_col, ci_upper_col, pval_col]
        null_value: the “no‐effect” line (1.0 for ORs, 0.0 for raw betas)
        """
        self.df = df.copy().reset_index(drop=True)
        self.feature_col  = feature_col
        self.est_col      = estimate_col
        self.cil_col      = ci_lower_col
        self.ciu_col      = ci_upper_col
        self.pval_col     = pval_col
        self.alpha        = alpha
        self.null_value   = null_value

        self._calculate_significance()

    def _calculate_significance(self):
        """Mark True if p < alpha AND CI does not cover the null_value."""
        df = self.df
        sig_mask = (
            (df[self.pval_col] < self.alpha) &
            (
                (df[self.cil_col] > self.null_value) |
                (df[self.ciu_col] < self.null_value)
            )
        )
        df['significant'] = sig_mask
        self.df = df

    def plot(self,
             ax=None,
             figsize=None,
             point_kwargs=None,
             err_kwargs=None,
             null_kwargs=None,
             or_fontsize: int = 10, or_fontweight: str = 'bold', or_fmt: str = "{:.2f}"):
        """
        Draw the forest plot with OR labels above each dot.

        New args:
        - or_fontsize: font size for the OR text
        - or_fontweight: font weight for the OR text ('bold', 'normal', etc.)
        - or_fmt: format string to display the OR (e.g. "{:.2f}" or "{:.1f}")
        """
        df = self.df
        n = len(df)
        if ax is None:
            if figsize is None:
                figsize = (6, max(4, 0.4 * n))
            fig, ax = plt.subplots(figsize=figsize)

        ys = list(range(n, 0, -1))

        # error bars
        err_kwargs = err_kwargs or dict(fmt='none', ecolor='gray', capsize=3)
        ax.errorbar(
            x=df[self.est_col], y=ys,
            xerr=[df[self.est_col] - df[self.cil_col],
                  df[self.ciu_col] - df[self.est_col]],
            **err_kwargs
        )

        # points
        point_kwargs = point_kwargs or dict(marker='o',
                                            facecolors='white',
                                            edgecolors='black',
                                            zorder=3)
        ax.scatter(df[self.est_col], ys, **point_kwargs)
        sig = df['significant']
        ax.scatter(df.loc[sig, self.est_col],
                   [ys[i] for i in df[sig].index],
                   marker='o', facecolors='red', edgecolors='black', zorder=4)

        # null line
        null_kwargs = null_kwargs or dict(color='black', linestyle='--', linewidth=1)
        ax.axvline(self.null_value, **null_kwargs)

        # --- HERE: add the OR text labels ---
        for x, y in zip(df[self.est_col], ys):
            txt = or_fmt.format(x)
            # place it 5 points above the marker
            ax.annotate(
                txt,
                xy=(x, y),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=or_fontsize,
                fontweight=or_fontweight,
                color=point_kwargs.get('edgecolors', 'black')
            )

        # labels & styling
        ax.set_yticks(ys)
        ax.set_yticklabels(df[self.feature_col])
        ax.set_xlabel(f'Estimate (95% CI), null={self.null_value}')
        ax.grid(axis='x', linestyle=':', alpha=0.5)
        ax.set_title("Forest plot of odds ratios")
        return ax


def compute_iterative_regression(
    data: pd.DataFrame,
    target: str,
    cols_iterate: List[str],
    cols_base: List[str]
) -> pd.DataFrame:
    """
    For each candidate in cols_iterate, fit a logistic regression of target ~ cols_base + candidate,
    extracting sample size, OR, 95% CI, and p-value for the candidate.
    Returns a DataFrame with rows [feature, sample_size, OR, CI_lower, CI_upper, p_value].
    """
    results = []

    # Ensure that any categorical columns are dtype 'category'
    # e.g. dem_gender, dem_race; you can adjust this list as needed
    for cat_col in ['dem_gender', 'dem_race']:
        if cat_col in data.columns:
            data[cat_col] = data[cat_col].astype('category')

    for cand in cols_iterate:
        # Build the working DataFrame
        cols = cols_base + [cand, target]
        df_model = data[cols].dropna().copy()
        n = df_model.shape[0]

        # Build formula terms
        terms: List[str] = []
        # base covariates
        for col in cols_base:
            if isinstance(df_model[col].dtypes, CategoricalDtype):
                terms.append(f"C({col})")
            else:
                terms.append(col)
        # candidate covariate
        if isinstance(df_model[cand].dtypes, CategoricalDtype):
            terms.append(f"C({cand})")
        else:
            terms.append(cand)

        formula = f"{target} ~ " + " + ".join(terms)

        # fit OLS
        model = smf.ols(formula=formula, data=df_model).fit(cov_type='HC1')

        # Extract OR, CI, and p-value for candidate term(s)
        conf = model.conf_int()
        for param_name, coef in model.params.items():
            if cand in param_name:
                se = model.bse[param_name]
                ci_low, ci_upp = conf.loc[param_name]
                results.append({
                    'feature': param_name,
                    'sample_size': n,
                    'coef': coef,
                    'std_err': se,
                    'OR': np.exp(coef).round(2),
                    'CI_lower': np.exp(ci_low).round(3),
                    'CI_upper': np.exp(ci_upp).round(3),
                    'p_value': model.pvalues[param_name]
                })

    # Assemble final DataFrame
    df_res = (
        pd.DataFrame(results)
          .reset_index(drop=True)
    )
    return df_res

if __name__ == '__main__':
    # %% Input data
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    path_output = config.get('results')['dir'].joinpath('regression_significance_test')
    out_file_csv = path_output / 'iterative_logistic_regression_results.csv'

    # %% Create output directory if it doesn't exist'
    path_output.mkdir(parents=True, exist_ok=True)
    # %%
    prefix_formal = {
        "mh": 'Medical History',
        "sa": 'Sleep Assessment',
        'presleep': 'Presleep',
        'postsleep': 'Postsleep',
    }

    if not out_file_csv.exists():
        cols_base = ["dem_age", "dem_bmi", "dem_gender", "dem_race"]
        target = 'ahi'
        # Collect all features to iterate
        all_results = []
        sections_iterate = [sec for sec in sections if not sec in ['dem_', 'resp']]
        for sec in tqdm(sections_iterate):
            cols_iterate = [col for col in df.columns if col.startswith(sec) and not 'pca_' in col]
            if not cols_iterate:
                continue
            df_summary = compute_iterative_regression(
                data=df,
                target=target,
                cols_iterate=cols_iterate,
                cols_base=cols_base
            )
            all_results.append(df_summary)

        # Concatenate results from all sections
        final_table = pd.concat(all_results, ignore_index=True)

        final_table.sort_values(by='OR', ascending=False, inplace=True)

        # compute log‐OR and its CIs
        final_table['logOR'] = np.log(final_table['OR'])
        final_table['logCI_low'] = np.log(final_table['CI_lower'])
        final_table['logCI_high'] = np.log(final_table['CI_upper'])

        # %% rename the feature columns to formal names
        feature_mapper = {key:val.get('definition', key).replace('_', ' ').title()+f' ({key.split("_")[0]})' for key, val in encoding.items()}
        final_table['var_name'] = final_table['feature']

        final_table['feature'] = final_table['feature'].replace(feature_mapper)
        final_table['section'] = final_table['feature'].str.extract(r'\((.*?)\)', expand=False)
        final_table['section'] = final_table['section'].replace(prefix_formal)
        final_table['section'] = final_table['section'].replace({'Weekend': 'Medical History',
                                        'Weekday': 'Medical History'})
        final_table['feature'] = final_table['feature'].apply(lambda x: x.split('(')[0])


        def check_significance(row, alpha: float = 0.05, null_value: float = 1.0) -> bool:
            return (
                    (row['p_value'] < alpha)
                    and ((row['CI_lower'] > null_value) or (row['CI_upper'] < null_value))
            )

        final_table['significant'] = final_table.apply(check_significance, axis=1)

        # for the plot we need to write the unit of measure of each feature
        final_table['unit_measure'] = final_table['var_name'].apply(lambda x: str(encoding.get(x)['encoding']))

        feature_unit_measure = {"{'high chance of dozing': 3, 'moderate chance': 2, 'slight chance': 1, 'would never doze': 0}": 'E',
         "{'yes': 1, 'no': 0}": 'B',
         '{}': 'C',
         "{'white (caucasian)': 0, 'african american': 1, 'hispanic': 2, 'american indian': 3, 'pacific islander': 4, 'asian': 5, 'alaska native': 6, 'native hawaiian': 7}": 'R',
         "{'quite a bit': 2, 'not at all / none': 0, 'extreme': 3, 'a little': 1}": 'A',
         "{'better': 1, 'same': 0, 'worse': -1}": 'D',
         "{'a little': 1, 'extremely': 3, 'not at all / none': 0, 'quite a bit': 2}": 'E',
         "{'moderate / sometimes': 2, 'not at all / none': 0, 'often': 3, 'severe / always': 4, 'slight / few times': 1}": 'F',
         "{'male': 1, 'female': 0}": 'G'
         }

        final_table['unit_measure_code'] = final_table['unit_measure'].map(feature_unit_measure)

        final_table['feature_display'] = final_table['feature'] + final_table['unit_measure_code'].apply(lambda c: f"$^{{{c}}}$")

        final_table['CI'] = final_table.apply(lambda row: f"[{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]", axis=1)
        final_table['coef'] = final_table['coef'].round(3)
        final_table['std_err'] = final_table['std_err'].round(3)
        # %% Save to CSV
        final_table.to_csv(out_file_csv, index=False)
    else:
        final_table = pd.read_csv(out_file_csv)

    # %% Let's do a forest plot
    {sect: (8,6) for sect in final_table['section'].unique()}
    figsize = {'Postsleep': (8, 6),
               'Medical History': (12, 14),
               'Sleep Assessment': (12, 15),
               'Presleep': (8,6 ),
               'ep': (8, 7),
               }


    for sect in final_table['section'].unique():
        df_plot = final_table.loc[(final_table['significant'] == True) &
                                        (final_table['section'] == sect), :]
        fp = ForestPlot(df_plot,
                        feature_col='feature_display',
                        estimate_col='logOR',
                        ci_lower_col='logCI_low',
                        ci_upper_col='logCI_high',
                        pval_col='p_value',
                        alpha=0.05,
                        # null_value=1.0,
                        null_value=0.0  # on the log‐OR scale, null is 0
                        )
        ax = fp.plot(figsize=figsize.get(sect, (8, 6)))
        ax.set_title(f"{sect} \n Significant Only")
        ax.set_xlabel("log(OR) (95% CI)")
        plt.tight_layout()
        plt.savefig(path_output.joinpath(f'{sect}_forest_plot.png'), dpi=300)
        plt.show()

























