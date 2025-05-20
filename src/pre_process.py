import pathlib
from config.config import config, encoding
from library.helper import get_mappers, classify_osa
import pandas  as pd
from tabulate import tabulate
import numpy as np
import re
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def print_missing_summary(df, round_pct: int = 2) -> pd.DataFrame:
    """
    Compute and print a table of missing-value counts and percentages per column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to inspect.
    round_pct : int, default=2
        Number of decimal places to round the percentage.

    Returns
    -------
    pd.DataFrame
        A summary table with columns ['missing_count', 'missing_pct'].
    """
    # count & percentage
    missing_count = df.isna().sum()
    missing_pct   = df.isna().mean() * 100

    # build summary
    missing_summary = pd.DataFrame({
        'missing_count': missing_count,
        'missing_pct':   missing_pct.round(round_pct)
    })

    print(tabulate(
        missing_summary,
        headers='keys',  # use DataFrame column names
        tablefmt='psql',  # try 'grid', 'fancy_grid', 'github', etc.
        showindex=True  # include the DataFrame index (column names)
    ))
    return missing_summary

def drop_high_missing_columns(df: pd.DataFrame,
                              threshold: float = 0.6,
                              round_pct: int = 2,
                              tablefmt: str = 'psql'
                             ) -> (pd.DataFrame, list):
    """
    Drop columns from `df` with more than `threshold` fraction of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    threshold : float, default=0.6
        Fraction of missing values above which to drop a column.
    round_pct : int, default=2
        Number of decimals to round the missing‐percentage to.
    tablefmt : str, default='psql'
        Any valid `tabulate` tablefmt (e.g. 'github', 'grid', ...).

    Returns
    -------
    df_clean : pd.DataFrame
        The DataFrame with the high‐missing columns dropped.
    dropped_info : list of dict
        One dict per dropped column, with keys:
          - 'column': column name
          - 'missing_count': number of NaNs
          - 'total': total rows in the DataFrame
          - 'missing_pct': percentage of missing values
    """
    total = len(df)
    missing_count = df.isna().sum()
    missing_frac  = df.isna().mean()

    # Find columns to drop
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()

    # Build a list of dicts for reporting
    dropped_info = []
    for col in cols_to_drop:
        dropped_info.append({
            'column':        col,
            'missing_count': int(missing_count[col]),
            'total':         total,
            'missing_pct':   round(missing_frac[col] * 100, round_pct)
        })

    # Print report via tabulate
    if dropped_info:
        report_df = pd.DataFrame(dropped_info)
        print(tabulate(
            report_df,
            headers='keys',
            tablefmt=tablefmt,
            showindex=False
        ))
    else:
        print(f"No columns dropped. All columns have ≤ {threshold*100:.0f}% missing values.")

    # Drop the columns and return
    df_clean = df.drop(columns=cols_to_drop)
    return df_clean, dropped_info

def _resp_column_name_cleaner(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Remove any literal '_-_'
    2) Normalize event codes:
         oa_1   → oa
         ca_2   → ca
         ma_3   → ma
         hyp_4  → hyp
         rera_5 → rera
    """
    mapping = {
        'oa_1':  'oa',
        'ca_2':  'ca',
        'ma_3':  'ma',
        'hyp_4': 'hyp',
        'rera_5':'rera',
    }
    def clean_name(col: str) -> str:
        col = col.replace('_-_', '')
        for old, new_lbl in mapping.items():
            col = re.sub(rf'(?i)\b{old}\b', new_lbl, col)
        return col

    df = df.copy()
    df.columns = [clean_name(c) for c in df.columns]
    return df

def clean_numeric_strings(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    For each column in `cols`, remove common formatting from strings
    (commas, spaces), then coerce to float.

    Parameters
    ----------
    df : pd.DataFrame
    cols : list of str
        Column names in `df` to clean.

    Returns
    -------
    pd.DataFrame
        A copy of `df` where each col in `cols` has been:
          1) cast to str
          2) commas removed
          3) stripped of whitespace
          4) pd.to_numeric(..., errors='coerce')
    """
    df = df.copy()
    for c in cols:
        # 1) ensure string, remove commas & whitespace
        s = df[c].astype(str).str.replace(',', '', regex=False).str.strip()
        # 2) coerce to float (invalid strings → NaN)
        df[c] = pd.to_numeric(s, errors='coerce')
    return df

# 1) clean each raw column name
def _clean_resp_col(col: str) -> str:
    """
    Normalize a raw respiratory column name:
      - lowercase
      - remove any trailing '_<digits>' (e.g. '_1', '_2', …)
      - collapse parenthetical groups to '_<first_part>'
      - normalize separators (-,_,+)
      - replace any other chars with '_'
      - collapse runs of '_' or '-' to single char
      - strip leading/trailing separators
    """
    col = col.lower()
    # Remove any trailing '_<digits>'
    col = re.sub(r'_\d+\b', '', col)

    # Replace "(…)" with "_" + the first part before any "/"
    def _keep_first(m: re.Match) -> str:
        inner = m.group(1)
        first = inner.split('/')[0]
        return f"_{first}"
    col = re.sub(r'\(([^)]*)\)', _keep_first, col)

    # Normalize separators
    col = col.replace('_-_', '-').replace('_+_', '+')

    # Replace any char not in [a-z0-9-+_] with "_"
    col = re.sub(r'[^0-9a-z\-\+_]', '_', col)

    # Collapse multiple "_" or "-" into a single one
    col = re.sub(r'[_\-]{2,}', lambda m: m.group(0)[0], col)

    # Trim leading/trailing "_" or "-"
    return col.strip('_-')



def transform_and_plot_ahi_with_severity(df:pd.DataFrame,
                                         column:str='ahi',
                                         upper_threshold:int=160,
                                         output_path:pathlib.Path=None) -> pd.DataFrame:
    """
    Perform multiple AHI transformations, classify OSA severity, and plot distributions.

    Applies five different transformations to the specified AHI column and plots
    raw vs. log1p distributions shaded by OSA severity categories:
      1. Winsorization (95th percentile cap)
         - Pros: Limits extreme outliers; preserves data rank
         - Cons: Arbitrary cap threshold; may distort tail
      2. IQR capping (Q3 + 1.5*IQR)
         - Pros: Robust to outliers; data-driven
         - Cons: Can still allow moderate extremes; depends on IQR
      3. Box-Cox transformation (on AHI + 1)
         - Pros: Often yields near-normal distribution; parameter optimized
         - Cons: Requires strictly positive data; shift needed for zeros
      4. Arctan scaling
         - Pros: Smoothly compresses large values; bounded output
         - Cons: Nonlinear distortion; less interpretable scale
      5. Rank-based normalization
         - Pros: Uniform, outlier-robust distribution
         - Cons: Loses original metric meaning; only ranks matter

    Each method's raw and log1p-transformed histograms are shown side by side
    with shared severity legend.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the AHI column.
    column : str, default 'ahi'
        Name of the column to transform.
    upper_threshold : float, default 100
        Scale for the arctan transformation.

    Returns
    -------
    df_t : pandas.DataFrame
        Copy of `df` enriched with new transformed columns:
        - ahi_winsorized, ahi_logp1_winsor
        - ahi_iqr_scaled, ahi_logp1_iqr
        - ahi_boxcox, ahi_logp1_boxcox
        - ahi_arctan_scaled, ahi_logp1_arctan
        - ahi_rank, ahi_logp1_rank
    fig : matplotlib.figure.Figure
        Figure object with the grid of histograms.
    """
    # Work on a copy to preserve original DataFrame
    df_t = df.copy()
    raw_mean, raw_max, raw_min = df_t[column].mean(), df_t[column].max(), df_t[column].min()

    # 1. Winsorization: cap values above the 95th percentile
    df_t['ahi_winsorized'] = winsorize(df_t[column].copy(), limits=[0, 0.05], inplace=False)
    df_t['ahi_logp1_winsor'] = np.log1p(df_t['ahi_winsorized'])

    # 2. IQR capping: cap values above Q3 + 1.5*IQR
    Q1, Q3 = df_t[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    df_t['ahi_iqr_scaled'] = df_t[column].copy().clip(upper=upper_bound)
    df_t['ahi_logp1_iqr'] = np.log1p(df_t['ahi_iqr_scaled'].copy())

    # 3. Box-Cox transform: apply to AHI + 1 for positivity
    pt_box = PowerTransformer(method='box-cox')
    vals = df_t[column] + 1
    df_t['ahi_boxcox'] = pt_box.fit_transform(vals.fillna(vals.median()).values.reshape(-1, 1))[:, 0]
    df_t['ahi_logp1_boxcox'] = np.log1p(df_t['ahi_boxcox'].copy())

    # 4. Arctan scaling: compress tails smoothly
    df_t['ahi_arctan_scaled'] = np.arctan(df_t[column] / upper_threshold) * upper_threshold
    df_t['ahi_logp1_arctan'] = np.log1p(df_t['ahi_arctan_scaled'].copy())

    # 5. Rank normalization: convert to percentile ranks
    df_t['ahi_rank'] = df_t[column].rank(pct=True)
    df_t['ahi_logp1_rank'] = np.log1p(df_t['ahi_rank'].copy())

    # Severity thresholds and colors
    raw_thresh = np.array([5, 15, 30])
    sev_colors = ['#ffe5cc', '#ffcc99', '#ff9999', '#ff6666']
    sev_labels = ['Normal (<5)', 'Mild (5–15)', 'Moderate (15–30)', 'Severe (>=30)']

    # Precompute threshold positions for shading
    thr_map = {
        'winsor': {
            'raw': winsorize(raw_thresh.copy(), limits=[0, 0.05]),
            'log': np.log1p(winsorize(raw_thresh.copy(), limits=[0, 0.05]))
        },
        'iqr': {
            'raw': np.clip(raw_thresh.copy(), None, upper_bound),
            'log': np.log1p(np.clip(raw_thresh, None, upper_bound))
        },
        'boxcox': {
            'raw': pt_box.transform((raw_thresh + 1).reshape(-1, 1))[:, 0],
            'log': np.log1p(pt_box.transform((raw_thresh + 1).reshape(-1, 1))[:, 0])
        },
        'arctan': {
            'raw': np.arctan(raw_thresh / upper_threshold) * upper_threshold,
            'log': np.log1p(np.arctan(raw_thresh / upper_threshold) * upper_threshold)
        },
        'rank': {
            'raw': pd.Series(raw_thresh).rank(pct=True).values,
            'log': np.log1p(pd.Series(raw_thresh).rank(pct=True).values)
        }
    }

    # List of (raw_col, key, log_col) for plotting
    methods = [
        ('ahi_winsorized', 'winsor', 'ahi_logp1_winsor'),
        ('ahi_iqr_scaled', 'iqr', 'ahi_logp1_iqr'),
        ('ahi_boxcox', 'boxcox', 'ahi_logp1_boxcox'),
        ('ahi_arctan_scaled', 'arctan', 'ahi_logp1_arctan'),
        ('ahi_rank', 'rank', 'ahi_logp1_rank')
    ]

    # Arrange 2 methods per row => 3 rows, 4 columns (raw, log) pairs
    n_methods = len(methods)
    methods_per_row = 2
    n_rows = int(np.ceil(n_methods / methods_per_row))
    fig, axes = plt.subplots(n_rows,
                             methods_per_row * 2,
                             figsize=(26, 4 * n_rows),
                             squeeze=False)
    axes = axes.reshape(n_rows, methods_per_row * 2)

    # labels title
    title_map = {
        # Winsorization
        'ahi_winsorized': 'AHI Winsorized (95th percentile cap)',
        'ahi_logp1_winsor': 'Log₁₊ of Winsorized AHI',

        # IQR capping
        'ahi_iqr_scaled': 'AHI IQR-Capped (Q₃ + 1.5 · IQR)',
        'ahi_logp1_iqr': 'Log₁₊ of IQR-Capped AHI',

        # Box-Cox
        'ahi_boxcox': 'Box-Cox Transformed AHI (+1 shift)',
        'ahi_logp1_boxcox': 'Log₁₊ of Box-Cox AHI',

        # Arctan scaling
        'ahi_arctan_scaled': 'Arctan-Scaled AHI',
        'ahi_logp1_arctan': 'Log₁₊ of Arctan-Scaled AHI',

        # Rank
        'ahi_rank': 'AHI Percentile Rank',
        'ahi_logp1_rank': 'Log₁₊ of AHI Percentile Rank'
    }


    used_axes = []
    for idx, (raw_col, key, log_col) in enumerate(methods):
        row = idx // methods_per_row
        col_start = (idx % methods_per_row) * 2
        ax_raw = axes[row, col_start]
        ax_log = axes[row, col_start + 1]
        used_axes.extend([ax_raw, ax_log])

        # Shade + plot raw
        bounds_raw = [0, *thr_map[key]['raw'], df_t[raw_col].max()]
        for j in range(4):
            ax_raw.axvspan(bounds_raw[j], bounds_raw[j + 1], color=sev_colors[j], alpha=0.3)
        ax_raw.hist(df_t[raw_col].dropna(), bins=30, color='gray', alpha=0.8)
        ax_raw.set_title(title_map[raw_col])
        # Shade + plot log
        bounds_log = [0, *thr_map[key]['log'], df_t[log_col].max()]
        for j in range(4):
            ax_log.axvspan(bounds_log[j], bounds_log[j + 1], color=sev_colors[j], alpha=0.3)
        ax_log.hist(df_t[log_col].dropna(), bins=30, color='gray', alpha=0.8)
        ax_log.set_title(title_map[log_col])

    # Turn off any unused subplots
    for ax in axes.flatten():
        if ax not in used_axes:
            ax.axis('off')

    # Shared legend
    legend_patches = [mpatches.Patch(color=sev_colors[i], alpha=0.3, label=sev_labels[i]) for i in range(4)]
    fig.legend(handles=legend_patches,
               loc='upper center',
               ncol=4,
               title='OSA Severity',
               # fontsize=12,
               # title_fontsize=14
               )

    fig.tight_layout(rect=[0, 0, 0.90, 0.90])  # rect=[left, bottom, right, top]
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.show()
    # Looking at the shaded histograms, the Box-Cox transform stands out as giving the most symmetric, “bell-shaped” distribution (even after log1p), which will benefit any regression or parametric learner that assumes roughly Gaussian errors.
    #
    # Winsorized and IQR-capped AHI still show a long right tail once logged.
    #
    # Arctan nicely compresses the top end, but the bulk of values remains heavily skewed.
    #
    # Rank makes everything uniform, which is great for trees but loses the original metric spacing.
    #
    # By contrast, the Box-Cox (AHI + 1) raw histogram is much more centered, and its log1p version is quite symmetric—ideal for linear models, regularized regressions, and any algorithm sensitive to skew.
    #
    # If you’re using tree-based ensembles (RF, gradient boosting) you could get away with just winsorizing, but for most ML regressors I’d go with Box-Cox.
    assert all([raw_mean == df_t[column].mean(),  raw_max == df_t[column].max(), raw_min == df_t[column].min()])
    return df_t



if __name__ == '__main__':
    # %% Paths data
    path_in = config.get('data')['raw_data']['q_resp']
    path_out = config.get('data')['pp_data']['q_resp']
    results_plot_path = config.get('results')['dir']
    # %% Input data
    df = pd.read_csv(config.get('data')['raw_data']['q_resp'], low_memory=False)
    all_map, grouped_map = get_mappers()

    # df_good_ahi = pd.read_csv(r'C:\Users\giorg\OneDrive - Fundacion Raices Italo Colombianas\projects\EpiThesisBioser\data\raw_data\pp_data.csv', low_memory=False)
    #
    # df_corrected = pd.merge(left=df,
    #                         right=df_good_ahi[['ID', 'AHI']],
    #                         on='ID',
    #                         how='left')
    # assert df_corrected.shape[0] == df.shape[0]
    #
    # df_good_ahi = df_good_ahi.loc[df_good_ahi['ID'].isin(df['ID']), :]
    #
    # df.sort_values(by=['ID'], inplace=True)
    # df.reset_index(drop=True, inplace=True)
    # df_good_ahi.sort_values(by=['ID'], inplace=True)
    # df_good_ahi.reset_index(drop=True, inplace=True)


    # %% missing data
    print_missing_summary(df=df, round_pct=2)
    df, dropped = drop_high_missing_columns(df, threshold=0.6)
    # %% Cleaning from past pre-processing
    # remove previously transformed columns and replace with original
    raw_cols = [col for col in df.columns if col.endswith('_raw')]
    base_names = [col.split('_raw')[0] for col in raw_cols]
    to_drop = [base for base in base_names if base in df.columns]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
    rename_map = dict(zip(raw_cols, base_names))
    df.rename(columns=rename_map, inplace=True)
    # %% Rename  columns
    df.rename(columns={'DisplayGender': 'gender'}, inplace=True)
    # %% column names
    df.columns = df.columns.str.lower()
    # %% Assume no response in binary columns is a no response
    binary_cols = [col for col in grouped_map['binary'].keys() if col in df.columns]
    # 2) Fill NaNs with 0 and cast to integer
    if binary_cols:
        df[binary_cols] = df[binary_cols] \
                            .fillna(0) \
                            .astype(int)
    # %% target columns
    # standardize names
    raw_resp_cols = [c for c in df.columns if c.startswith('resp-') and 'n/a' not in c]
    # clean strings to all floats
    df = clean_numeric_strings(df, raw_resp_cols)
    # map the names of the resp columns
    clean_map = {c: _clean_resp_col(c) for c in raw_resp_cols}
    df.rename(columns=clean_map, inplace=True)
    # Apply log10(x+1) transformation with NaN handling
    df['ai_logp1'] = df['ai'].apply(lambda x: np.log10(x + 1) if pd.notnull(x) else np.nan)
    df['ahi_logp1'] = df['ahi'].apply(lambda x: np.log10(x + 1) if pd.notnull(x) else np.nan)

    # %% OSA severity levels
    # drop rows where the AHI is not defined
    df = df.loc[df['ahi'].notna(), :]
    df['osa_four'] = df['ahi'].apply(lambda x: classify_osa(x,
                                                            scheme='four',
                                                            return_code=False))
    df['osa_four_numeric'] = df['ahi'].apply(lambda x: classify_osa(x,
                                                                 scheme='four',
                                                                 return_code=True))

    df['osa_binary'] = df['ahi'].apply(lambda x: classify_osa(x,
                                                              scheme='binary',
                                                              return_code=False))
    df['osa_binary_numeric'] = df['ahi'].apply(lambda x: classify_osa(x,
                                                                   scheme='binary',
                                                                   return_code=True))

    df['binary_fifteenth'] = df['ahi'].apply(lambda x: classify_osa(x,
                                                                   scheme='binary_fifteenth',
                                                                   return_code=False))
    df['binary_fifteenth_numeric'] = df['ahi'].apply(lambda x: classify_osa(x,
                                                                   scheme='binary_fifteenth',
                                                                   return_code=True))
    # evaluate different transformations
    # df = transform_and_plot_ahi_with_severity(df,
    #                                           column='ahi',
    #                                           upper_threshold=df['ahi'].max(),
    #                                           output_path=results_plot_path.joinpath('ahi_transformations.png'))
    # cols_ahi = [col for col in df.columns if 'ahi' in col]
    # for regression use:  'ahi_logp1_boxcox',
    # for tree based models use:  'ahi_logp1_winsor'

    # %% recode race
    # 1) get the “active” race column name per row
    df['race'] = df[['race_0', 'race_1', 'race_2', 'race_3', 'race_4', 'race_5', 'race_6']] \
        .idxmax(axis=1)

    # 2) pull out the integer (e.g. 'race_2' → 2)
    df['race'] = df['race'].str.split('_').str[1].astype(int)
    race_map = {val: key for key, val in encoding.get('dem_race')['encoding'].items()}
    # df['race'] = df['race_code'].map(race_map)
    df = df.drop(columns=[f'race_{i}' for i in range(7)] )  # + ['race_code']

    # %% remove bipap
    df = df.loc[df['bipap'] == 'No', :]
    df = df.loc[df['oxygen'] == 'No', :]
    df.drop(columns=['bipap',
                     'oxygen',
                     # 'studytype',
                     'lab_cat',
                     'dob_year',
                     # 'md_identifyer',
                     ], inplace=True)
    # %% rename column sections
    # rename page 2 and page 3 to slee assessment section
    # Define specific ph2 columns for sleep assessment
    ph2_to_sa = [
        'ph2_shift_1st', 'ph2_shift_2nd', 'ph2_shift_3rd', 'ph2_shift_swing',
        'ph2_too_little_sleep', 'ph2_work_sleep_fatique', 'ph2_too_much_sleep',
        'ph2_accident_sleep_mistakes', 'ph2_accident_sleep_fatigue'
    ]
    # Identify all ph3 columns related to sleep assessment
    ph3_to_sa = [col for col in df.columns if col.startswith("ph3_")]
    # Combine ph2 and ph3 columns
    ph_to_sa = ph2_to_sa + ph3_to_sa
    # Generate new column names with 'sa' prefix
    sa_columns = [col.replace('ph2_', 'sa_').replace('ph3_', 'sa_') for col in ph_to_sa]
    # Create renaming dictionary
    rename_dict_sa = dict(zip(ph_to_sa, sa_columns))
    # Rename columns in the DataFrame
    df.rename(columns=rename_dict_sa, inplace=True)

    ph2_to_mh = {col: col.replace('ph2', 'mh') for col in df.columns if col.startswith("ph2")}
    # rename the
    df.rename(columns=ph2_to_mh, inplace=True)

    # epworth columns
    col_ep = [ep for ep in df.columns if ep.startswith('ep')]
    df['ep_score'] = df[col_ep].sum(axis=1)
    ep_cols = [c for c in df.columns
               if c.startswith("ep") and not c.startswith("ep_")]
    # 2) map each to 'ep_' + the rest of its name (stripping any extra '_')
    ep_map = {
        c: f"ep_{c[len('ep'):].lstrip('_')}"
        for c in ep_cols
    }
    df = df.rename(columns=ep_map)

    dem_mapper = {
        'age': 'dem_age',
        'bmi': 'dem_bmi',
        'race': 'dem_race',
        'gender': 'dem_gender',
    }
    df = df.rename(columns=dem_mapper)

    # %% drop columns
    cols_drop = ['mh_no_of_accidents_div_week']
    df.drop(columns=cols_drop, inplace=True)

    # re-orde the columns
    priority_cols = [
        'id', 'study_year', 'md_identifyer', 'studytype', 'dem_age', 'dem_gender', 'dem_race', 'dem_bmi', 'ess', 'rdi',
        'lowsat', 'tib', 'tst', 'sme', 'so', 'rol', 'ai', 'plms', 'di', 'sen', 'sao2_per', 'lps', 'ahi', 'isl', 'usl',
        'wasorace'
    ]
    priority_cols = [col for col in priority_cols if col in df.columns]
    # Get all other columns not in the priority list
    remaining_cols = sorted([col for col in df.columns if col not in priority_cols])

    # Reorder the DataFrame
    df = df[priority_cols + remaining_cols]

    # %% check if they are all in the encoding dict defintion
    for old, new in rename_dict_sa.items():
        if not new in encoding.keys():
            print(new)
    # %% wich columns in encoding are not in the df
    for col in encoding.keys():
        if col not in df.columns:
            print(col)

    df.rename(mapper={'mh_no_of_misses_binary': 'sa_no_of_misses_binary',
                      'mh_sleepiness_affect_performance': 'sa_sleepiness_affect_performance',
                      'mh_work_sleep_accidents_div_week': 'sa_work_sleep_accidents_div_week'},
              axis=1,
              inplace=True)

    # awake_feeling_questionnaire
    # sleep_quality_questionnaire
    # epworth_questionnaire
    # refreshed_questionnaire
    # anxiety_questionnaire

    # %% Reset index is important for the d mstrix index seleciton and not loc
    df.reset_index(drop=True, inplace=True)
    # %% Renamed
    # {'ph2_shift_1st': 'sa_shift_1st',
    #  'ph2_shift_2nd': 'sa_shift_2nd',
    #  'ph2_shift_3rd': 'sa_shift_3rd',
    #  'ph2_shift_swing': 'sa_shift_swing',
    #  'ph2_too_little_sleep': 'sa_too_little_sleep',
    #  'ph2_work_sleep_fatique': 'sa_work_sleep_fatique',
    #  'ph2_too_much_sleep': 'sa_too_much_sleep',
    #  'ph2_accident_sleep_mistakes': 'sa_accident_sleep_mistakes',
    #  'ph2_accident_sleep_fatigue': 'sa_accident_sleep_fatigue',
    #  'ph3_problem_going_tosleep_atnight': 'sa_problem_going_tosleep_atnight',
    #  'ph3_third_awaken_part': 'sa_third_awaken_part',
    #  'ph3_problem_wakingup_during_night': 'sa_problem_wakingup_during_night',
    #  'ph3_falling_asleep_thought': 'sa_falling_asleep_thought',
    #  'ph3_problem_not_feeling_rested': 'sa_problem_not_feeling_rested',
    #  'ph3_falling_asleep_depressed': 'sa_falling_asleep_depressed',
    #  'ph3_problem_with_tiredness': 'sa_problem_with_tiredness',
    #  'ph3_falling_asleep_worry': 'sa_falling_asleep_worry',
    #  'ph3_problem_with_sleepiness': 'sa_problem_with_sleepiness',
    #  'ph3_falling_asleep_muscular_tension': 'sa_falling_asleep_muscular_tension',
    #  'ph3_falling_asleep_afraid': 'sa_falling_asleep_afraid',
    #  'ph3_falling_asleep_paralyzed': 'sa_falling_asleep_paralyzed',
    #  'ph3_before_bed_watch_tv': 'sa_before_bed_watch_tv',
    #  'ph3_falling_asleep_jerk': 'sa_falling_asleep_jerk',
    #  'ph3_before_bed_sleep_aids': 'sa_before_bed_sleep_aids',
    #  'ph3_falling_asleep_legpain': 'sa_falling_asleep_legpain',
    #  'ph3_first_awaken_part': 'sa_first_awaken_part',
    #  'ph3_falling_asleep_dream': 'sa_falling_asleep_dream',
    #  'ph3_second_awaken_part': 'sa_second_awaken_part',
    #  'ph3_falling_asleep_discomfort': 'sa_falling_asleep_discomfort',
    #  'ph3_do_you_nap': 'sa_do_you_nap',
    #  'ph3_time_in_bed_weekday_hours': 'sa_time_in_bed_weekday_hours',
    #  'ph3_time_in_bed_weekend_hours': 'sa_time_in_bed_weekend_hours'}

    # %% save output
    df.to_csv(path_out, index=False)
    print(f'Pre-processed data saved to {path_out}')


