import pickle
from config.config import config, encoding
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from scipy.stats.mstats import winsorize
from typing import Union

def get_mappers() -> (dict, dict):
    """
    Load the questionnaire encoding mappings, normalize all keys to lower‐case,
    and split into categories: 'ordinal', 'binary', and 'continuous'.

    # Usage:
    # all_map, grouped_map = get_mappers()
    # print("All keys:", list(all_map.keys()))
    # print("Ordinal keys:", list(grouped_map['ordinal'].keys()))

    Returns
    -------
    mapper_all : dict
        Nested dictionary of all encodings with:
          - outer keys   : lower‐cased questionnaire column names
          - 'definition' : lower‐cased definition string
          - 'encoding'   : dict mapping lower‐cased responses to numeric codes

    mapper_grouped : dict
        A dict with three keys:
          - 'ordinal'    : items where encoding has >2 levels
          - 'binary'     : items where encoding has exactly 2 levels
          - 'continuous' : items with no encoding dict or ≤1 level
        Each sub‐dict maps column_name → the same structure as mapper_all.
    """
    # 1) Load raw mapping
    with open(str(config.get('data')['encodings']), 'rb') as f:
        raw_data = pickle.load(f)

    # find which encoder_dict keys are not yet in raw_data
    missing = set(encoding) - set(raw_data.keys())

    if missing:
        # 2) inject only the truly missing mappings
        for name in missing:
            raw_data[name] = {
                'definition': name,
                'encoding': encoding[name]
            }

        # 3) persist the updated dict
        with open(str(config.get('data')['encodings']), 'wb') as f:
            pickle.dump(raw_data, f)

    # 2) Normalize to lower‐case
    mapper_all = {}
    for col, info in raw_data.items():
        col_lc = col.lower()
        if isinstance(info, dict):
            mapper_all[col_lc] = {
                'definition': info.get('definition', '').lower(),
                'encoding': {resp.lower(): code for resp, code in info.get('encoding', {}).items()}
            }
        else:
            mapper_all[col_lc] = info

    # 3) Group by encoding size
    mapper_grouped = {'ordinal': {}, 'binary': {}, 'continuous': {}}
    for col, info in mapper_all.items():
        enc = info.get('encoding')
        if isinstance(enc, dict):
            n_levels = len(enc)
            if n_levels > 2:
                mapper_grouped['ordinal'][col] = info
            elif n_levels == 2:
                mapper_grouped['binary'][col] = info
            else:
                mapper_grouped['continuous'][col] = info
        else:
            mapper_grouped['continuous'][col] = info

    return mapper_all, mapper_grouped


def classify_osa(ahi: float,
                 scheme: str = 'four',
                 return_code: bool = False) -> Union[str, int, float]:
    """
    Classify obstructive sleep apnea (OSA) severity from the AHI,
    returning either the label or the numeric code.

    Parameters
    ----------
    ahi : float
        Apnea–hypopnea index (events per hour).
    scheme : str, default='four'
        'four'  – Four-class severity: Normal, Mild, Moderate, Severe
        'binary' – Binary severity: No OSA vs. OSA
    return_code : bool, default=False
        If True, return the integer code; otherwise return the string label.

    Returns
    -------
    str or int
        The severity label or code according to the chosen scheme.

    Raises
    ------
    ValueError
        If `scheme` is not one of the keys in OSA_SEVERITY.
    """
    # define this at module level so it stays “static”
    OSA_SEVERITY = {
        'four': [
            ('Normal', float('-inf'), 5, 0),
            ('Mild', 5, 15, 1),
            ('Moderate', 15, 30, 2),
            ('Severe', 30, float('inf'), 3),
        ],
        'binary_five': [
            ('No OSA', float('-inf'), 5, 0),
            ('OSA', 5, float('inf'), 1),
        ],
        'binary_fifteenth': [
            ('No OSA', float('-inf'), 15, 0),
            ('OSA', 15, float('inf'), 1),
        ]
    }
    if scheme not in OSA_SEVERITY:
        raise ValueError(f"`scheme` must be one of {list(OSA_SEVERITY)}")

    # handle NaN
    if ahi is None or ahi != ahi:
        return np.nan

    for label, lo, hi, code in OSA_SEVERITY[scheme]:
        if lo <= ahi < hi:
            return code if return_code else label

    # should never happen
    raise RuntimeError(f"Failed to classify AHI={ahi!r} under scheme='{scheme}'")



def slice_psg_events(df: pd.DataFrame,
                     metrics: dict,
                     event: str,
                     position: str,
                     sleep_stage: str,
                     id_col: str = 'id') -> pd.DataFrame:
    """
    Return a DataFrame containing the id column plus all columns matching
    the given event, position, and sleep_stage based on metrics_psg dict.

    Usage
        out = slice_psg_events(df,
                           metrics_psg,
                           event='oa',
                           position='supine',
                           sleep_stage='rem',
                           id_col='id')


    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with PSG metrics columns.
    metrics : dict
        Dictionary defining 'event_keywords', 'position_keywords', and 'sleep_stage' lists.
    event : str
        One of metrics['resp_events']['event_keywords'].
    position : str
        One of metrics['resp_events']['position_keywords'].
    sleep_stage : str
        One of metrics['resp_events']['sleep_stage'].
    id_col : str, optional
        Name of the identifier column to include (default is 'id').

    Returns
    -------
    pd.DataFrame
        A slice of `df` with only the id column and the matching metric columns.
    """
    # extract allowed keywords
    evt_list = metrics['resp_events']['event_keywords']
    pos_list = metrics['resp_events']['position_keywords']
    ss_list = metrics['resp_events']['sleep_stage']

    # validate inputs
    for name, val, allowed in [('event', event, evt_list),
                               ('position', position, pos_list),
                               ('sleep_stage', sleep_stage, ss_list)]:
        if val not in allowed:
            raise ValueError(f"{name!r} must be one of {allowed}, got {val!r}.")

    # find matching columns
    matching = [
        col for col in df.columns
        if (event in col) and (position in col) and (sleep_stage in col)
    ]

    # ensure id_col exists
    if id_col not in df.columns:
        raise KeyError(f"Identifier column {id_col!r} not found in DataFrame.")

    # return slice
    return df[[id_col] + matching]


# %% AHI transformations

# y dependent varbiable
def inverse_transform_ahi(preds,
                          df,
                          column='ahi',
                          method='boxcox',
                          scale='log1p',
                          upper_threshold=100):
    """
    Inverse-transform regression predictions back to raw AHI scale.

    Parameters
    ----------
    preds : array-like
        Predicted values from regression on transformed scale.
    df : pandas.DataFrame
        Original DataFrame with raw AHI values for threshold and quantile mapping.
    column : str, default 'ahi'
        Name of the raw AHI column in `df`.
    method : {'winsor', 'iqr', 'boxcox', 'arctan', 'rank'}, default 'boxcox'
        Transformation method used during training.
    scale : {'raw', 'log1p'}, default 'log1p'
        Whether predictions are on the raw-transformed scale or log1p scale.
    upper_threshold : float, default 100
        Threshold used in arctan scaling (only for method='arctan').

    # Example usage:
    # raw_preds = inverse_transform_ahi(model_preds, df, method='boxcox', scale='log1p')



    Returns
    -------
    numpy.ndarray
        Predictions on the original AHI scale, clipped or inverted appropriately.
    """
    preds = np.asarray(preds)

    # 1) If stored on log1p scale, invert the log
    if scale == 'log1p':
        preds_trans = np.expm1(preds)
    elif scale == 'raw':
        preds_trans = preds.copy()
    else:
        raise ValueError("scale must be 'raw' or 'log1p'")

    # Inverse of each transformation
    if method == 'winsor':
        # Pros: simple inverse; Cons: truncated above 95th percentile
        thresh = np.percentile(df[column].dropna(), 95)
        raw_preds = np.clip(preds_trans, None, thresh)

    elif method == 'iqr':
        # Pros: robust to outliers; Cons: still caps extremes
        Q1, Q3 = df[column].quantile([0.25, 0.75])
        ub = Q3 + 1.5 * (Q3 - Q1)
        raw_preds = np.clip(preds_trans, None, ub)

    elif method == 'boxcox':
        # Pros: well-behaved inverse; Cons: requires positive shift; loses exact zeros
        pt = PowerTransformer(method='box-cox')
        vals = (df[column].fillna(df[column].median()) + 1).values.reshape(-1, 1)
        pt.fit(vals)
        # preds_trans is boxcox scale if scale='raw', or log1p(boxcox) if scale='log1p'
        if scale == 'log1p':
            bc_vals = np.expm1(preds_trans)
        else:
            bc_vals = preds_trans
        # inverse_transform returns (value + 1) => subtract shift
        inv = pt.inverse_transform(bc_vals.reshape(-1, 1))[:, 0] - 1
        raw_preds = inv

    elif method == 'arctan':
        # Pros: smooth inverse; Cons: limited to [-pi/2,pi/2] mapping
        raw_preds = np.tan(preds_trans / upper_threshold) * upper_threshold

    elif method == 'rank':
        # Pros: maps percentiles; Cons: loses metric spacing, approximate via empirical quantiles
        if not (0 <= preds_trans).all() or not (preds_trans <= 1).all():
            raise ValueError("Rank inverse requires predictions between 0 and 1")
        series = df[column].dropna().sort_values()
        raw_preds = series.quantile(preds_trans).values

    else:
        raise ValueError(f"Unknown method '{method}'")

    return raw_preds

def transform_ahi(arr, method='boxcox', scale='log1p', upper_threshold=100):
    """
    Transform AHI values for regression modeling and optionally apply log1p.

    # Example usage:
    # raw = df['ahi']
    # X_wins = transform_ahi(raw, method='winsor', scale='log1p')
    # X_box = transform_ahi(raw, method='boxcox', scale='raw')

    Parameters
    ----------
    arr : array-like
        Raw AHI values.
    method : {'winsor', 'iqr', 'boxcox', 'arctan', 'rank'}, default 'boxcox'
        Transformation to apply:
        - winsor: Cap at 95th percentile
          Pros: limits outliers without removing data
          Cons: arbitrary cutpoint; may distort tail
        - iqr:  Cap at Q3 + 1.5*IQR
          Pros: data-driven cap; robust to extreme outliers
          Cons: still allows moderate extremes; depends on sample IQR
        - boxcox: Box-Cox on (x+1)
          Pros: often yields near-normal; optimizes lambda
          Cons: requires positive shift; loses exact zero interpretation
        - arctan: arctan(x/upper_threshold)*upper_threshold
          Pros: smoothly compresses large values; bounded output
          Cons: nonlinear scale; less intuitive mapping
        - rank: percentile rank [0,1]
          Pros: uniform, robust to outliers
          Cons: loses original scale meaning; only relative ordering
    scale : {'raw', 'log1p'}, default 'log1p'
        If 'log1p', return log1p of the transformed values; otherwise return raw-transformed.
    upper_threshold : float, default 100
        Threshold parameter for arctan scaling.

    Returns
    -------
    numpy.ndarray
        Transformed (and optionally log1p-scaled) AHI values.
    """
    # Convert to pandas Series for convenience
    s = pd.Series(arr).astype(float)

    # Apply chosen transformation
    if method == 'winsor':
        transformed = winsorize(s, limits=[0, 0.05])
    elif method == 'iqr':
        Q1, Q3 = s.quantile([0.25, 0.75])
        ub = Q3 + 1.5 * (Q3 - Q1)
        transformed = s.clip(upper=ub)
    elif method == 'boxcox':
        pt = PowerTransformer(method='box-cox')
        vals = (s + 1).fillna((s + 1).median()).values.reshape(-1, 1)
        transformed = pt.fit_transform(vals)[:, 0]
    elif method == 'arctan':
        transformed = np.arctan(s / upper_threshold) * upper_threshold
    elif method == 'rank':
        transformed = s.rank(pct=True)
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Optionally apply log1p
    if scale == 'log1p':
        out = np.log1p(transformed)
    elif scale == 'raw':
        out = transformed
    else:
        raise ValueError("scale must be 'raw' or 'log1p'")

    return np.asarray(out)

# coefficient

# def _inverse_transform_betas(
#     betas,
#     df: pd.DataFrame,
#     column: str = 'ahi',
#     method: str = 'boxcox',
#     scale: str = 'log1p',
#     upper_threshold: float = 100.0
# ):
#     """
#     Recursively inverse-transform regression coefficients (or tuples of them)
#     from the transformed AHI scale back to the raw AHI scale, using the same
#     logic as `inverse_transform_ahi`.
#
#     Parameters
#     ----------
#     betas : float or tuple/list of floats
#         A single coefficient or a nested structure (tuple/list) of coefficients
#         corresponding to e.g. (coef, LCL, UCL).
#     df : pandas.DataFrame
#         Original DataFrame containing the raw AHI column for fitting the transformer.
#     column : str
#         Name of the raw AHI column in `df`.
#     method : str
#         Transformation method used (e.g., 'boxcox', 'winsor', etc.).
#     scale : str
#         Scaling applied ('log1p' or 'raw').
#     upper_threshold : float
#         Parameter for arctan scaling if used.
#
#     Returns
#     -------
#     Same structure as `betas`, with each numeric value inverted to the raw scale.
#     """
#     # Base case: single numeric beta
#     if isinstance(betas, (int, float, np.floating, np.integer)):
#         return inverse_transform_ahi(
#             preds=[betas],
#             df=df,
#             column=column,
#             method=method,
#             scale=scale,
#             upper_threshold=upper_threshold
#         )[0]
#
#     # Recursive case: tuple or list
#     if isinstance(betas, (tuple, list)):
#         transformed = [
#             _inverse_transform_betas(
#                 b, df, column, method, scale, upper_threshold
#             ) for b in betas
#         ]
#         return type(betas)(transformed)
#     # 3) String containing a parenthesized tuple of numbers
#     if isinstance(betas, str):
#         # Extract all floats (including negatives and decimals)
#         nums = re.findall(r'-?\d+\.?\d*', betas)
#         if not nums:
#             # No numbers to invert, return original
#             return betas
#         # Convert to floats
#         vals = [float(n) for n in nums]
#         # Invert each
#         inv_vals = [
#             _inverse_transform_betas(v, df, column, method, scale, upper_threshold)
#             for v in vals
#         ]
#         # Re-format as "(v1, v2, ...)"
#         formatted = ", ".join(f"{v:.3f}" for v in inv_vals)
#         return f"({formatted})"
#     # Otherwise, unsupported type
#     raise TypeError(f"Cannot inverse-transform object of type {type(betas)}")

def _inverse_transform_betas(
    betas,
    df: pd.DataFrame,
    column: str = 'ahi',
    method: str = 'boxcox',
    scale: str = 'log1p',
    upper_threshold: float = 100.0,
    baseline_pred: float = None
):
    if isinstance(betas, (int, float, np.floating, np.integer)):
        if baseline_pred is None:
            # Use median transformed AHI as baseline
            transformed_ahi = transform_ahi(df[column], method=method, scale=scale)
            baseline_pred = np.median(transformed_ahi)
        # Compute AHI at baseline and baseline + beta
        ahi1 = inverse_transform_ahi([baseline_pred], df, column, method, scale, upper_threshold)[0]
        ahi2 = inverse_transform_ahi([baseline_pred + betas], df, column, method, scale, upper_threshold)[0]
        return ahi2 - ahi1  # Effect on raw AHI scale

    if isinstance(betas, (tuple, list)):
        return type(betas)([
            _inverse_transform_betas(b, df, column, method, scale, upper_threshold, baseline_pred)
            for b in betas
        ])

    if isinstance(betas, str):
        nums = re.findall(r'-?\d+\.?\d*', betas)
        if not nums:
            return betas
        vals = [float(n) for n in nums]
        inv_vals = [
            _inverse_transform_betas(v, df, column, method, scale, upper_threshold, baseline_pred)
            for v in vals
        ]
        return f"({', '.join(f'{v:.3f}' for v in inv_vals)})"

    raise TypeError(f"Cannot inverse-transform object of type {type(betas)}")

def approximate_beta_effect(
    beta: float,
    df: pd.DataFrame,
    column: str = 'ahi',
    method: str = 'boxcox',
    scale: str = 'log1p'
):
    if method != 'boxcox' or scale != 'log1p':
        raise NotImplementedError("Only implemented for boxcox with log1p")
    pt = PowerTransformer(method='box-cox')
    vals = (df[column].fillna(df[column].median()) + 1).values.reshape(-1, 1)
    pt.fit(vals)
    lambda_ = pt.lambdas_[0]
    if abs(lambda_) < 0.1:  # Near-log transformation
        percentage_change = np.expm1(beta) - 1
        return f"{percentage_change * 100:.1f}% change in AHI per unit increase"
    else:
        # Fallback to difference-based method
        return _inverse_transform_betas(beta, df, column, method, scale)


def apply_inverse_transform_betas_to_df(
        df_summary: pd.DataFrame,
        columns: list[str],
        df_original: pd.DataFrame,
        column: str = 'ahi',
        method: str = 'boxcox',
        scale: str = 'log1p',
        upper_threshold: float = 100.0
) -> pd.DataFrame:
    """
    Inverse-transform regression coefficients in specified columns of a summary DataFrame
    back to the raw AHI scale.

    Parameters
    ----------
    df_summary : pd.DataFrame
        DataFrame containing the transformed coefficients (floats or tuples).
    columns : list of str
        Column names in df_summary to inverse-transform.
    df_original : pd.DataFrame
        Original DataFrame used for fitting, containing raw AHI column.
    column : str
        Name of the raw AHI column in df_original.
    method : str
        Transformation method used ('winsor', 'iqr', 'boxcox', etc.).
    scale : str
        Scale applied ('raw' or 'log1p').
    upper_threshold : float
        Parameter for arctan scaling if used.

    Returns
    -------
    pd.DataFrame
        A copy of df_summary with specified columns replaced by their back-transformed values.
    """

    df_new = df_summary.copy()
    for col in columns:
        df_new[col] = df_new[col].apply(
            lambda v: _inverse_transform_betas(
                v,
                df_original,
                column=column,
                method=method,
                scale=scale,
                upper_threshold=upper_threshold
            )
        )
    return df_new




# %%

import re
import textwrap


def pretty_col_name(col: str, max_width: int = None) -> str:
    """
    Formats a column name string into a pretty label for plots.
    - Replaces underscores and hyphens with spaces
    - Capitalizes words, uppercases known acronyms
    - If max_width is set, inserts '\n' to wrap lines <= max_width
    """
    ACRONYMS = ['ahi', 'oa', 'ma', 'hyp', 'ca', 'rdi']

    # 1) Normalize separators
    s = re.sub(r'[_\-]+', ' ', col)
    # 2) Capitalize or uppercase acronyms
    words = []
    for w in s.split():
        lw = w.lower()
        words.append(lw.upper() if lw in ACRONYMS else lw.capitalize())
    label = ' '.join(words)
    # 3) Optional wrap
    if max_width and isinstance(max_width, int):
        lines = textwrap.wrap(label,
                              width=max_width,
                              break_long_words=False,
                              break_on_hyphens=False)
        label = '\n'.join(lines)
    return label

# Example
examples = ['resp_oa_total', 'ahi_no_reras', 'avg_rdi_score', 'a_very_long_column_name']
for c in examples:
    print(f"{c!r} ->\n{pretty_col_name(c, max_width=20)}\n")