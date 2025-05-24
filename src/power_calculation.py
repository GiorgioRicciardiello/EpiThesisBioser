import numpy as np
import scipy.stats as stats
from scipy.stats import spearmanr, pointbiserialr
from config.config import config, metrics_psg, encoding, sections
import pandas  as pd

def power_for_corr(r, n, alpha=0.05):
    """
    Compute post-hoc power to detect a correlation of magnitude r
    in a sample of size n at significance level alpha (two-sided).
    """
    # Fisher Z-transform of the effect size
    z_r = np.arctanh(r)
    # Standard error of Z
    se  = 1 / np.sqrt(n - 3)
    # two-sided critical Z
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # non-centrality parameter
    z_effect = z_r / se
    # Type II error β
    beta = stats.norm.cdf(z_alpha - z_effect) - stats.norm.cdf(-z_alpha - z_effect)
    return 1 - beta

# --- Example usage ---
# Assume your DataFrame is `df` with these columns:
#   - 'ahi'      : continuous Apnea–Hypopnea Index
#   - 'sym_cont' : continuous symptom score (e.g., fatigue scale)
#   - 'sym_ord'  : ordinal symptom (e.g., Likert 1–5)
#   - 'sym_bin'  : binary symptom (0 = no, 1 = yes)
if __name__ == '__main__':
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)

    # Subset and drop missing data
    sub = df[['ahi', 'sym_cont', 'sym_ord', 'sym_bin']].dropna()
    n   = len(sub)
    print(f"Sample size: {n}")

    # 1. Continuous outcome → Pearson's r
    #    Measures linear correlation between two continuous variables
    r_cont = sub['ahi'].corr(sub['sym_cont'])
    power_cont = power_for_corr(r_cont, n)
    print(f"Continuous symptom: Pearson r = {r_cont:.3f}, power ≈ {power_cont:.1%}")

    # 2. Ordinal outcome → Spearman’s ρ
    #    Measures rank‐based (monotonic) correlation for ordinal data
    r_ord, _   = spearmanr(sub['ahi'], sub['sym_ord'])
    power_ord  = power_for_corr(r_ord, n)
    print(f"Ordinal symptom: Spearman ρ = {r_ord:.3f}, power ≈ {power_ord:.1%}")

    # 3. Binary outcome → Point‐biserial r
    #    Measures correlation between a binary variable and a continuous variable
    r_bin, _   = pointbiserialr(sub['sym_bin'], sub['ahi'])
    power_bin  = power_for_corr(r_bin, n)
    print(f"Binary symptom: Point-biserial r = {r_bin:.3f}, power ≈ {power_bin:.1%}")
