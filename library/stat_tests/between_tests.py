import statsmodels.formula.api as smf
import pandas  as pd



def doctor_variation_ai(df,
                        ai_col: str = 'ai',
                        severity_col: str = 'osa_four',
                        doctor_col: str = 'md_identifyer',
                        min_patients_per_doc: int = 5,
                        tol_var: float = 1e-8):
    """
    Quantify between‐doctor variance in continuous AI scores and adjust for OSA severity.

    1. Null model: ai_{p,d} = μ + u_d + ε_{p,d}
         ICC = σ²_u / (σ²_u + σ²_e)

    2. Adjusted model: ai_{p,d} = β₀
                             + β₁·I[osa_four=Moderate]
                             + β₂·I[osa_four=Severe]
                             + u_d + ε_{p,d}

    If σ²_u < tol_var or covariance is singular, we set all û_d = 0.
    """
    # Prepare data
    df_work = df[[ai_col, severity_col, doctor_col]].dropna()
    counts = df_work[doctor_col].value_counts()
    keep = counts[counts >= min_patients_per_doc].index
    df2 = df_work[df_work[doctor_col].isin(keep)].reset_index(drop=True)
    df2[severity_col] = df2[severity_col].astype('category')

    # 1) Null model for ICC
    m0 = smf.mixedlm(f"{ai_col} ~ 1", df2, groups=df2[doctor_col]).fit()
    var_u = m0.cov_re.iloc[0, 0]
    var_e = m0.scale
    icc = var_u / (var_u + var_e)
    print(f"ICC (doctor clustering) = {icc:.3f}")

    # 2) Adjusted model
    m1 = smf.mixedlm(f"{ai_col} ~ C({severity_col})",
                     df2,
                     groups=df2[doctor_col]).fit(method='lbfgs')
    print(m1.summary())

    # 3) Doctor‐level BLUPs
    if var_u < tol_var:
        # No between‐doctor variance → all effects = 0
        doctors = df2[doctor_col].unique()
        doc_effects = pd.DataFrame({
            'doctor': doctors,
            'doctor_intercept': 0.0
        }).sort_values('doctor_intercept', ascending=False)
        print("Between-doctor variance ≈ 0; setting all BLUPs to zero.")
    else:
        try:
            re = m1.random_effects
            doc_effects = (
                pd.DataFrame.from_dict(re, orient='index', columns=['doctor_intercept'])
                .assign(doctor=lambda d: d.index)
                .sort_values('doctor_intercept', ascending=False)
            )
        except ValueError:
            # Singular covariance, fallback to zeros
            doctors = df2[doctor_col].unique()
            doc_effects = pd.DataFrame({
                'doctor': doctors,
                'doctor_intercept': 0.0
            }).sort_values('doctor_intercept', ascending=False)
            print("Singular random-effects covariance; setting all BLUPs to zero.")

    # 4) Save and return
    return m0, m1, icc, doc_effects