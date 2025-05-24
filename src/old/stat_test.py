from config.config import config, sections, encoder_dict, metrics_psg
from library.helper import get_mappers, classify_osa
import pandas  as pd
from library.stat_tests.between_tests import doctor_variation_ai
from library.stat_tests.stats_resp_events import  (RespiratoryPositionPipeline,
                                                   plot_event_trends_by_group,
                                                   summarize_event_counts)
from scipy.stats import (
    shapiro,
    friedmanchisquare,
    wilcoxon,
    ttest_rel
)
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

from itertools import combinations


if __name__ == '__main__':
    # %% Input data
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    output_path = config.get('results')['stat_tests']
    # %% Labels for plots
    all_map, grouped_map = get_mappers()
    # %%  Explore Variable Categories
    epworth_vars = [col for col in df.columns if col.startswith("ep")]
    ph3_vars = [col for col in df.columns if col.startswith("ph3")]
    sa_vars = [col for col in df.columns if col.startswith("sa_")]
    pre_vars = [col for col in df.columns if col.startswith("presleep")]
    post_vars = [col for col in df.columns if col.startswith("postsleep")]
    mh_vars = [col for col in df.columns if col.startswith("mh_")]
    ph2_vars = [col for col in df.columns if col.startswith("ph2_")]
    metrics_resp = [col for col in df.columns if col.startswith("resp")]

    # %% test the doctors and the AHI
    COMPUTE_DOCTOR_VARIATION = False
    if COMPUTE_DOCTOR_VARIATION:
        m0, m1, icc, doc_effects = doctor_variation_ai(df=df)

    # The mixed‐effects model you fit was:
    # \[
    # \mathit{ai}_{p,d} = \beta_{0} + \beta_{1}\,\mathbf{1}[\text{Moderate}]_{p} + \beta_{2}\,\mathbf{1}[\text{Severe}]_{p} + u_{d} + \varepsilon_{p,d},
    # \]
    # with \(u_{d}\sim N(0,\sigma^{2}_{u})\) and \(\varepsilon_{p,d}\sim N(0,\sigma^{2}_{e})\). Here is how to read the key output:
    #
    # • **Random‐effects variance (“Group Var”) = 0.000**
    #   The model estimated \(\sigma^{2}_{u}\approx 0\). In other words, once you account for OSA severity, there is essentially *no* residual variation in the AI score attributable to which doctor performed the assessment. Consequently, the algorithm could not compute nonzero BLUPs, and we set all doctor‐level intercepts to zero. This tells you that doctors do *not* systematically score higher or lower once severity is in the model.
    #
    # • **Intercept**:
    #   \(\hat\beta_{0} = 3.449\), but with a wildly inflated standard error (\(\mathit{SE}\approx 3.4\times10^{6}\)) and 95 % CI spanning \([-6.7\times10^{6}, +6.7\times10^{6}]\).
    #   This instability stems from the singular random‐effects covariance: when \(\sigma^{2}_{u}\approx0\), the mixed‐model intercept’s sampling distribution becomes numerically ill-conditioned. You should *not* overinterpret this intercept estimate or its SE/CI.
    #
    # • **Moderate vs. Mild OSA (β₁)**:
    #   \(\hat\beta_{1} = 3.012\) (SE = 0.176; z = 17.08; *p* < 10⁻¹⁵), 95 % CI [2.667, 3.358].
    #   After adjusting for clustering by doctor (even though the variance came out zero), Moderate‐OSA patients score on average ~3.0 units higher on the AI scale than Mild‐OSA patients.
    #
    # • **Severe vs. Mild OSA (β₂)**:
    #   \(\hat\beta_{2} = 11.749\) (SE = 0.192; z = 61.22; *p* < 10⁻⁶⁰), 95 % CI [11.373, 12.125].
    #   Severe‐OSA patients have an even larger shift, scoring about 11.7 units higher than those with Mild OSA.
    #
    # ---
    #
    # **Epidemiological interpretation**: once you control for OSA severity, doctors *do not* contribute detectable systematic bias to the continuous AI score (ICC ≈ 0). The vast majority of variability in AI is explained by patient severity (Moderate vs. Severe), not by the identity of the diagnosing physician. The huge standard error on the intercept is a numerical artifact of the zero‐variance estimate and should be disregarded—focus instead on the precise, highly significant severity coefficients.

    # %% Evaluate the obstructive apneas, central apneas, hyponeas and rera indices
    covariats = []
    events = ['oa', 'ca', 'ma', 'hyp_4_30_airflow_reduction_desaturation',
       'rera_5_hyp_with_arousal_without_desaturation',]
    event = events[0]
    pos = 'total'
    t_col = 'resp-position-total'  #  metrics_psg.get('resp_events')['position_keywords']
    cnt_col = f"resp-{event}-{pos}"
    df[[t_col, cnt_col]]

    # %% Explore how OA, CA and MA counts vary by age, gender and BMI:
    # Overall summary
    events = ['oa', 'ca', 'ma']
    summary_global = summarize_event_counts(df, events)
    print(summary_global)

    # By age and BMI groups
    group_specs = {
        'age_group': {
            'col': 'age',
            'bins': [0, 30, 45, 60, 100],
            'labels': ['<30', '30–45', '45–60', '60+']
        },
        'bmi_group': {
            'col': 'bmi',
            'bins': [0, 18.5, 25, 30, 100],
            'labels': ['Underweight', 'Normal', 'Overweight', 'Obese']
        }
    }

    # 2) Auto‐create your group columns
    for grp_name, spec in group_specs.items():
        df[grp_name] = pd.cut(
            df[spec['col']],
            bins=spec['bins'],
            labels=spec['labels']
        )

    # 3) Define events and grouping cols
    events = ['oa', 'ca', 'ma']
    group_cols = ['gender'] + list(group_specs.keys())

    # 4) Summarize
    summary_by_group = summarize_event_counts(df, events, group_cols)
    print(summary_by_group)

    events = ['oa', 'ca', 'ma']

    plot_event_trends_by_group(
        df,
        events,
        age_group_col='age_group',
        bmi_group_col='bmi_group',
        count_fmt='resp-{}-total',
        figsize_per_plot=(4, 4)
    )

    # %% Evalaute the relationship, the counts
    formula = 'Q("resp-oa-total") ~ age + C(gender) + bmi'

    for var in ['age', 'bmi']:
        sns.lmplot(x=var,
                   y='resp-oa-total',
                   data=df,
                   lowess=True,
                   scatter_kws={'alpha': 0.5},
                   line_kws={'color': 'red'})
        plt.title(f"OA count vs {var} (LOWESS)")
        plt.show()

    # Fit NB
    nb = smf.glm(formula, data=df, family=sm.families.NegativeBinomial()).fit()
    print(nb.summary())

    import statsmodels.discrete.count_model as cm

    # ZINB
    zinb = cm.ZeroInflatedNegativeBinomialP.from_formula(
        formula, df, p_formula="age + C(gender) + bmi"
    ).fit(method="bfgs")
    print(zinb.summary())

    from pygam import PoissonGAM, s, f
    df_model = df.loc[~df['resp-oa-total'].isna(), ['age', 'gender', 'bmi', 'resp-oa-total']]
    # s(age) and s(bmi) are smooth terms; f(gender) is categorical
    gam = PoissonGAM(s(0) + f(1) + s(2)).fit(
        df_model[['age', 'gender', 'bmi']].values,
        df_model['resp-oa-total'].values
    )
    gam.summary()

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    X = pd.get_dummies(df_model[['age', 'gender', 'bmi']], drop_first=True)
    y = df_model['resp-oa-total']

    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=10)
    scores = cross_val_score(rf, X, y, scoring='neg_mean_squared_error', cv=5)
    print("MSE:", -scores.mean())
    from sklearn.inspection import plot_partial_dependence

    rf.fit(X, y)
    plot_partial_dependence(rf, X, ['age', 'bmi'])
    plt.show()



    # %%Evaluate the indices
    metrics_psg.get('resp_events')['indices']
    # %% Statistical test of the respiratory measures
    ID_COL = 'id'  # ← your subject identifier column
    POS_TIME_PREFIX = 'resp-position'
    events = ['oa', 'ca', 'ma', 'hyp_4_30_airflow_reduction_desaturation',
       'rera_5_hyp_with_arousal_without_desaturation', 'apnea',
       'apnea+hyp', 'apnea+hyp+rera', 'rdi_with_reras', 'ahi_no_reras',
       'hi_hypopneas_only', 'ai_apneas_only', 'ri_rera_only']

    for event in events:
        pipeline = RespiratoryPositionPipeline(
            df=df,
            metrics=metrics_psg,
            id_col='id',
            pos_time_prefix='resp-position',
            output_path=output_path,
            min_time=10.0     # drop any position‐epochs <10 min
        )
        pipeline.run(event=event)

    event= 'ahi_no_reras'
    # single event and mask positional
    pipeline = RespiratoryPositionPipeline(
        df=df,
        metrics=metrics_psg,
        id_col='id',
        pos_time_prefix='resp-position',
        output_path=output_path,
        min_time=10.0  # drop any position‐epochs <10 min
    )
    pipeline.run(event=event)

    mask_ahi = pipeline.label_positional_osa(
        event=event,
        lateral_positions=['left', 'right'],
        require_rem_supine=False
    )

    # attach it back to your original wide‐format DataFrame
    df[f'positional_{event}'] = df['id'].map(mask_ahi)

    # view how many patients are flagged
    print(df['positional_{event}'].sum(), "out of", len(df), "subjects have positional OSA by AHI.")

