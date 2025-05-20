#!/usr/bin/env python
"""
Script: ahi_late_fusion_xgb.py
Predict AHI using late-fusion of XGBoost component models (OA, CA, MA) with CUDA support.
"""
from config.config import config, metrics_psg
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


def extract_component_rates(df, components, hours_col='resp_total_sleep_time_hours'):
    """
    Compute hourly rates for specified respiratory events at 'total' position.
    """
    rates = {}
    for evt in components:
        raw_col = f"{evt}_total"
        rate_col = f"{evt}_rate"
        rates[rate_col] = df[raw_col] / df[hours_col]
    return pd.DataFrame(rates, index=df.index)


if __name__ == '__main__':
    # %% Load data
    df = pd.read_csv(config.get('data')['pp_data']['q_resp'], low_memory=False)
    output_path = config.get('results')['stat_tests']

    # %% Define components and target
    components = ['oa', 'ca', 'ma']
    features = extract_component_rates(df, components)
    target   = df['ahi_no_reras']  # AHI index (# events/hr)

    # %% Train/test split (80% train for CV, 20% final test)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, train_size=0.8, test_size=0.2, random_state=42
    )

    # %% Component models: 5-fold CV out-of-fold preds + train on full
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    comp_oof  = pd.DataFrame(index=X_train.index)
    comp_preds = pd.DataFrame(index=X_test.index)

    for comp in X_train.columns:
        model = XGBRegressor(
            tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
            objective='reg:squarederror', random_state=42
        )
        # OOF train predictions
        comp_oof[comp] = cross_val_predict(model, X_train, X_train[comp], cv=kf, n_jobs=-1)
        # Fit full train & predict test
        model.fit(X_train, X_train[comp])
        comp_preds[comp] = model.predict(X_test)

    # %% Meta-model: late fusion of component predictions
    meta = XGBRegressor(
        tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
        objective='reg:squarederror', random_state=42
    )
    meta.fit(comp_oof, y_train)
    y_pred = meta.predict(comp_preds)

    # %% Evaluation
    r2   = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Late-fusion AHI R²: {r2:.3f}")
    print(f"Late-fusion AHI RMSE: {rmse:.3f} events/hr")

    # %% Save results
    results = X_test.copy()
    results['ahi_true'] = y_test
    results['ahi_pred'] = y_pred
    results.to_csv(output_path / 'ahi_late_fusion_predictions.csv', index=False)
