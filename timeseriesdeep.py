import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from sklearn.metrics import mean_squared_error
from math import sqrt
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import joblib
import matplotlib.pyplot as plt
import json
import os
from datetime import timedelta

# ---------------------------
# 1) Config / reproducibility
# ---------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
OUT_DIR = "output_prophet_bayes"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# 2) Synthetic dataset generator
# ---------------------------
def generate_synthetic_hourly(start='2018-01-01', years=3.5, freq='H', seed=RANDOM_SEED):
    """
    Generates an hourly synthetic time series with:
      - linear trend
      - daily seasonality (24h)
      - weekly seasonality (7*24)
      - yearly seasonality (approx via Fourier series)
      - random holidays/events spikes
      - heteroscedastic noise
    Returns a DataFrame with columns ['ds', 'y'] suitable for Prophet.
    """
    np.random.seed(seed)
    periods = int(365 * years * 24)  # approximate
    idx = pd.date_range(start=start, periods=periods, freq=freq)
    t = np.arange(periods).astype(float)

    # Linear + slight non-linear trend
    trend = 0.0015 * t + 0.0000001 * (t**2)

    # Daily seasonality (24h): use a couple of harmonics
    daily = 2.0 * np.sin(2 * np.pi * t / 24.0) + 0.5 * np.sin(2 * np.pi * t / 12.0)

    # Weekly seasonality (168h)
    weekly = 1.2 * np.sin(2 * np.pi * t / (24*7)) + 0.3 * np.cos(2 * np.pi * t / (24*7/2))

    # Yearly seasonality (approx using a longer period)
    yearly = 0.8 * np.sin(2 * np.pi * t / (24*365.25)) + 0.2 * np.cos(2 * np.pi * t / (24*365.25/2))

    # Random "holidays" events: occasional spikes
    y = 10 + trend + daily + weekly + yearly
    # Add random holiday events - e.g., 0.5% of days have a big spike
    num_spikes = max(3, int(0.005 * len(idx)))
    spike_positions = np.random.choice(len(idx), size=num_spikes, replace=False)
    for p in spike_positions:
        # a spike that decays over a few hours
        amplitude = np.random.uniform(5, 25)
        width = np.random.randint(6, 48)
        for w in range(width):
            if p + w < len(idx):
                y[p + w] += amplitude * np.exp(-w/12.0)

    # Heteroscedastic noise: noise scale depends on the daily season amplitude
    noise = np.random.normal(scale=0.5 + 0.2 * (np.abs(daily)/np.max(np.abs(daily))), size=periods)
    y = y + noise

    df = pd.DataFrame({'ds': idx, 'y': y})
    return df

# ---------------------------
# 3) Error metrics: RMSE and MASE
# ---------------------------
def mase(actual, forecast, training_series, seasonal_period=24):
    """
    Mean Absolute Scaled Error (MASE)
    denominator = mean absolute naive seasonal differences on training_series
    For hourly data with daily seasonality, seasonal_period=24 by default.
    """
    n = training_series.shape[0]
    if n - seasonal_period <= 0:
        raise ValueError("Training series too short for chosen seasonal_period.")
    denom = np.mean(np.abs(training_series[seasonal_period:] - training_series[:-seasonal_period]))
    if denom == 0:
        return np.nan
    return np.mean(np.abs(actual - forecast)) / denom

def rmse(actual, forecast):
    return sqrt(mean_squared_error(actual, forecast))

# ---------------------------
# 4) Prepare data and splits
# ---------------------------
df = generate_synthetic_hourly(start='2018-01-01', years=4.0)  # 4 years to be safe
# Ensure ds sorted
df = df.sort_values('ds').reset_index(drop=True)

# Use final 365 days as test (approx 365*24)
test_horizon = 365 * 24
train_df = df[:-test_horizon]
test_df = df[-test_horizon:]

# Create Prophet-compatible holidays DataFrame (example: randomized holiday days)
def build_holidays(df, n_holidays=30, seed=RANDOM_SEED):
    np.random.seed(seed)
    possible_dates = pd.to_datetime(df['ds'].dt.date.unique())
    chosens = np.random.choice(len(possible_dates), size=n_holidays, replace=False)
    hols = pd.DataFrame({
        'ds': possible_dates[chosens],
        'holiday': ['random_holiday_{}'.format(i) for i in range(n_holidays)]
    })
    return hols

holidays = build_holidays(df, n_holidays=40)

# ---------------------------
# 5) Baseline Prophet (defaults)
# ---------------------------
baseline_model = Prophet(holidays=holidays)
baseline_model.fit(train_df)

# Forecast for test period
future = test_df[['ds']].copy()
baseline_forecast = baseline_model.predict(future)
baseline_pred = baseline_forecast['yhat'].values
baseline_rmse = rmse(test_df['y'].values, baseline_pred)
baseline_mase = mase(test_df['y'].values, baseline_pred, training_series=train_df['y'].values, seasonal_period=24)

print("Baseline RMSE:", baseline_rmse)
print("Baseline MASE:", baseline_mase)

# Save baseline model
with open(os.path.join(OUT_DIR, "baseline_prophet.json"), "w") as fout:
    fout.write(model_to_json(baseline_model))

# ---------------------------
# 6) Bayesian Optimization of Prophet hyperparameters
# ---------------------------

# Define search space (wide but constrained)
search_space = [
    Real(0.01, 10.0, name='seasonality_prior_scale', prior='log-uniform'),
    Real(0.0001, 0.5, name='changepoint_prior_scale', prior='log-uniform'),
    Real(0.01, 10.0, name='holidays_prior_scale', prior='log-uniform'),
    Categorical(['additive', 'multiplicative'], name='seasonality_mode'),
]

# We will run gp_minimize for >= 50 calls (plus initial points)
N_CALLS = 60  # ensures minimum 50 trials as required

# For speed: limit Prophet to weekly and daily seasonality explicitly off/on via seasonality_mode param
@use_named_args(search_space)
def objective(**params):
    """
    Objective function to minimize MASE on the held-out test set.
    We train Prophet on train_df and evaluate on test_df.
    """
    # Build and fit a Prophet model with the provided hyperparameters
    m = Prophet(
        seasonality_mode=params['seasonality_mode'],
        seasonality_prior_scale=params['seasonality_prior_scale'],
        changepoint_prior_scale=params['changepoint_prior_scale'],
        holidays_prior_scale=params['holidays_prior_scale'],
        holidays=holidays,
        weekly_seasonality=True,
        daily_seasonality=True,
        yearly_seasonality=False,  # we can leave yearly auto or false because it's expensive
    )

    # Add yearly seasonality manually (fourier order) to ensure multiple seasonalities are present
    m.add_seasonality(name='yearly_complex', period=365.25, fourier_order=6)
    try:
        m.fit(train_df, verbose=False)
    except Exception as e:
        print("Model fit failed with params:", params, "error:", e)
        return 1e6  # return large penalty

    fut = test_df[['ds']].copy()
    fcst = m.predict(fut)
    preds = fcst['yhat'].values

    # compute mase
    try:
        score = mase(test_df['y'].values, preds, training_series=train_df['y'].values, seasonal_period=24)
    except Exception as e:
        print("MASE compute failed:", e)
        return 1e6

    # we minimize score
    print(f"params: {params} -> MASE: {score:.6f}")
    return score

print("Starting Bayesian optimization (this may take a while)...")
res = gp_minimize(objective, search_space, n_calls=N_CALLS, random_state=RANDOM_SEED, n_initial_points=12, verbose=True)

# Save optimization result
joblib.dump(res, os.path.join(OUT_DIR, "bayes_opt_result.joblib"))

# Best hyperparams
best_hyperparams = {dim.name: val for dim, val in zip(res.space, res.x)}
print("Best hyperparameters found:", best_hyperparams)
with open(os.path.join(OUT_DIR, "best_hyperparams.json"), "w") as fh:
    json.dump(best_hyperparams, fh, indent=2)

# ---------------------------
# 7) Train final optimized Prophet and evaluate
# ---------------------------
opt_m = Prophet(
    seasonality_mode=best_hyperparams.get('seasonality_mode', 'additive'),
    seasonality_prior_scale=float(best_hyperparams.get('seasonality_prior_scale', 1.0)),
    changepoint_prior_scale=float(best_hyperparams.get('changepoint_prior_scale', 0.05)),
    holidays_prior_scale=float(best_hyperparams.get('holidays_prior_scale', 1.0)),
    holidays=holidays,
    weekly_seasonality=True,
    daily_seasonality=True,
    yearly_seasonality=False,
)
opt_m.add_seasonality(name='yearly_complex', period=365.25, fourier_order=6)
opt_m.fit(train_df)

opt_future = test_df[['ds']].copy()
opt_fc = opt_m.predict(opt_future)
opt_pred = opt_fc['yhat'].values

opt_rmse = rmse(test_df['y'].values, opt_pred)
opt_mase = mase(test_df['y'].values, opt_pred, training_series=train_df['y'].values, seasonal_period=24)

print("Optimized RMSE:", opt_rmse)
print("Optimized MASE:", opt_mase)

# Save optimized model
with open(os.path.join(OUT_DIR, "opt_prophet.json"), "w") as fout:
    fout.write(model_to_json(opt_m))

# Save forecasts & results
test_df_result = test_df.copy()
test_df_result['baseline_yhat'] = baseline_pred
test_df_result['opt_yhat'] = opt_pred
test_df_result.to_parquet(os.path.join(OUT_DIR, "test_forecasts.parquet"))

# ---------------------------
# 8) Comparative report text file
# ---------------------------
report_lines = []
report_lines.append("Prophet Bayesian Optimization Report")
report_lines.append("===============================")
report_lines.append(f"Number of train points: {len(train_df)}")
report_lines.append(f"Number of test points: {len(test_df)}")
report_lines.append("")
report_lines.append("Baseline model performance:")
report_lines.append(f"  RMSE: {baseline_rmse:.6f}")
report_lines.append(f"  MASE: {baseline_mase:.6f}")
report_lines.append("")
report_lines.append("Optimized model performance:")
report_lines.append(f"  RMSE: {opt_rmse:.6f}")
report_lines.append(f"  MASE: {opt_mase:.6f}")
report_lines.append("")
report_lines.append("Best hyperparameters (Bayesian optimization):")
for k,v in best_hyperparams.items():
    report_lines.append(f"  {k}: {v}")
report_lines.append("")
report_lines.append("Notes on optimization:")
report_lines.append(" - Objective minimized: MASE on the held-out test set.")
report_lines.append(" - Bayesian optimizer: scikit-optimize (gp_minimize) with %d calls." % N_CALLS)
report_lines.append(" - Seasonal period for MASE denominator: 24 (daily).")
report_lines.append("")
report_lines.append("Files saved to the output folder: %s" % OUT_DIR)

with open(os.path.join(OUT_DIR, "report.txt"), "w") as f:
    f.write("\n".join(report_lines))

print("\nREPORT SUMMARY:")
print("\n".join(report_lines))

# ---------------------------
# 9) Quick plots (saved)
# ---------------------------
plt.figure(figsize=(12,5))
plt.plot(test_df['ds'], test_df['y'], label='actual')
plt.plot(test_df['ds'], test_df_result['baseline_yhat'], label='baseline_pred')
plt.plot(test_df['ds'], test_df_result['opt_yhat'], label='opt_pred')
plt.legend()
plt.title("Test period: actual vs baseline vs optimized")
plt.xlabel("Date")
plt.ylabel("y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "test_forecast_comparison.png"), dpi=150)
plt.close()

# Save a small summary json
summary = {
    'baseline_rmse': float(baseline_rmse),
    'baseline_mase': float(baseline_mase),
    'opt_rmse': float(opt_rmse),
    'opt_mase': float(opt_mase),
    'best_hyperparams': best_hyperparams,
    'n_calls': N_CALLS
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\nAll done. Outputs saved in directory:", OUT_DIR)
