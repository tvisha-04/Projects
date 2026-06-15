import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


# 1. DATA INGESTION
print("[INFO] Fetching WTI front-month futures data from yfinance...")

oil_data = yf.download('CL=F', start='2016-01-01', end='2026-01-01', progress=False)

# yfinance returns a MultiIndex for single-ticker downloads; flatten it
if isinstance(oil_data.columns, pd.MultiIndex):
    oil_data.columns = oil_data.columns.get_level_values(0)

oil_df = oil_data.reset_index().dropna(subset=['Close'])
df = pd.DataFrame({'Date': oil_df['Date'], 'Spot_Price': oil_df['Close'].astype(float)})
df = df.reset_index(drop=True)


# 2. SYNTHETIC MACRO COVARIATES
# in production these come from NOAA (weather) and EIA weekly petroleum reports (inventory)
n = len(df)
np.random.seed(42)

df['Weather_Volatility_Index'] = np.abs(np.random.normal(loc=1.5, scale=0.8, size=n))  # half-normal; volatility is strictly positive
df['Inventory_Drawdown_Ratio'] = np.random.uniform(0.5, 1.5, size=n)  # >1 = net draw, <1 = net build


# 3. COST-OF-CARRY FORWARD PRICE & IMPLIED CONVENIENCE YIELD
# F = S * exp((r + c) * τ) + demand_shock  →  y = r + c - (1/τ) * ln(F/S)
r   = 0.05       # annualised risk-free rate
c   = 0.02       # annualised physical storage cost (tank lease + insurance)
tau = 30 / 365   # 1-month maturity window

demand_shock = (df['Inventory_Drawdown_Ratio'] * df['Weather_Volatility_Index']) * 1.25
df['Forward_Price_1M'] = df['Spot_Price'] * np.exp((r + c) * tau) + demand_shock

# positive y → backwardation (scarcity); negative y → contango (ample supply)
df['Implied_Convenience_Yield'] = (r + c) - (1 / tau) * np.log(df['Forward_Price_1M'] / df['Spot_Price'])


# 4. TARGET — 5-DAY FORWARD YIELD
# predicting 5 days out is short enough to be actionable but filters intraday noise
df['Yield_T5'] = df['Implied_Convenience_Yield'].shift(-5)
df = df.dropna().reset_index(drop=True)


# 5. FEATURE MATRIX & CHRONOLOGICAL TRAIN / TEST SPLIT
# random splits leak future data into training for time series; always split by index
features = ['Spot_Price', 'Weather_Volatility_Index', 'Inventory_Drawdown_Ratio', 'Implied_Convenience_Yield']

X = df[features]
y = df['Yield_T5']

split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


# 6. ROBUST SCALING
# RobustScaler uses median + IQR so fat-tailed energy price spikes don't dominate the feature space
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)


# 7. RANDOM FOREST REGRESSOR
# tree ensembles capture threshold effects in inventory and asymmetric weather responses naturally
rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
preds = rf.predict(X_test_scaled)


# 8. DIAGNOSTICS
# using MAE instead of MAPE — convenience yields can be near zero or negative, making % errors unstable
r2  = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)

print("\n" + "=" * 50)
print("      CRUDE OIL CONVENIENCE YIELD MODEL")
print("=" * 50)
print(f"  R²  (Explained Variance)  :  {r2:.4f}")
print(f"  MAE (Yield Points)        :  {mae:.6f}")
print("\n  Feature Importances")
print("  " + "-" * 40)
for name, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"    {name:<32}  {imp * 100:.2f}%")
print("=" * 50)


# 9. VISUALISATION
print("\n[INFO] Rendering diagnostic plots...")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#0f1923')

for ax in axes:
    ax.set_facecolor('#0f1923')
    ax.tick_params(colors='#cdd6f4')
    ax.xaxis.label.set_color('#cdd6f4')
    ax.yaxis.label.set_color('#cdd6f4')
    ax.title.set_color('#cdd6f4')
    for spine in ax.spines.values():
        spine.set_edgecolor('#313244')

# panel A — feature importances sorted ascending for readability
importances     = rf.feature_importances_
sorted_idx      = np.argsort(importances)
sorted_features = [features[i] for i in sorted_idx]

axes[0].barh(range(len(sorted_idx)), importances[sorted_idx], color='#89b4fa', edgecolor='#313244', height=0.55)
axes[0].set_yticks(range(len(sorted_idx)))
axes[0].set_yticklabels(sorted_features, fontsize=11)
axes[0].set_xlabel('Relative Importance', fontsize=11)
axes[0].set_title('Feature Importance — RF Ensemble', fontsize=12, fontweight='bold', pad=12)
axes[0].grid(axis='x', linestyle=':', alpha=0.3, color='#585b70')

# panel B — actual vs predicted over the first 100 test days
axes[1].plot(y_test.values[:100], label='Actual Yield',   color='#f38ba8', lw=2.2, alpha=0.9)
axes[1].plot(preds[:100],         label='Model Forecast', color='#a6e3a1', lw=2.2, linestyle='--')
axes[1].set_xlabel('Days (Test Window)', fontsize=11)
axes[1].set_ylabel('Implied Convenience Yield  y(t)', fontsize=11)
axes[1].set_title('Actual vs Forecasted — 5-Day Ahead Yield', fontsize=12, fontweight='bold', pad=12)
axes[1].legend(fontsize=10, facecolor='#1e1e2e', edgecolor='#313244', labelcolor='#cdd6f4')
axes[1].grid(True, linestyle=':', alpha=0.3, color='#585b70')

plt.tight_layout(pad=2.0)
plt.savefig('crude_oil_convenience_yield.png', dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
print("[INFO] Chart saved → crude_oil_convenience_yield.png")
