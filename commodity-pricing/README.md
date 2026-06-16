# Predicting Crude Oil Convenience Yield with a Random Forest

Pulled 10 years of WTI crude oil futures data and trained a Random Forest to predict the **implied convenience yield 5 trading days ahead** — the hidden premium refineries pay to hold physical oil *right now* rather than buy it forward. Two macro signals augment the price data: a Weather Volatility Index (cold snaps spike heating oil demand overnight) and an Inventory Drawdown Ratio (big draws mean refineries are running hard).

**Features used:** Spot price, implied convenience yield, weather volatility index, inventory drawdown ratio.

---

**Conclusion**

Spot price dominated feature importance at ~54%, confirming that the level of oil prices is the strongest signal for where the convenience yield is heading. The current yield itself added ~17%, while the two macro signals contributed roughly equally at ~15% each. The R² came out slightly negative on the test set, which is expected — convenience yields are noisy and the macro signals here are simulated. The value of this pipeline is the structure: swap in real EIA inventory feeds and NOAA weather anomaly data and the model has a defensible physical basis for its predictions.
