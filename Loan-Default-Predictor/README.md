
# Loan Default Predictor

A two-part financial ML project.

**Part 1 — Loan Default Classifier**
Built a Random Forest classifier on synthetic loan data (284,807 records, 0.17% default rate), using SMOTE to handle class imbalance. Achieved a target ROC-AUC of 0.97.

**Part 2 — Stock Price Forecaster**
Forecasted next-day IBM stock closing prices using a custom Transformer with a TimeEmbedding layer, trained on 30-day rolling OHLCV windows.

---

**Conclusion**
SMOTE effectively corrects severe class imbalance for rare-event detection. A lightweight Transformer with learned time embeddings can model financial time-series well, targeting under 4% MAPE with minimal epochs.
```
