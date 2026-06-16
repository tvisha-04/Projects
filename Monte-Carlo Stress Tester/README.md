
# Market Crash Trading Strategy Tester

Stress-tested four trading strategies against real S&P 500 data across three market regimes — a calm year (2019), the COVID crash (2020), and the 2008 financial crisis — using Monte Carlo simulation (100 alternate price timelines over 60 days).

**Strategies tested:** Buy & Hold, Trailing Stop Loss, Moving Average Crossover, and Mean Reversion.

---

**Conclusion**
The Trailing Stop Loss performed best in crashes by cutting losses early while keeping upside open. Buy & Hold looked decent on average but hid catastrophic losses on bad paths. Moving Average got "whipsawed" — it sold on temporary dips and missed recoveries. Mean Reversion thrived in the calm 2019 market but broke down in high-volatility crashes by buying dips that never recovered.
```
