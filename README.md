# Tail Risk Modeling

[![Python](https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white)]()
[![NumPy](https://img.shields.io/badge/NumPy-numerical%20computing-013243?logo=numpy&logoColor=white)]()
[![Pandas](https://img.shields.io/badge/Pandas-data%20analysis-150458?logo=pandas&logoColor=white)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)]()
[![CatBoost](https://img.shields.io/badge/CatBoost-gradient%20boosting-yellow)]()
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)]()
[![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?logo=docker&logoColor=white)]()
[![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-11557C?logo=python&logoColor=white)]()

[![Made with](https://img.shields.io/badge/Made%20with-Machine%20Learning-red)]()

---

<p align="left">
  <video src="assets/20260308-1647-09.7740132.mp4" width="85%" controls></video>
</p>

---

<div style="background-color:#2b2b2b; padding:5px; border:1px solid #e0e0e0; border-left:6px solid #e0e0e0; border-radius:8px;">
<h2 style="color:#ffffff; margin:0;"> Table of contents</h2>
</div>

- [Motivation](#motivation)  
- [Problem statement](#problem-statement)  
- [Dataset](#dataset)  
- [Evaluation & Metrics](#evaluation--metrics)  
- [Key results (summary)](#key-results-summary)   

---

<div style="background-color:#2b2b2b; padding:5px; border:1px solid #e0e0e0; border-left:6px solid #e0e0e0; border-radius:8px;">
<h2 style="color:#ffffff; margin:0;"> Motivation</h2>
</div>

In most insurance portfolios, the majority of claims are relatively small, while a small number of catastrophic events account for a disproportionate share of total losses. These rare but high-severity claims are the primary drivers of reserve requirements, capital allocation, and long-term insurer solvency.
Traditional regression approaches focus on predicting the average claim outcome, which can lead to systematic underestimation of extreme losses. In practice, however, understanding the behavior of the upper tail of the loss distribution is far more important for risk management.

In this project, I address this gap by explicitly modeling the upper quantiles of claim severity, enabling more reliable estimation of potential extreme losses.

---

<div style="background-color:#2b2b2b; padding:5px; border:1px solid #e0e0e0; border-left:6px solid #e0e0e0; border-radius:8px;">
<h2 style="color:#ffffff; margin:0;"> Problem statement</h2>
</div>

Modeling insurance claim severity is challenging due to the highly skewed and heavy-tailed nature of loss distributions. Most claims cluster around relatively small values, while extreme losses appear infrequently but can be several orders of magnitude larger. This imbalance makes conventional regression approaches difficult to apply effectively, as models optimized for average prediction error tend to be dominated by the bulk of small claims.

The goal of this project is to develop a modeling framework that can reliably estimate upper-tail risk in claim severity. Instead of predicting the conditional mean loss, the model is designed to estimate conditional quantiles of the loss distribution, such as the 90th percentile, for a given set of claim features.

By focusing on quantile estimation, the system aims to produce stable and interpretable risk thresholds for unseen claims. These thresholds provide a practical way to characterize potential extreme outcomes within the data and evaluate how claim characteristics influence the upper tail of the loss distribution.

---

<div style="background-color:#2b2b2b; padding:5px; border:1px solid #e0e0e0; border-left:6px solid #e0e0e0; border-radius:8px;">
<h2 style="color:#ffffff; margin:0;"> Dataset</h2>
</div>

This project uses the **Allstate Claims Severity** dataset

### **Summary**

- ~188,000 records (historical insurance claims)  
- Continuous features: 14 normalized numeric attributes (`cont1`–`cont14`)  
- Categorical features: 116 high-cardinality categorical variables (`cat1`–`cat116`)  
- Target: claim severity (loss amount) — strongly right-skewed / heavy tail  
- Features are anonymized (simulates enterprise constraints)

---

**Why quantile regression?**

- Quantile regression directly targets a chosen tail level (e.g., 90th percentile) rather than the mean.  
- For risk management, predicting a conservative threshold is much more useful than a point mean.  
- Quantile predictions feed naturally into reserving, pricing margins, capital allocation, and stress tests.

---

<div style="background-color:#2b2b2b; padding:5px; border:1px solid #e0e0e0; border-left:6px solid #e0e0e0; border-radius:8px;">
<h2 style="color:#ffffff; margin:0;"> Evaluation & metrics</h2>
</div>


This project emphasizes tail-specific evaluation metrics:

1. **Pinball Loss (Quantile Loss)** — asymmetric loss used to train & compare quantile models. Heavier penalty for underprediction for upper quantiles.  
2. **D² (Quantile R²)** — a quantile analogue of R² that measures improvement relative to an unconditional quantile baseline.  
3. **Coverage (Calibration)** — proportion of observed claims that fall below the predicted quantile; for τ=0.90, ideal coverage ≈ 0.90.

**Why these matter:** Good pinball loss indicates low quantile error; high D² shows explanatory power relative to a naive baseline; correct coverage means the model is well-calibrated for risk thresholds.

---

<div style="background-color:#2b2b2b; padding:5px; border:1px solid #e0e0e0; border-left:6px solid #e0e0e0; border-radius:8px;">
<h2 style="color:#ffffff; margin:0;"> Key results</h2>
</div>

- Quantile targeted: **τ = 0.90**  
- Mean pinball loss (τ=0.90): **341.46**  
- D² (quantile R²): **0.4859** (≈ 48.6% improvement vs. naive unconditional quantile)  
- Empirical coverage: **0.8940** (very close to nominal 0.90 — well-calibrated)

---