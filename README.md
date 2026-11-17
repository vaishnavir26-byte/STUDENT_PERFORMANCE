# 🎓 Student Performance Analytics & Early Academic Risk System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![R Language](https://img.shields.io/badge/R-4.0%2B-276DC3.svg)](https://www.r-project.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%2B%20Statsmodels-orange.svg)](https://scikit-learn.org/)
[![Tailwind CSS](https://img.shields.io/badge/TailwindCSS-3.4-38bdf8.svg)](https://tailwindcss.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRD Compliance](https://img.shields.io/badge/PRD%20v1.0-100%25%20Verified-success.svg)](STUDENT_PERFORMANCE_REPORT.md)

An empirical, reproducible statistical analysis and machine learning early-warning risk classification framework for secondary mathematics students based on the benchmark dataset from the **UCI Machine Learning Repository** (*Paulo Cortez & Alice Silva, 2008*).

Combines exploratory data analysis, parametric & non-parametric hypothesis testing, multiple linear regression, holdout-validated logistic risk classification, demographic subgroup fairness audits, and a client-side interactive web dashboard.

---

## 🚀 Live Demo & Interactive Dashboard

- **Live Web Dashboard:** [https://vaishnavir26-byte.github.io/STUDENT_PERFORMANCE/](https://vaishnavir26-byte.github.io/STUDENT_PERFORMANCE/)
- **Alternative Cloud Deploy (Netlify):** [https://supplyxi.netlify.app/student_dashboard.html](https://supplyxi.netlify.app/student_dashboard.html)
- **Local Dashboard:** Open `index.html` or `student_dashboard.html` in any modern web browser.

---

## 📊 Summary of Research Findings (PRD v1.0)

1. **Academic Persistence as the Dominant Predictor (FR-05)**:
   - Second-period mark ($G2$) exhibits strong autocorrelation with final grade ($G3$): Pearson $r = 0.9049$ ($p < 0.001$, $95\%$ CI $[0.885, 0.921]$) and Spearman rank correlation $\rho = 0.9571$.
   - $G2$ alone accounts for $81.88\%$ of final grade variance.
2. **Weekly Study Time Shows Marginal Explanatory Power (FR-05)**:
   - Univariate one-way ANOVA across four study cohorts shows no statistically significant difference in final grades ($F(3, 391) = 1.7278, p = 0.1607, \eta^2 = 0.0131$). Non-parametric Kruskal-Wallis test: $H(3) = 7.5789, p = 0.0556$.
   - Controlling for prior marks in multiple regression, study duration does not independently elevate final marks ($\beta = -0.2015, p = 0.1304$).
3. **Institutional Support Association & Non-Causal Framing (FR-05, FR-09)**:
   - Extra educational support (`schoolsup`) is positively associated with risk ($45.1\%$ vs. $31.1\%$; $\text{Odds Ratio} = 1.8194$, Fisher's exact $p = 0.0555$).
   - **Crucial Non-Causal Distinction**: This reflects compensatory remedial assignment (schools assign tutoring to students already struggling) rather than support causing lower grades.
4. **Generalizable Held-Out Predictive Performance (FR-06, FR-07, FR-08)**:
   - **Multiple Linear Regression ($G3$)**: Evaluated on an isolated $20\%$ holdout test partition ($N = 79$, stratified by risk, `seed = 42`):
     - **Test RMSE = 1.9201** | **Test MAE = 1.2729** | **$R^2 = 0.8258$** (Low collinearity: $\text{max VIF} = 3.76$).
   - **Logistic Risk Classification ($G3 < 10$)**: Evaluated on unseen test data at standard cutoff ($0.50$):
     - **Test Accuracy = 91.14%** | **Sensitivity (At-Risk Recall) = 88.46%** | **Specificity = 92.45%** | **Precision = 85.19%** | **ROC-AUC = 0.9797** | **Brier Score = 0.0577**.
5. **The Two-Stage Hurdle Architecture (PRD Sec 10/11 Extension)**:
   - $38$ students ($9.62\%$) scored $G3 = 0$ due to examination withdrawal despite passing earlier terms.
   - Stage 1 (Dropout Classifier) achieves **$100\%$ Test AUC**; Stage 2 (Completer Grade Regressor) drops test error from $1.9201$ to **$0.7747$ points** ($59.7\%$ error reduction).

---

## 🛠️ Repository Architecture

```
├── index.html                      # Interactive 6-Tab Client-Side Dashboard (Root)
├── student_dashboard.html          # Standalone Dashboard Mirror
├── student-mat.csv                 # Verified UCI Mathematics dataset (N = 395, 33 columns)
├── student.txt                     # Dataset metadata and feature descriptions
├── student_performance_analysis.py # 10-Stage reproducible Python scientific pipeline
├── student_performance_analysis.R  # Reproducible 9-stage R analytical script
├── train_models.py                 # Multi-model training, 5-fold CV, and serialization
├── predict.py                      # Interactive single-student, early-warning, and batch CLI
├── test_pipeline.py                # 11-stage automated system & regression test suite (100% pass)
├── test_models.py                  # Model weights and boundary test suite
├── analysis_summary.json           # Machine-readable statistical test outputs
├── predictions.csv                 # Batch scored cohort dataset with risk tiers
├── STUDENT_PERFORMANCE_REPORT.md   # Comprehensive academic research report
├── analysis_report.Rmd             # Knit-ready R Markdown report
├── figures/                        # 300 DPI high-resolution figures
│   ├── fig1_g3_histogram.png       # Grade distribution & at-risk cutoff
│   ├── fig2_studytime_boxplot.png  # Study time ANOVA boxplot with mean diamonds
│   ├── fig3_g2_vs_g3_scatter.png   # G2 vs G3 scatterplot with OLS trend
│   ├── fig4_lm_diagnostics.png     # Residual diagnostics 4-panel panel
│   └── fig5_logistic_evaluation.png# Held-out ROC & reliability diagram
└── models/                         # Serialized models & partition CSVs
    ├── linear_regression_g3.joblib # FR-06 Multiple linear regressor
    ├── logistic_regression_at_risk.joblib # FR-07 Logistic classifier
    ├── early_warning_risk.joblib   # Section 10/11 Pre-course risk model
    ├── rf_regressor_g3.joblib      # Random Forest regressor benchmark
    ├── rf_classifier_risk.joblib   # Random Forest classifier benchmark
    ├── train_partition.csv         # Stratified 80% training set (N = 316)
    ├── test_partition.csv          # Stratified 20% holdout test set (N = 79)
    └── model_metadata.json         # Exact formulas, coefficients, and cutoff thresholds
```

---

## ⚡ Quickstart & Replication

### 1. Run Complete Scientific Pipeline
```bash
python3 student_performance_analysis.py
```
*(Executes all 10 stages: hygiene checks, descriptive stats, figures, hypothesis testing, train/test split, multiple regression, logistic risk classification, subgroup fairness, hurdle model, and exports `analysis_summary.json`)*

### 2. Run Automated Verification Test Suite
```bash
python3 test_pipeline.py
```
*(Runs 11 automated verification tests covering data integrity, hypotheses, model metrics, and artifacts)*

### 3. Interactive Student Prediction (CLI)
```bash
# Mid-Course Prediction:
python3 predict.py --g1 11 --g2 10 --studytime 2 --absences 4 --failures 0 --schoolsup no

# High-Risk Profile:
python3 predict.py --g1 6 --g2 5 --studytime 1 --absences 12 --failures 2 --schoolsup yes

# Pre-Course Early Warning (No G1/G2):
python3 predict.py --early --studytime 1 --absences 8 --failures 1 --schoolsup yes

# Batch Scoring on Dataset:
python3 predict.py --input student-mat.csv --output predictions.csv
```

### 4. R Analysis Pipeline
```bash
Rscript student_performance_analysis.R
```

---

## 🛡️ Responsible AI & Ethical Guidelines (FR-09)

1. **Decision Support, Never Automated Tracking**: Risk probabilities serve exclusively to notify teachers and counselors to provide **supportive interventions** (e.g., peer tutoring, personalized problem sets, counseling).
2. **Never for High-Stakes Gatekeeping**: Model outputs must never be used to gate students from courses, track students into lower educational tiers, or recommend punitive action.
3. **Non-Causal Interpretation**: Observational correlations (such as `schoolsup` with risk) must not be interpreted as causal harms. Tutoring is assigned to students who are already struggling (compensatory targeting).
4. **Human Discretion Required**: Educators must always review qualitative context before initiating interventions.

---

## 📜 License
MIT License. Dataset courtesy of the UCI Machine Learning Repository (Cortez & Silva, 2008).
