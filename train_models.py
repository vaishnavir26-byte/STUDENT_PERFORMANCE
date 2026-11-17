#!/usr/bin/env python3
"""
Student Performance Analytics: Model Training & Serialization Pipeline
========================================================================
Reference: PRD v1.0 (FR-06, FR-07, FR-08, Section 10 & 11)
Dataset: UCI Student Performance (student-mat.csv)

Models Trained:
1. Multiple Linear Regression (FR-06): Estimates G3 final grade using [G1, G2, studytime, absences, failures].
2. Binary Logistic Regression (FR-07): Estimates at-risk probability P(G3 < 10) using [G1, G2, studytime, absences, failures, schoolsup].
3. Early-Warning Model (PRD Section 10/11 Extension): Predicts at-risk status WITHOUT G1 and G2 for beginning-of-term intervention.
4. Random Forest Regressor & Classifier: Non-linear benchmark comparisons.

Zero Leakage Guarantee (FR-08):
- 80/20 Stratified Train/Test Split (fixed seed = 42).
- Preprocessing, threshold tuning, and scaling fitted strictly on training data.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    recall_score,
    precision_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    f1_score,
    brier_score_loss
)
import statsmodels.api as sm
import statsmodels.formula.api as smf

def load_and_preprocess(filepath="student-mat.csv"):
    """Load dataset, validate hygiene, and create project indicators."""
    print("=" * 70)
    print(" STAGE 1: Dataset Import & Hygiene Inspection (FR-01, FR-02)")
    print("=" * 70)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset '{filepath}' not found.")
        
    df = pd.read_csv(filepath, sep=';')
    print(f" Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f" Missing values: {df.isna().sum().sum()} | Duplicates: {df.duplicated().sum()}")
    
    # Target definition (PRD Section 5)
    df['at_risk'] = (df['G3'] < 10).astype(int)
    print(f" At-Risk rate (G3 < 10): {df['at_risk'].mean():.2%} ({df['at_risk'].sum()} students)")
    
    return df

def split_data(df, seed=42):
    """Stratified train/test split to guarantee zero data leakage (FR-08)."""
    print("\n" + "=" * 70)
    print(" STAGE 2: Stratified Holdout Partitioning (FR-08)")
    print("=" * 70)
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=seed,
        stratify=df['at_risk']
    )
    
    print(f" Split Configuration: 80% Training ({len(train_df)} rows), 20% Testing ({len(test_df)} rows)")
    print(f" Random Seed: {seed}")
    print(f" Training at-risk prevalence: {train_df['at_risk'].mean():.2%}")
    print(f" Testing at-risk prevalence : {test_df['at_risk'].mean():.2%}")
    
    return train_df.copy(), test_df.copy()

def train_linear_model(train_df, test_df):
    """Train Multiple Linear Regression for final grade G3 (FR-06)."""
    print("\n" + "=" * 70)
    print(" STAGE 3: Multiple Linear Regression for Final Grade G3 (FR-06)")
    print("=" * 70)
    
    features = ['G1', 'G2', 'studytime', 'absences', 'failures']
    X_train = train_df[features]
    y_train = train_df['G3']
    X_test = test_df[features]
    y_test = test_df['G3']
    
    # 1. Statsmodels fit for inference statistics & standard errors
    sm_model = smf.ols('G3 ~ G1 + G2 + studytime + absences + failures', data=train_df).fit()
    
    # 2. Scikit-learn fit for serialized inference
    sk_model = LinearRegression()
    sk_model.fit(X_train, y_train)
    
    # Predictions
    train_preds = sk_model.predict(X_train)
    test_preds = sk_model.predict(X_test)
    
    # Metrics
    train_rmse = float(np.sqrt(mean_squared_error(y_train, train_preds)))
    train_mae = float(mean_absolute_error(y_train, train_preds))
    train_r2 = float(r2_score(y_train, train_preds))
    
    test_rmse = float(np.sqrt(mean_squared_error(y_test, test_preds)))
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_r2 = float(r2_score(y_test, test_preds))
    
    # 5-Fold Cross-Validation on Training Set
    cv_r2 = cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(LinearRegression(), X_train, y_train, cv=5, scoring='neg_mean_squared_error'))
    
    print("\n--- Model Coefficients & Statistical Significance ---")
    print(f"  Intercept : {sk_model.intercept_:8.4f} (p = {sm_model.pvalues['Intercept']:.4f})")
    for feat, coef in zip(features, sk_model.coef_):
        pval = sm_model.pvalues[feat]
        sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
        print(f"  {feat:10}: {coef:8.4f} (p = {pval:.4f}) [{sig}]")
        
    print("\n--- Generalization Evaluation ---")
    print(f"  Training  -> RMSE: {train_rmse:.4f} | MAE: {train_mae:.4f} | R²: {train_r2:.4f}")
    print(f"  5-Fold CV -> Mean RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f}) | Mean R²: {cv_r2.mean():.4f}")
    print(f"  Held-Out  -> RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")
    
    metrics = {
        "features": features,
        "intercept": float(sk_model.intercept_),
        "coefficients": {f: float(c) for f, c in zip(features, sk_model.coef_)},
        "p_values": {k: float(v) for k, v in sm_model.pvalues.items()},
        "training": {"rmse": train_rmse, "mae": train_mae, "r2": train_r2},
        "cross_val_5fold": {"rmse_mean": float(cv_rmse.mean()), "rmse_std": float(cv_rmse.std()), "r2_mean": float(cv_r2.mean())},
        "held_out_test": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2}
    }
    
    return sk_model, metrics

def train_logistic_model(train_df, test_df):
    """Train Binary Logistic Regression for At-Risk status (FR-07)."""
    print("\n" + "=" * 70)
    print(" STAGE 4: Logistic Regression for Academic Risk (FR-07)")
    print("=" * 70)
    
    features = ['G1', 'G2', 'studytime', 'absences', 'failures', 'schoolsup']
    
    def prep_X(df_in):
        X = df_in[['G1', 'G2', 'studytime', 'absences', 'failures']].copy()
        X['schoolsup_yes'] = (df_in['schoolsup'] == 'yes').astype(int)
        return X
        
    X_train = prep_X(train_df)
    y_train = train_df['at_risk']
    X_test = prep_X(test_df)
    y_test = test_df['at_risk']
    
    # 1. Statsmodels fit for unpenalized maximum likelihood inference
    sm_logit = smf.logit(
        'at_risk ~ G1 + G2 + studytime + absences + failures + C(schoolsup, Treatment(reference="no"))',
        data=train_df
    ).fit(disp=False)
    
    # 2. Scikit-learn fit (C=1e5 for unpenalized MLE equivalence)
    sk_logit = LogisticRegression(C=1e5, solver='lbfgs', max_iter=1000, random_state=42)
    sk_logit.fit(X_train, y_train)
    
    # Probabilities
    train_probs = sk_logit.predict_proba(X_train)[:, 1]
    test_probs = sk_logit.predict_proba(X_test)[:, 1]
    
    # Youden's J optimal threshold calculation on TRAINING set ONLY
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(y_train, train_probs)
    j_scores = tpr_tr - fpr_tr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds_tr[optimal_idx])
    
    print(f" Youden's J Optimal Cutoff (computed on training set): {optimal_threshold:.4f}")
    
    def evaluate_at_threshold(y_true, probs, cutoff, label="Threshold"):
        preds = (probs >= cutoff).astype(int)
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = cm.ravel()
        
        acc = float(accuracy_score(y_true, preds))
        sens = float(recall_score(y_true, preds)) # recall for at-risk
        spec = float(tn / (tn + fp))
        prec = float(precision_score(y_true, preds, zero_division=0))
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1 = float(f1_score(y_true, preds, zero_division=0))
        auc = float(roc_auc_score(y_true, probs))
        brier = float(brier_score_loss(y_true, probs))
        
        print(f"\n--- Held-Out Test Evaluation at {label} ({cutoff:.4f}) ---")
        print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"  Accuracy    : {acc:.4f} ({acc:.2%})")
        print(f"  Sensitivity : {sens:.4f} (Recall for At-Risk)")
        print(f"  Specificity : {spec:.4f} (True Negative Rate)")
        print(f"  Precision   : {prec:.4f} (PPV)")
        print(f"  NPV         : {npv:.4f}")
        print(f"  F1-Score    : {f1:.4f}")
        print(f"  ROC-AUC     : {auc:.4f}")
        
        return {
            "threshold": cutoff,
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "accuracy": acc,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "npv": npv,
            "f1_score": f1,
            "roc_auc": auc,
            "brier_score": brier
        }
        
    eval_standard = evaluate_at_threshold(y_test, test_probs, 0.50, "Standard Cutoff")
    eval_optimal = evaluate_at_threshold(y_test, test_probs, optimal_threshold, "Youden-Optimal Cutoff")
    
    # Odds ratios
    or_values = {f: float(np.exp(c)) for f, c in zip(X_train.columns, sk_logit.coef_[0])}
    
    metrics = {
        "features": list(X_train.columns),
        "intercept": float(sk_logit.intercept_[0]),
        "coefficients": {f: float(c) for f, c in zip(X_train.columns, sk_logit.coef_[0])},
        "odds_ratios": or_values,
        "optimal_threshold_youden": optimal_threshold,
        "held_out_standard_050": eval_standard,
        "held_out_optimal_youden": eval_optimal
    }
    
    return sk_logit, metrics

def train_early_warning_model(train_df, test_df):
    """
    Train Early-Warning Model (PRD Section 10 & 11 Extension).
    Predicts academic risk WITHOUT G1 and G2 for beginning-of-semester intervention.
    """
    print("\n" + "=" * 70)
    print(" STAGE 5: Early-Warning Pre-Course Risk Model (Without G1/G2)")
    print("=" * 70)
    
    early_features = ['studytime', 'failures', 'absences', 'Medu', 'Fedu', 'goout', 'Dalc', 'Walc', 'age']
    
    def prep_early_X(df_in):
        X = df_in[early_features].copy()
        X['schoolsup_yes'] = (df_in['schoolsup'] == 'yes').astype(int)
        X['famsup_yes'] = (df_in['famsup'] == 'yes').astype(int)
        X['higher_yes'] = (df_in['higher'] == 'yes').astype(int)
        return X
        
    X_train = prep_early_X(train_df)
    y_train = train_df['at_risk']
    X_test = prep_early_X(test_df)
    y_test = test_df['at_risk']
    
    early_logit = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    early_logit.fit(X_train, y_train)
    
    test_probs = early_logit.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= 0.50).astype(int)
    
    auc = float(roc_auc_score(y_test, test_probs))
    acc = float(accuracy_score(y_test, test_preds))
    sens = float(recall_score(y_test, test_preds))
    spec = float(confusion_matrix(y_test, test_preds)[0,0] / (confusion_matrix(y_test, test_preds)[0,0] + confusion_matrix(y_test, test_preds)[0,1]))
    
    print(" Early-warning model trained (Pre-Course Screening):")
    print(f"  Features used: {list(X_train.columns)}")
    print(f"  Test ROC-AUC: {auc:.4f} | Accuracy: {acc:.2%} | Sensitivity: {sens:.2%} | Specificity: {spec:.2%}")
    print("  Note: As expected, early pre-course prediction yields lower precision than mid-term G1/G2 models,")
    print("  providing valuable coarse screening for proactive orientation.")
    
    metrics = {
        "features": list(X_train.columns),
        "test_roc_auc": auc,
        "test_accuracy": acc,
        "test_sensitivity": sens,
        "test_specificity": spec
    }
    
    return early_logit, metrics

def train_random_forest_benchmark(train_df, test_df):
    """Train Random Forest models to assess non-linear relationships."""
    print("\n" + "=" * 70)
    print(" STAGE 6: Non-Linear Random Forest Benchmark")
    print("=" * 70)
    
    features = ['G1', 'G2', 'studytime', 'absences', 'failures']
    X_train = train_df[features]
    X_test = test_df[features]
    
    # Regression
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, train_df['G3'])
    rf_reg_preds = rf_reg.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(test_df['G3'], rf_reg_preds))
    rf_r2 = r2_score(test_df['G3'], rf_reg_preds)
    
    # Classification
    X_train_c = X_train.copy()
    X_train_c['schoolsup_yes'] = (train_df['schoolsup'] == 'yes').astype(int)
    X_test_c = X_test.copy()
    X_test_c['schoolsup_yes'] = (test_df['schoolsup'] == 'yes').astype(int)
    
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf.fit(X_train_c, train_df['at_risk'])
    rf_probs = rf_clf.predict_proba(X_test_c)[:, 1]
    rf_auc = roc_auc_score(test_df['at_risk'], rf_probs)
    
    print(f"  Random Forest Regressor -> Test RMSE: {rf_rmse:.4f} | Test R²: {rf_r2:.4f}")
    print(f"  Random Forest Classifier -> Test ROC-AUC: {rf_auc:.4f}")
    
    importances = {f: float(imp) for f, imp in zip(X_test_c.columns, rf_clf.feature_importances_)}
    print(f"  Classifier Feature Importances: {importances}")
    
    return rf_reg, rf_clf

def main():
    os.makedirs("models", exist_ok=True)
    
    # 1. Load data
    df = load_and_preprocess("student-mat.csv")
    
    # 2. Stratified train/test split (FR-08)
    train_df, test_df = split_data(df, seed=42)
    
    # 3. Train models
    lm_model, lm_metrics = train_linear_model(train_df, test_df)
    logit_model, logit_metrics = train_logistic_model(train_df, test_df)
    early_model, early_metrics = train_early_warning_model(train_df, test_df)
    rf_reg, rf_clf = train_random_forest_benchmark(train_df, test_df)
    
    # 4. Save trained models
    print("\n" + "=" * 70)
    print(" STAGE 7: Serializing Model Artifacts into 'models/'")
    print("=" * 70)
    
    joblib.dump(lm_model, "models/linear_regression_g3.joblib")
    joblib.dump(logit_model, "models/logistic_regression_at_risk.joblib")
    joblib.dump(early_model, "models/early_warning_risk.joblib")
    joblib.dump(rf_reg, "models/rf_regressor_g3.joblib")
    joblib.dump(rf_clf, "models/rf_classifier_risk.joblib")
    
    # Save training and test splits for reproducibility
    train_df.to_csv("models/train_partition.csv", index=False)
    test_df.to_csv("models/test_partition.csv", index=False)
    
    # Combine metadata
    metadata = {
        "dataset": "UCI Student Performance (Mathematics, student-mat.csv)",
        "split_seed": 42,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "linear_regression_g3": lm_metrics,
        "logistic_regression_at_risk": logit_metrics,
        "early_warning_model": early_metrics
    }
    
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(" Saved model artifacts:")
    print("  - models/linear_regression_g3.joblib")
    print("  - models/logistic_regression_at_risk.joblib")
    print("  - models/early_warning_risk.joblib")
    print("  - models/rf_regressor_g3.joblib")
    print("  - models/rf_classifier_risk.joblib")
    print("  - models/model_metadata.json")
    print("  - models/train_partition.csv & test_partition.csv")
    print("\n Model training pipeline completed successfully.")

if __name__ == "__main__":
    main()
