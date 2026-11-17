#!/usr/bin/env python3
"""
Student Performance Analytics: Model Inference & Prediction CLI
================================================================
Reference: PRD v1.0 (FR-06, FR-07, FR-09)

Use cases:
1. Single-student interactive prediction:
   python3 predict.py --g1 12 --g2 11 --studytime 2 --absences 4 --failures 0 --schoolsup no

2. Single-student early-warning prediction (without G1/G2):
   python3 predict.py --early --studytime 2 --absences 6 --failures 1 --schoolsup yes --medu 2 --fedu 2 --goout 3

3. Batch prediction on a dataset:
   python3 predict.py --input student-mat.csv --output predictions.csv
"""

import sys
import os
import argparse
import json
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

def load_models():
    """Load serialized models and metadata."""
    lm_path = os.path.join(MODEL_DIR, "linear_regression_g3.joblib")
    logit_path = os.path.join(MODEL_DIR, "logistic_regression_at_risk.joblib")
    early_path = os.path.join(MODEL_DIR, "early_warning_risk.joblib")
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    
    if not (os.path.exists(lm_path) and os.path.exists(logit_path)):
        print("Model artifacts not found in 'models/'. Running training pipeline first...")
        import train_models
        train_models.main()
        
    lm = joblib.load(lm_path)
    logit = joblib.load(logit_path)
    early = joblib.load(early_path) if os.path.exists(early_path) else None
    
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        
    return lm, logit, early, metadata

def predict_single_student(g1, g2, studytime, absences, failures, schoolsup, lm, logit, metadata):
    """Generate G3 grade estimate and risk probability for mid-course student."""
    # Linear Regression Feature Vector
    X_reg = pd.DataFrame([{
        'G1': g1,
        'G2': g2,
        'studytime': studytime,
        'absences': absences,
        'failures': failures
    }])
    
    pred_g3 = float(lm.predict(X_reg)[0])
    pred_g3_clamped = max(0.0, min(20.0, pred_g3))
    
    # 95% approximate prediction interval (RMSE ~ 1.92)
    test_rmse = metadata["linear_regression_g3"]["held_out_test"]["rmse"]
    ci_lower = max(0.0, pred_g3 - 1.96 * test_rmse)
    ci_upper = min(20.0, pred_g3 + 1.96 * test_rmse)
    
    # Logistic Regression Feature Vector
    schoolsup_num = 1 if str(schoolsup).lower() in ['yes', '1', 'true', 'y'] else 0
    X_clf = pd.DataFrame([{
        'G1': g1,
        'G2': g2,
        'studytime': studytime,
        'absences': absences,
        'failures': failures,
        'schoolsup_yes': schoolsup_num
    }])
    
    risk_prob = float(logit.predict_proba(X_clf)[0, 1])
    opt_cutoff = metadata["logistic_regression_at_risk"]["optimal_threshold_youden"]
    
    # Risk categorization
    if risk_prob >= 0.50:
        tier = "HIGH RISK"
        badge = "🔴"
        action = "Immediate academic counselor outreach, peer mentoring, and remedial mathematics review."
    elif risk_prob >= opt_cutoff:
        tier = "MODERATE RISK"
        badge = "🟡"
        action = "Early supportive check-in, targeted practice problem sets, and study habits assessment."
    else:
        tier = "LOW RISK"
        badge = "🟢"
        action = "Standard instructional pacing. Continue positive reinforcement."
        
    print("\n" + "=" * 65)
    print(" 🎓 STUDENT PERFORMANCE ANALYTICS: PREDICTION RESULT")
    print("=" * 65)
    print(" Student Input Profile:")
    print(f"  • Period 1 Grade (G1)  : {g1:.1f} / 20")
    print(f"  • Period 2 Grade (G2)  : {g2:.1f} / 20")
    print(f"  • Study Time Category  : {studytime} (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    print(f"  • Past Class Failures  : {failures}")
    print(f"  • Recorded Absences    : {absences}")
    print(f"  • Extra School Support : {'Yes' if schoolsup_num == 1 else 'No'}")
    print("-" * 65)
    print(" 📈 Predicted Final Mathematics Grade (G3):")
    print(f"  • Estimated Score      : {pred_g3_clamped:.2f} / 20  (Raw OLS: {pred_g3:.2f})")
    print(f"  • 95% Prediction Interval: [{ci_lower:.2f}, {ci_upper:.2f}]")
    print("-" * 65)
    print(" ⚠️  Academic Risk Assessment (Target: G3 < 10):")
    print(f"  • Risk Probability     : {risk_prob:.2%}  (Logistic Model)")
    print(f"  • Classification Status: {badge} {tier}")
    print(f"  • Decision Cutoffs     : Standard (0.50) | Youden-Optimal Safety Net ({opt_cutoff:.2f})")
    print("-" * 65)
    print(" 💡 Recommended Supportive Action (Non-Punitive):")
    print(f"  {action}")
    print("=" * 65)
    print(" Note: Predictions are decision-support tools for supportive intervention")
    print(" and must never be used for punitive tracking or automated grading.")
    print("=" * 65 + "\n")
    
    return {
        "pred_g3": pred_g3_clamped,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "risk_prob": risk_prob,
        "risk_tier": tier
    }

def predict_early_student(studytime, absences, failures, schoolsup, medu, fedu, goout, early_model):
    """Generate risk estimate for beginning-of-term student without G1/G2."""
    schoolsup_num = 1 if str(schoolsup).lower() in ['yes', '1', 'true', 'y'] else 0
    X_early = pd.DataFrame([{
        'studytime': studytime,
        'failures': failures,
        'absences': absences,
        'Medu': medu,
        'Fedu': fedu,
        'goout': goout,
        'Dalc': 1,
        'Walc': 1,
        'age': 16,
        'schoolsup_yes': schoolsup_num,
        'famsup_yes': 1,
        'higher_yes': 1
    }])
    
    prob = float(early_model.predict_proba(X_early)[0, 1])
    tier = "ELEVATED PRE-COURSE RISK" if prob >= 0.50 else "STANDARD PRE-COURSE RISK"
    badge = "🟡" if prob >= 0.50 else "🟢"
    
    print("\n" + "=" * 65)
    print(" 🕒 EARLY-WARNING PRE-COURSE RISK ESTIMATION (NO G1/G2)")
    print("=" * 65)
    print(f"  • Risk Probability: {prob:.2%}")
    print(f"  • Classification  : {badge} {tier}")
    print("  • Note: Coarse pre-term indicator for early tutorial allocation.")
    print("=" * 65 + "\n")
    return {"early_risk_prob": prob, "tier": tier}

def batch_predict(input_file, output_file, lm, logit, metadata):
    """Run batch prediction on an entire CSV file."""
    print(f"\nProcessing batch prediction from: {input_file}")
    df = pd.read_csv(input_file, sep=';' if ';' in open(input_file).readline() else ',')
    
    required_cols = ['G1', 'G2', 'studytime', 'absences', 'failures']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input file missing required column: '{col}'")
            
    # Regression
    X_reg = df[['G1', 'G2', 'studytime', 'absences', 'failures']].copy()
    pred_g3 = lm.predict(X_reg)
    df['predicted_G3'] = np.clip(np.round(pred_g3, 2), 0.0, 20.0)
    
    # Classification
    X_clf = X_reg.copy()
    if 'schoolsup' in df.columns:
        X_clf['schoolsup_yes'] = (df['schoolsup'].astype(str).str.lower() == 'yes').astype(int)
    else:
        X_clf['schoolsup_yes'] = 0
        
    risk_probs = logit.predict_proba(X_clf)[:, 1]
    df['risk_probability'] = np.round(risk_probs, 4)
    df['at_risk_prediction'] = (risk_probs >= 0.50).astype(int)
    
    opt_cutoff = metadata["logistic_regression_at_risk"]["optimal_threshold_youden"]
    conditions = [
        (df['risk_probability'] >= 0.50),
        (df['risk_probability'] >= opt_cutoff),
        (df['risk_probability'] < opt_cutoff)
    ]
    tiers = ['HIGH RISK', 'MODERATE RISK', 'LOW RISK']
    df['risk_tier'] = np.select(conditions, tiers, default='UNKNOWN')
    
    df.to_csv(output_file, index=False)
    print(f" Successfully processed {len(df)} student records.")
    print(f" Predictions written to: {output_file}")
    print(" Summary of Risk Distribution:")
    print(df['risk_tier'].value_counts())

def main():
    parser = argparse.ArgumentParser(description="Student Performance Analytics - Prediction CLI")
    parser.add_argument("--g1", type=float, help="Period 1 grade (0-20)")
    parser.add_argument("--g2", type=float, help="Period 2 grade (0-20)")
    parser.add_argument("--studytime", type=int, default=2, help="Study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    parser.add_argument("--absences", type=int, default=4, help="Number of absences (0-93)")
    parser.add_argument("--failures", type=int, default=0, help="Number of past class failures (0-4)")
    parser.add_argument("--schoolsup", type=str, default="no", help="Extra school educational support (yes/no)")
    parser.add_argument("--medu", type=int, default=2, help="Mother's education (0-4)")
    parser.add_argument("--fedu", type=int, default=2, help="Father's education (0-4)")
    parser.add_argument("--goout", type=int, default=3, help="Going out with friends (1-5)")
    parser.add_argument("--early", action="store_true", help="Run early warning model without G1/G2")
    parser.add_argument("--input", type=str, help="Input CSV path for batch inference")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV path for batch inference")
    
    args = parser.parse_args()
    
    lm, logit, early, metadata = load_models()
    
    if args.input:
        batch_predict(args.input, args.output, lm, logit, metadata)
    elif args.early:
        predict_early_student(
            studytime=args.studytime,
            absences=args.absences,
            failures=args.failures,
            schoolsup=args.schoolsup,
            medu=args.medu,
            fedu=args.fedu,
            goout=args.goout,
            early_model=early
        )
    elif args.g1 is not None and args.g2 is not None:
        predict_single_student(
            g1=args.g1,
            g2=args.g2,
            studytime=args.studytime,
            absences=args.absences,
            failures=args.failures,
            schoolsup=args.schoolsup,
            lm=lm,
            logit=logit,
            metadata=metadata
        )
    else:
        # Default demo student run
        print("\nNo inputs provided. Running prediction demo on a representative sample student:")
        predict_single_student(
            g1=11.0,
            g2=10.0,
            studytime=2,
            absences=4,
            failures=0,
            schoolsup="no",
            lm=lm,
            logit=logit,
            metadata=metadata
        )

if __name__ == "__main__":
    main()
