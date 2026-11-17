#!/usr/bin/env python3
"""
EduPrescribe AI: Student Behavioral Wellness & Counterfactual Study Optimizer
=============================================================================
A novel prescriptive analytics framework combining:
1. Burnout Vulnerability Index (BVI) calculation from behavioral and lifestyle stressors.
2. 4-Quadrant Student Well-Being Segmentation (Thriving, Burnout-Risk, Coasting, Crisis).
3. Counterfactual "What-If" Uplift Engine finding the minimum actionable intervention
   vector that transitions a student from At-Risk to Passing while mitigating burnout.
4. Serialized multi-objective models and cohort-wide prescription generator.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, classification_report

os.makedirs("eduprescribe_models", exist_ok=True)
os.makedirs("eduprescribe_figures", exist_ok=True)

def compute_burnout_index(df):
    """
    Computes the composite Burnout Vulnerability Index (BVI in [0, 100])
    from 5 multidimensional behavioral dimensions:
    - S1: Substance Coping (Dalc, Walc)
    - S2: Social Depletion / Rest Imbalance (goout vs freetime)
    - S3: Physical & Vitality Deficit (health)
    - S4: Absenteeism Strain (absences)
    - S5: Relational Friction (famrel)
    """
    df_out = df.copy()
    
    # 1. Substance coping: Workday alcohol (Dalc) is weighted 2x vs weekend (Walc)
    s1 = (2.0 * df_out['Dalc'] + df_out['Walc'] - 3.0) / 12.0
    
    # 2. Social depletion: Going out exceeding free time indicates sleep/recovery sacrifice
    s2 = np.clip((df_out['goout'] - df_out['freetime'] + 4.0) / 8.0, 0.0, 1.0)
    
    # 3. Vitality deficit: health from 1 (poor) to 5 (excellent)
    s3 = (5.0 - df_out['health']) / 4.0
    
    # 4. Absenteeism strain: chronic absence saturation
    s4 = np.clip(df_out['absences'] / 20.0, 0.0, 1.0)
    
    # 5. Relational friction: family relationship from 1 (bad) to 5 (excellent)
    s5 = (5.0 - df_out['famrel']) / 4.0
    
    # Weighted composite index (0 to 100 scale)
    bvi = 100.0 * (0.25 * s1 + 0.20 * s2 + 0.25 * s3 + 0.15 * s4 + 0.15 * s5)
    df_out['BVI'] = np.round(bvi, 2)
    
    # Burnout Risk Tier
    conditions = [
        (df_out['BVI'] < 25.0),
        (df_out['BVI'] >= 25.0) & (df_out['BVI'] < 40.0),
        (df_out['BVI'] >= 40.0) & (df_out['BVI'] < 55.0),
        (df_out['BVI'] >= 55.0)
    ]
    tiers = ['Low Burnout', 'Moderate Stress', 'Elevated Burnout Risk', 'Acute Exhaustion']
    df_out['burnout_tier'] = np.select(conditions, tiers, default='Moderate Stress')
    
    return df_out

def assign_quadrants(df):
    """
    Assigns each student to one of 4 Student Well-Being Quadrants:
    - Q1: Thriving Achievers (G3 >= 10 & BVI < 40)
    - Q2: High-Performing Burnout Risk (G3 >= 10 & BVI >= 40)
    - Q3: Dual Crisis / High Vulnerability (G3 < 10 & BVI >= 40)
    - Q4: Academic Disengagement / Coasting (G3 < 10 & BVI < 40)
    """
    df_out = df.copy()
    
    conditions = [
        (df_out['G3'] >= 10) & (df_out['BVI'] < 40.0),
        (df_out['G3'] >= 10) & (df_out['BVI'] >= 40.0),
        (df_out['G3'] < 10) & (df_out['BVI'] >= 40.0),
        (df_out['G3'] < 10) & (df_out['BVI'] < 40.0)
    ]
    quadrants = [
        'Q1: Thriving Achievers',
        'Q2: High-Performing Burnout Risk',
        'Q3: Dual Crisis / High Vulnerability',
        'Q4: Academic Disengagement / Coasting'
    ]
    df_out['quadrant'] = np.select(conditions, quadrants, default='Q1: Thriving Achievers')
    
    return df_out

def train_prescriptive_models(df):
    """
    Trains dual predictive models:
    1. Grade Estimator (Linear OLS with behavioral terms)
    2. Passing Probability Classifier (Logistic Regression)
    3. Burnout Regressor (Random Forest)
    """
    print("\n--- Training EduPrescribe Machine Learning Models ---")
    
    # Features
    features_academic = ['G1', 'G2', 'studytime', 'absences', 'failures']
    X_acad = df[features_academic].copy()
    X_acad['schoolsup_yes'] = (df['schoolsup'] == 'yes').astype(int)
    
    # 1. Grade Regressor
    grade_model = LinearRegression()
    grade_model.fit(X_acad, df['G3'])
    
    # 2. Risk/Passing Classifier (Target: 1 = Passing G3 >= 10)
    y_pass = (df['G3'] >= 10).astype(int)
    pass_clf = LogisticRegression(max_iter=1000, random_state=42)
    pass_clf.fit(X_acad, y_pass)
    
    # 3. Burnout Regressor (Lifestyle features -> BVI)
    lifestyle_features = ['Dalc', 'Walc', 'goout', 'freetime', 'health', 'absences', 'famrel']
    X_life = df[lifestyle_features].copy()
    burnout_rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)
    burnout_rf.fit(X_life, df['BVI'])
    
    # Metrics
    train_pred_g3 = grade_model.predict(X_acad)
    rmse = np.sqrt(mean_squared_error(df['G3'], train_pred_g3))
    pass_probs = pass_clf.predict_proba(X_acad)[:, 1]
    auc = roc_auc_score(y_pass, pass_probs)
    bvi_preds = burnout_rf.predict(X_life)
    bvi_r2 = r2_score(df['BVI'], bvi_preds)
    
    print(f" Academic Grade Model RMSE : {rmse:.4f} points")
    print(f" Passing Classifier ROC-AUC: {auc:.4f}")
    print(f" Burnout Regressor R² Score: {bvi_r2:.4f}")
    
    # Serialize models
    joblib.dump(grade_model, "eduprescribe_models/grade_regressor.joblib")
    joblib.dump(pass_clf, "eduprescribe_models/pass_classifier.joblib")
    joblib.dump(burnout_rf, "eduprescribe_models/burnout_regressor.joblib")
    
    return grade_model, pass_clf, burnout_rf

def optimize_counterfactual(row, grade_model, pass_clf, target_pass_prob=0.85, max_bvi=38.0):
    """
    Computes actionable counterfactual intervention options for any student.
    Actionable levers:
    - delta_study: +0, +1, +2
    - delta_absences: -0, -2, -4, -6
    - add_schoolsup: 0 or 1
    - delta_goout: -0, -1, -2
    """
    current_g1 = row['G1']
    current_g2 = row['G2']
    current_study = row['studytime']
    current_abs = row['absences']
    current_fail = row['failures']
    current_sup = 1 if str(row['schoolsup']).lower() == 'yes' else 0
    current_goout = row['goout']
    current_free = row['freetime']
    current_health = row['health']
    current_famrel = row['famrel']
    current_dalc = row['Dalc']
    current_walc = row['Walc']
    current_bvi = row['BVI']
    
    current_feat = pd.DataFrame([{
        'G1': current_g1, 'G2': current_g2,
        'studytime': current_study, 'absences': current_abs,
        'failures': current_fail, 'schoolsup_yes': current_sup
    }])
    baseline_pass_prob = float(pass_clf.predict_proba(current_feat)[0, 1])
    baseline_pred_g3 = float(grade_model.predict(current_feat)[0])
    
    # If student is already thriving (>90% pass prob & low BVI), no radical intervention needed
    if baseline_pass_prob >= 0.90 and current_bvi < 35.0:
        return {
            "status": "Already Thriving",
            "baseline_g3": baseline_pred_g3,
            "baseline_pass_prob": baseline_pass_prob,
            "baseline_bvi": current_bvi,
            "recommended_action": "Maintain current study rhythm; positive reinforcement.",
            "expected_uplift": 0.0,
            "new_bvi": current_bvi,
            "new_pass_prob": baseline_pass_prob,
            "pathway_type": "Maintenance"
        }
        
    candidates = []
    
    # Search space of realistic behavioral nudges
    for d_study in [0, 1, 2]:
        for d_abs in [0, 2, 4, 6]:
            for d_sup in [0, 1] if current_sup == 0 else [0]:
                for d_goout in [0, 1, 2]:
                    new_study = min(4, current_study + d_study)
                    new_abs = max(0, current_abs - d_abs)
                    new_sup = 1 if (current_sup == 1 or d_sup == 1) else 0
                    new_goout = max(1, current_goout - d_goout)
                    
                    # Compute new grade and pass probability
                    feat = pd.DataFrame([{
                        'G1': current_g1, 'G2': current_g2,
                        'studytime': new_study, 'absences': new_abs,
                        'failures': current_fail, 'schoolsup_yes': new_sup
                    }])
                    p_pass = float(pass_clf.predict_proba(feat)[0, 1])
                    p_g3 = float(grade_model.predict(feat)[0])
                    
                    # Compute new BVI
                    s1 = (2.0 * current_dalc + current_walc - 3.0) / 12.0
                    s2 = np.clip((new_goout - current_free + 4.0) / 8.0, 0.0, 1.0)
                    s3 = (5.0 - current_health) / 4.0
                    s4 = np.clip(new_abs / 20.0, 0.0, 1.0)
                    s5 = (5.0 - current_famrel) / 4.0
                    new_bvi = round(100.0 * (0.25 * s1 + 0.20 * s2 + 0.25 * s3 + 0.15 * s4 + 0.15 * s5), 2)
                    
                    # Cost function: minimum behavioral friction
                    cost = (d_study * 1.5) + (d_abs * 0.5) + (d_sup * 1.2) + (d_goout * 0.8)
                    
                    if p_pass >= target_pass_prob and new_bvi <= max_bvi:
                        candidates.append({
                            "cost": cost,
                            "delta_study": d_study,
                            "delta_abs": d_abs,
                            "add_schoolsup": d_sup,
                            "delta_goout": d_goout,
                            "new_study": new_study,
                            "new_abs": new_abs,
                            "new_sup": new_sup,
                            "new_goout": new_goout,
                            "new_g3": p_g3,
                            "new_pass_prob": p_pass,
                            "new_bvi": new_bvi,
                            "uplift_g3": p_g3 - baseline_pred_g3,
                            "bvi_reduction": current_bvi - new_bvi
                        })
                        
    if not candidates:
        # Relax target to find closest reachable intervention
        fallback_study = min(4, current_study + 1)
        fallback_abs = max(0, current_abs - 4)
        feat = pd.DataFrame([{
            'G1': current_g1, 'G2': current_g2,
            'studytime': fallback_study, 'absences': fallback_abs,
            'failures': current_fail, 'schoolsup_yes': 1
        }])
        p_pass = float(pass_clf.predict_proba(feat)[0, 1])
        p_g3 = float(grade_model.predict(feat)[0])
        return {
            "status": "Intensive Academic Recovery",
            "baseline_g3": baseline_pred_g3,
            "baseline_pass_prob": baseline_pass_prob,
            "baseline_bvi": current_bvi,
            "recommended_action": f"Enroll in remedial math tutoring, add +1 study bracket, and recover {current_abs - fallback_abs} absences.",
            "expected_uplift": round(p_g3 - baseline_pred_g3, 2),
            "new_bvi": current_bvi,
            "new_pass_prob": round(p_pass, 4),
            "pathway_type": "Intensive Tutoring Pathway"
        }
        
    # Sort by minimum intervention cost
    candidates.sort(key=lambda x: x['cost'])
    best = candidates[0]
    
    actions = []
    if best['delta_study'] > 0:
        actions.append(f"Increase weekly study by +{best['delta_study']} tier ({'+2-3 hrs' if best['delta_study']==1 else '+5+ hrs'})")
    if best['add_schoolsup'] == 1:
        actions.append("Enroll in extra school mathematics tutoring")
    if best['delta_abs'] > 0:
        actions.append(f"Reduce absences by {best['delta_abs']} days via structured attendance recovery")
    if best['delta_goout'] > 0:
        actions.append(f"Rebalance social outings by -{best['delta_goout']} level to restore sleep/rest")
        
    action_str = "; ".join(actions) if actions else "Maintain balanced study and social routine."
    
    return {
        "status": "Optimized Counterfactual Pathway Found",
        "baseline_g3": round(baseline_pred_g3, 2),
        "baseline_pass_prob": round(baseline_pass_prob, 4),
        "baseline_bvi": current_bvi,
        "recommended_action": action_str,
        "expected_uplift": round(best['uplift_g3'], 2),
        "new_bvi": best['new_bvi'],
        "new_pass_prob": round(best['new_pass_prob'], 4),
        "cost_score": round(best['cost'], 2),
        "pathway_type": "Targeted Balanced Uplift"
    }

def generate_cohort_prescriptions(df, grade_model, pass_clf):
    """Generates counterfactual prescriptions for the entire 395-student cohort."""
    print("\n--- Generating Cohort-Wide Counterfactual Prescriptions ---")
    prescriptions = []
    
    for idx, row in df.iterrows():
        res = optimize_counterfactual(row, grade_model, pass_clf)
        prescriptions.append({
            "student_id": idx + 1,
            "school": row['school'],
            "sex": row['sex'],
            "age": row['age'],
            "G1": row['G1'],
            "G2": row['G2'],
            "G3_actual": row['G3'],
            "BVI": row['BVI'],
            "burnout_tier": row['burnout_tier'],
            "quadrant": row['quadrant'],
            "baseline_predicted_G3": res['baseline_g3'],
            "baseline_pass_prob": res['baseline_pass_prob'],
            "prescription_status": res['status'],
            "recommended_action": res['recommended_action'],
            "expected_uplift": res['expected_uplift'],
            "target_pass_prob": res['new_pass_prob'],
            "target_bvi": res['new_bvi']
        })
        
    presc_df = pd.DataFrame(prescriptions)
    presc_df.to_csv("eduprescribe_cohort_prescriptions.csv", index=False)
    print(f" Generated prescriptions for {len(presc_df)} students -> 'eduprescribe_cohort_prescriptions.csv'")
    return presc_df

def generate_visualizations(df):
    """Generates 4 publication-grade figures for EduPrescribe."""
    print("\n--- Generating EduPrescribe Visualizations (300 DPI) ---")
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Figure 1: 4-Quadrant Academic Mastery vs Burnout Vulnerability Matrix
    plt.figure(figsize=(10, 7))
    quadrant_colors = {
        'Q1: Thriving Achievers': '#2ca02c',
        'Q2: High-Performing Burnout Risk': '#ff7f0e',
        'Q3: Dual Crisis / High Vulnerability': '#d62728',
        'Q4: Academic Disengagement / Coasting': '#1f77b4'
    }
    
    for q_label, col in quadrant_colors.items():
        sub = df[df['quadrant'] == q_label]
        plt.scatter(sub['BVI'], sub['G3'], color=col, label=f"{q_label} (N={len(sub)})", alpha=0.75, s=55, edgecolors='white')
        
    plt.axhline(10, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    plt.axvline(40, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Quadrant text labels
    plt.text(12, 18.5, "Q1: THRIVING ACHIEVERS\n(High Grade, Balanced Lifestyle)", fontsize=9, fontweight='bold', color='#1b781b', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    plt.text(45, 18.5, "Q2: BURNOUT RISK\n(High Grade, Elevated Stress)", fontsize=9, fontweight='bold', color='#c25e00', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    plt.text(45, 1.5, "Q3: DUAL CRISIS\n(Failing & Acute Strain)", fontsize=9, fontweight='bold', color='#a31313', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    plt.text(12, 1.5, "Q4: COASTING / DISENGAGED\n(Failing, Low Stress)", fontsize=9, fontweight='bold', color='#125688', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
    
    plt.title("Student Well-Being & Academic Performance Quadrant Matrix\nMapping Final Grade (G3) vs. Burnout Vulnerability Index (BVI)", fontsize=13, fontweight='bold', pad=14)
    plt.xlabel("Burnout Vulnerability Index (BVI: 0 - 100)", fontsize=11, fontweight='semibold')
    plt.ylabel("Final Mathematics Grade (G3: 0 - 20)", fontsize=11, fontweight='semibold')
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    plt.savefig("eduprescribe_figures/fig1_quadrant_matrix.png", dpi=300)
    plt.close()
    print(" [X] Saved eduprescribe_figures/fig1_quadrant_matrix.png")
    
    # Figure 2: BVI Distribution & Burnout Tiers
    plt.figure(figsize=(9, 5.5))
    sns.histplot(df['BVI'], kde=True, bins=25, color='#e377c2', edgecolor='white', alpha=0.75)
    plt.axvline(25, color='#2ca02c', linestyle=':', linewidth=1.8, label='Moderate Stress Cutoff (25)')
    plt.axvline(40, color='#ff7f0e', linestyle='--', linewidth=2.0, label='Elevated Burnout Cutoff (40)')
    plt.axvline(55, color='#d62728', linestyle='-', linewidth=2.0, label='Acute Exhaustion Cutoff (55)')
    plt.title("Distribution of Student Burnout Vulnerability Index (BVI)\nSynthesized across alcohol coping, rest imbalance, health, and absences", fontsize=13, fontweight='bold', pad=12)
    plt.xlabel("Burnout Vulnerability Index (BVI)", fontsize=11, fontweight='semibold')
    plt.ylabel("Student Frequency", fontsize=11, fontweight='semibold')
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig("eduprescribe_figures/fig2_bvi_distribution.png", dpi=300)
    plt.close()
    print(" [X] Saved eduprescribe_figures/fig2_bvi_distribution.png")

def main():
    print("=" * 75)
    print(" EduPrescribe AI: Behavioral Wellness & Counterfactual Uplift Pipeline")
    print("=" * 75)
    
    raw_df = pd.read_csv("student-mat.csv", sep=';')
    print(f" Loaded raw dataset: {raw_df.shape[0]} students")
    
    # 1. Compute BVI & Quadrants
    df_bvi = compute_burnout_index(raw_df)
    df_proc = assign_quadrants(df_bvi)
    
    print("\nQuadrant Distribution:")
    print(df_proc['quadrant'].value_counts())
    print("\nBurnout Tier Distribution:")
    print(df_proc['burnout_tier'].value_counts())
    
    # 2. Train Prescriptive Models
    grade_model, pass_clf, burnout_rf = train_prescriptive_models(df_proc)
    
    # 3. Generate Cohort Prescriptions
    presc_df = generate_cohort_prescriptions(df_proc, grade_model, pass_clf)
    
    # 4. Generate Visualizations
    generate_visualizations(df_proc)
    
    # Export summary metadata
    summary = {
        "project": "EduPrescribe AI - Student Wellness & Prescriptive Uplift",
        "cohort_size": len(df_proc),
        "quadrant_counts": df_proc['quadrant'].value_counts().to_dict(),
        "burnout_tier_counts": df_proc['burnout_tier'].value_counts().to_dict(),
        "mean_bvi": float(df_proc['BVI'].mean()),
        "median_bvi": float(df_proc['BVI'].median()),
        "at_risk_students_count": int((df_proc['G3'] < 10).sum()),
        "burnout_vulnerable_count": int((df_proc['BVI'] >= 40).sum()),
        "hidden_burnout_achievers_count": int(((df_proc['G3'] >= 10) & (df_proc['BVI'] >= 40)).sum())
    }
    with open("eduprescribe_models/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 75)
    print(" EduPrescribe AI Pipeline executed successfully.")
    print(" Deliverables generated:")
    print("  - eduprescribe_cohort_prescriptions.csv (Action plans for all 395 students)")
    print("  - eduprescribe_models/*.joblib (Predictive & Uplift models)")
    print("  - eduprescribe_figures/*.png (Matrix & Distribution figures)")
    print("=" * 75)

if __name__ == "__main__":
    main()
