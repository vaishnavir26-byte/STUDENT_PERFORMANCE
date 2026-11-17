#!/usr/bin/env python3
"""
EduPrescribe AI: Student Burnout Diagnosis & Counterfactual Prescriber CLI
========================================================================
Interactive diagnosis and prescriptive action planner for academic advisors.

Usage Examples:
1. Diagnose a student with high grades but severe burnout risk:
   python3 eduprescribe_cli.py --g1 14 --g2 14 --studytime 3 --absences 14 --dalc 2 --walc 4 --goout 5 --freetime 2 --health 2

2. Diagnose an at-risk student needing counterfactual recovery:
   python3 eduprescribe_cli.py --g1 7 --g2 7 --studytime 1 --absences 8 --failures 1 --dalc 1 --walc 2 --goout 4 --freetime 3

3. Batch inspect a student by ID from the cohort prescriptions:
   python3 eduprescribe_cli.py --student-id 42
"""

import sys
import os
import argparse
import joblib
import numpy as np
import pandas as pd

# Inline counterfactual optimization logic to decouple CLI from matplotlib/plotting
def optimize_counterfactual(row, grade_model, pass_clf, target_pass_prob=0.85, max_bvi=38.0):
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
    for d_study in [0, 1, 2]:
        for d_abs in [0, 2, 4, 6]:
            for d_sup in [0, 1] if current_sup == 0 else [0]:
                for d_goout in [0, 1, 2]:
                    new_study = min(4, current_study + d_study)
                    new_abs = max(0, current_abs - d_abs)
                    new_sup = 1 if (current_sup == 1 or d_sup == 1) else 0
                    new_goout = max(1, current_goout - d_goout)
                    
                    feat = pd.DataFrame([{
                        'G1': current_g1, 'G2': current_g2,
                        'studytime': new_study, 'absences': new_abs,
                        'failures': current_fail, 'schoolsup_yes': new_sup
                    }])
                    p_pass = float(pass_clf.predict_proba(feat)[0, 1])
                    p_g3 = float(grade_model.predict(feat)[0])
                    
                    s1 = (2.0 * current_dalc + current_walc - 3.0) / 12.0
                    s2 = np.clip((new_goout - current_free + 4.0) / 8.0, 0.0, 1.0)
                    s3 = (5.0 - current_health) / 4.0
                    s4 = np.clip(new_abs / 20.0, 0.0, 1.0)
                    s5 = (5.0 - current_famrel) / 4.0
                    new_bvi = round(100.0 * (0.25 * s1 + 0.20 * s2 + 0.25 * s3 + 0.15 * s4 + 0.15 * s5), 2)
                    
                    cost = (d_study * 1.5) + (d_abs * 0.5) + (d_sup * 1.2) + (d_goout * 0.8)
                    if p_pass >= target_pass_prob and new_bvi <= max_bvi:
                        candidates.append({
                            "cost": cost, "delta_study": d_study, "delta_abs": d_abs,
                            "add_schoolsup": d_sup, "delta_goout": d_goout,
                            "new_study": new_study, "new_abs": new_abs, "new_sup": new_sup,
                            "new_goout": new_goout, "new_g3": p_g3, "new_pass_prob": p_pass,
                            "new_bvi": new_bvi, "uplift_g3": p_g3 - baseline_pred_g3
                        })
                        
    if not candidates:
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
            "baseline_g3": baseline_pred_g3, "baseline_pass_prob": baseline_pass_prob,
            "baseline_bvi": current_bvi,
            "recommended_action": f"Enroll in remedial math tutoring, add +1 study bracket, and recover {current_abs - fallback_abs} absences.",
            "expected_uplift": round(p_g3 - baseline_pred_g3, 2),
            "new_bvi": current_bvi, "new_pass_prob": round(p_pass, 4),
            "pathway_type": "Intensive Tutoring Pathway"
        }
        
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

def load_eduprescribe_models():
    grade_model = joblib.load("eduprescribe_models/grade_regressor.joblib")
    pass_clf = joblib.load("eduprescribe_models/pass_classifier.joblib")
    burnout_rf = joblib.load("eduprescribe_models/burnout_regressor.joblib")
    return grade_model, pass_clf, burnout_rf

def diagnose_student(args, grade_model, pass_clf, burnout_rf):
    # Construct student record
    student = {
        'G1': args.g1, 'G2': args.g2,
        'studytime': args.studytime, 'absences': args.absences,
        'failures': args.failures, 'schoolsup': args.schoolsup,
        'Dalc': args.dalc, 'Walc': args.walc,
        'goout': args.goout, 'freetime': args.freetime,
        'health': args.health, 'famrel': args.famrel
    }
    
    # Compute BVI
    s1 = (2.0 * student['Dalc'] + student['Walc'] - 3.0) / 12.0
    s2 = np.clip((student['goout'] - student['freetime'] + 4.0) / 8.0, 0.0, 1.0)
    s3 = (5.0 - student['health']) / 4.0
    s4 = np.clip(student['absences'] / 20.0, 0.0, 1.0)
    s5 = (5.0 - student['famrel']) / 4.0
    bvi = round(100.0 * (0.25 * s1 + 0.20 * s2 + 0.25 * s3 + 0.15 * s4 + 0.15 * s5), 2)
    student['BVI'] = bvi
    
    # Baseline predictions
    feat = pd.DataFrame([{
        'G1': student['G1'], 'G2': student['G2'],
        'studytime': student['studytime'], 'absences': student['absences'],
        'failures': student['failures'],
        'schoolsup_yes': 1 if str(student['schoolsup']).lower() == 'yes' else 0
    }])
    baseline_pred_g3 = float(grade_model.predict(feat)[0])
    baseline_pass_prob = float(pass_clf.predict_proba(feat)[0, 1])
    
    # Quadrant assignment
    if baseline_pred_g3 >= 10 and bvi < 40:
        quadrant = "Q1: Thriving Achiever"
        q_badge = "🟢"
        q_desc = "High academic mastery with sustainable lifestyle balance."
    elif baseline_pred_g3 >= 10 and bvi >= 40:
        quadrant = "Q2: High-Performing Burnout Risk"
        q_badge = "🟡"
        q_desc = "Academic passing marks, but elevated stress and behavioral exhaustion."
    elif baseline_pred_g3 < 10 and bvi >= 40:
        quadrant = "Q3: Dual Crisis / High Vulnerability"
        q_badge = "🔴"
        q_desc = "Academic at-risk failure compounded by severe lifestyle/behavioral fatigue."
    else:
        quadrant = "Q4: Academic Disengagement / Coasting"
        q_badge = "🔵"
        q_desc = "Academic at-risk status without lifestyle burnout (motivation or curriculum pacing issue)."
        
    # Counterfactual prescription
    prescription = optimize_counterfactual(student, grade_model, pass_clf)
    
    print("\n" + "=" * 70)
    print(" 🌟 EDUPRESCRIBE AI: STUDENT DIAGNOSIS & PRESCRIPTION REPORT")
    print("=" * 70)
    print(" 👤 Student Profile Overview:")
    print(f"  • Academic Inputs      : G1 = {student['G1']:.1f} | G2 = {student['G2']:.1f} | Failures = {student['failures']}")
    print(f"  • Behavioral Inputs    : Study = {student['studytime']} (tier) | Absences = {student['absences']} days | Tutoring = {student['schoolsup']}")
    print(f"  • Lifestyle Stressors  : Health = {student['health']}/5 | GoOut = {student['goout']}/5 | Dalc/Walc = {student['Dalc']}/{student['Walc']}")
    print("-" * 70)
    print(" 🩺 Holistic Diagnostic Findings:")
    print(f"  • Predicted Final Grade (G3) : {baseline_pred_g3:.2f} / 20")
    print(f"  • Probability of Passing     : {baseline_pass_prob:.1%}")
    print(f"  • Burnout Vulnerability (BVI): {bvi:.1f} / 100")
    print(f"  • Well-Being Quadrant        : {q_badge} {quadrant}")
    print(f"    ↳ Interpretation           : {q_desc}")
    print("-" * 70)
    print(" 💡 Prescriptive Counterfactual Action Plan:")
    print(f"  • Strategy Category          : {prescription.get('pathway_type', 'Prescription')}")
    print(f"  • Recommended Action         : {prescription.get('recommended_action', 'None')}")
    print(f"  • Projected Grade Uplift     : +{prescription.get('expected_uplift', 0.0):.2f} points (Target: {baseline_pred_g3 + prescription.get('expected_uplift', 0.0):.2f})")
    print(f"  • Projected Passing Chance   : {prescription.get('new_pass_prob', baseline_pass_prob):.1%}")
    print(f"  • Projected Burnout Index    : {prescription.get('new_bvi', bvi):.1f} / 100")
    print("=" * 70)
    print(" Guidance: Prescriptions provide actionable decision-support for school")
    print(" counselors and mentors to foster positive behavioral development.")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="EduPrescribe AI - Counterfactual Study Optimizer CLI")
    parser.add_argument("--g1", type=float, default=9.0, help="Period 1 grade (0-20)")
    parser.add_argument("--g2", type=float, default=8.5, help="Period 2 grade (0-20)")
    parser.add_argument("--studytime", type=int, default=1, help="Weekly study time (1: <2h, 2: 2-5h, 3: 5-10h, 4: >10h)")
    parser.add_argument("--absences", type=int, default=8, help="Recorded school absences (0-93)")
    parser.add_argument("--failures", type=int, default=1, help="Number of past class failures (0-4)")
    parser.add_argument("--schoolsup", type=str, default="no", help="Extra school educational support (yes/no)")
    parser.add_argument("--dalc", type=int, default=1, help="Workday alcohol consumption (1-5)")
    parser.add_argument("--walc", type=int, default=2, help="Weekend alcohol consumption (1-5)")
    parser.add_argument("--goout", type=int, default=4, help="Going out with friends (1-5)")
    parser.add_argument("--freetime", type=int, default=3, help="Free time after school (1-5)")
    parser.add_argument("--health", type=int, default=3, help="Current health status (1-5)")
    parser.add_argument("--famrel", type=int, default=4, help="Family relationship quality (1-5)")
    parser.add_argument("--student-id", type=int, help="Inspect precomputed prescription for a specific student ID (1-395)")
    
    args = parser.parse_args()
    grade_model, pass_clf, burnout_rf = load_eduprescribe_models()
    
    if args.student_id:
        if os.path.exists("eduprescribe_cohort_prescriptions.csv"):
            df_cohort = pd.read_csv("eduprescribe_cohort_prescriptions.csv")
            sub = df_cohort[df_cohort['student_id'] == args.student_id]
            if len(sub) > 0:
                row = sub.iloc[0]
                print(f"\nCohort Record for Student #{args.student_id}:")
                for k, v in row.items():
                    print(f"  {k:22}: {v}")
                print()
                return
        print(f"Student ID #{args.student_id} not found in cohort prescriptions.")
    else:
        diagnose_student(args, grade_model, pass_clf, burnout_rf)

if __name__ == "__main__":
    main()
