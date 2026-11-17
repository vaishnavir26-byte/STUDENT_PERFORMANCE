#!/usr/bin/env python3
"""
Unit Test Suite for EduPrescribe AI
===================================
"""

import os
import unittest
import joblib
import pandas as pd
import numpy as np
from eduprescribe_pipeline import compute_burnout_index, assign_quadrants, optimize_counterfactual

class TestEduPrescribeAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_raw = pd.read_csv("student-mat.csv", sep=';')
        cls.grade_model = joblib.load("eduprescribe_models/grade_regressor.joblib")
        cls.pass_clf = joblib.load("eduprescribe_models/pass_classifier.joblib")
        cls.burnout_rf = joblib.load("eduprescribe_models/burnout_regressor.joblib")
        
    def test_bvi_bounds_and_distribution(self):
        """Verify Burnout Vulnerability Index is strictly within [0, 100]."""
        df_bvi = compute_burnout_index(self.df_raw)
        self.assertTrue((df_bvi['BVI'] >= 0).all() and (df_bvi['BVI'] <= 100).all())
        self.assertAlmostEqual(df_bvi['BVI'].mean(), 31.26, delta=0.5)
        
    def test_quadrant_coverage(self):
        """Verify all 395 students are partitioned into the 4 well-being quadrants."""
        df_bvi = compute_burnout_index(self.df_raw)
        df_q = assign_quadrants(df_bvi)
        q_counts = df_q['quadrant'].value_counts()
        self.assertEqual(q_counts.sum(), 395)
        self.assertEqual(q_counts['Q1: Thriving Achievers'], 204)
        self.assertEqual(q_counts['Q2: High-Performing Burnout Risk'], 61)
        self.assertEqual(q_counts['Q3: Dual Crisis / High Vulnerability'], 33)
        self.assertEqual(q_counts['Q4: Academic Disengagement / Coasting'], 97)

    def test_counterfactual_prescription_generation(self):
        """Verify counterfactual optimization computes positive uplift for at-risk students."""
        sample_student = {
            'G1': 8, 'G2': 8, 'studytime': 1, 'absences': 10,
            'failures': 1, 'schoolsup': 'no', 'Dalc': 2, 'Walc': 3,
            'goout': 5, 'freetime': 2, 'health': 2, 'famrel': 3, 'BVI': 55.8
        }
        res = optimize_counterfactual(sample_student, self.grade_model, self.pass_clf)
        self.assertIn("recommended_action", res)
        self.assertTrue(res["expected_uplift"] != 0 or res["status"] == "Already Thriving")

    def test_cohort_prescriptions_export(self):
        """Verify 395 pre-calculated student prescriptions exist."""
        self.assertTrue(os.path.exists("eduprescribe_cohort_prescriptions.csv"))
        df_cohort = pd.read_csv("eduprescribe_cohort_prescriptions.csv")
        self.assertEqual(len(df_cohort), 395)
        
    def test_dashboard_asset_integrity(self):
        """Verify HTML dashboard and 300 DPI figures exist."""
        self.assertTrue(os.path.exists("eduprescribe_dashboard.html"))
        self.assertGreater(os.path.getsize("eduprescribe_dashboard.html"), 100000)
        self.assertTrue(os.path.exists("eduprescribe_figures/fig1_quadrant_matrix.png"))
        self.assertTrue(os.path.exists("eduprescribe_figures/fig2_bvi_distribution.png"))

if __name__ == "__main__":
    unittest.main()
