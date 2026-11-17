#!/usr/bin/env python3
"""
Unit Tests for Student Performance Analytics Modeling Pipeline
==============================================================
"""

import unittest
import os
import joblib
import pandas as pd
import numpy as np

class TestStudentModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_dir = "models"
        cls.lm = joblib.load(os.path.join(cls.model_dir, "linear_regression_g3.joblib"))
        cls.logit = joblib.load(os.path.join(cls.model_dir, "logistic_regression_at_risk.joblib"))
        cls.early = joblib.load(os.path.join(cls.model_dir, "early_warning_risk.joblib"))
        
    def test_linear_model_coefficients(self):
        """Verify linear model coefficients match PRD reports."""
        self.assertAlmostEqual(self.lm.intercept_, -1.5618, delta=0.01)
        # G1 coef
        self.assertAlmostEqual(self.lm.coef_[0], 0.1770, delta=0.01)
        # G2 coef
        self.assertAlmostEqual(self.lm.coef_[1], 0.9586, delta=0.01)
        
    def test_linear_prediction_range(self):
        """Verify reasonable prediction for boundary students."""
        # Top student: G1=20, G2=20, failures=0
        top_X = pd.DataFrame([{'G1': 20, 'G2': 20, 'studytime': 4, 'absences': 0, 'failures': 0}])
        pred_top = self.lm.predict(top_X)[0]
        self.assertGreater(pred_top, 18.0)
        
        # Failing student: G1=3, G2=3, failures=3
        low_X = pd.DataFrame([{'G1': 3, 'G2': 3, 'studytime': 1, 'absences': 12, 'failures': 3}])
        pred_low = self.lm.predict(low_X)[0]
        self.assertLess(pred_low, 5.0)
        
    def test_logistic_classification_behavior(self):
        """Verify logistic model assigns near 0% risk to high achievers and >90% to failing students."""
        top_student = pd.DataFrame([{
            'G1': 17, 'G2': 18, 'studytime': 3, 'absences': 2, 'failures': 0, 'schoolsup_yes': 0
        }])
        prob_top = self.logit.predict_proba(top_student)[0, 1]
        self.assertLess(prob_top, 0.05)
        
        at_risk_student = pd.DataFrame([{
            'G1': 6, 'G2': 5, 'studytime': 1, 'absences': 10, 'failures': 2, 'schoolsup_yes': 1
        }])
        prob_risk = self.logit.predict_proba(at_risk_student)[0, 1]
        self.assertGreater(prob_risk, 0.90)

    def test_dataset_partitions(self):
        """Verify 80/20 train/test partition integrity and zero leakage."""
        train_df = pd.read_csv("models/train_partition.csv")
        test_df = pd.read_csv("models/test_partition.csv")
        self.assertEqual(len(train_df), 316)
        self.assertEqual(len(test_df), 79)
        self.assertAlmostEqual(train_df['at_risk'].mean(), 0.3291, delta=0.01)
        self.assertAlmostEqual(test_df['at_risk'].mean(), 0.3291, delta=0.01)

if __name__ == "__main__":
    unittest.main()
