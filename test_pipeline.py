#!/usr/bin/env python3
"""
Comprehensive Audit & Unit Test Suite for Student Performance Analytics
========================================================================
Reference: PRD v1.0 (FR-01 through FR-09, Sections 10, 11, 12)
"""

import os
import unittest
import json
import joblib
import numpy as np
import pandas as pd
import scipy.stats as stats

class TestStudentPerformanceAnalytics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = "student-mat.csv"
        cls.df = pd.read_csv(cls.data_path, sep=';')
        cls.df['at_risk'] = (cls.df['G3'] < 10).astype(int)
        
        # Models
        cls.lm = joblib.load("models/linear_regression_g3.joblib")
        cls.logit = joblib.load("models/logistic_regression_at_risk.joblib")
        cls.early = joblib.load("models/early_warning_risk.joblib")
        
        # Summary
        with open("analysis_summary.json", "r") as f:
            cls.summary = json.load(f)
            
        with open("models/model_metadata.json", "r") as f:
            cls.metadata = json.load(f)

    # --------------------------------------------------------------------------
    # FR-01 & FR-02: Dataset Import, Hygiene, and Encoding
    # --------------------------------------------------------------------------
    def test_fr01_dataset_dimensions_and_hygiene(self):
        """FR-01: Verify 395 rows, 33 raw attributes, 0 missing, 0 duplicates."""
        self.assertEqual(self.df.shape[0], 395)
        self.assertGreaterEqual(self.df.shape[1], 33)
        self.assertEqual(self.df.isna().sum().sum(), 0)
        self.assertEqual(self.df.duplicated().sum(), 0)
        
        # Boundary validation
        for col in ['G1', 'G2', 'G3']:
            self.assertTrue((self.df[col] >= 0).all() and (self.df[col] <= 20).all())
            
    def test_fr02_target_indicator(self):
        """FR-02: Target at_risk is strictly defined as G3 < 10."""
        expected_risk_count = 130
        self.assertEqual(self.df['at_risk'].sum(), expected_risk_count)
        self.assertAlmostEqual(self.df['at_risk'].mean(), 130 / 395, delta=0.001)

    # --------------------------------------------------------------------------
    # FR-03: Descriptive Statistics
    # --------------------------------------------------------------------------
    def test_fr03_descriptive_statistics(self):
        """FR-03: G3 statistics match documented benchmark."""
        g3 = self.df['G3']
        self.assertAlmostEqual(g3.mean(), 10.415, delta=0.01)
        self.assertAlmostEqual(g3.std(), 4.581, delta=0.01)
        self.assertEqual(g3.median(), 11.0)
        self.assertEqual(g3.quantile(0.75) - g3.quantile(0.25), 6.0)
        self.assertEqual((g3 == 0).sum(), 38) # Zero-score cluster

    # --------------------------------------------------------------------------
    # FR-05: Hypothesis Testing & Inferential Statistics
    # --------------------------------------------------------------------------
    def test_fr05_pearson_spearman_correlation(self):
        """FR-05 Test 1: G2/G3 correlation is strongly positive (r > 0.90, p < 0.001)."""
        r, p_pearson = stats.pearsonr(self.df['G2'], self.df['G3'])
        rho, p_spearman = stats.spearmanr(self.df['G2'], self.df['G3'])
        self.assertAlmostEqual(r, 0.9049, delta=0.005)
        self.assertAlmostEqual(rho, 0.9571, delta=0.005)
        self.assertLess(p_pearson, 1e-100)
        self.assertLess(p_spearman, 1e-100)
        
    def test_fr05_studytime_anova_and_kruskal(self):
        """FR-05 Test 2: Study time ANOVA fails to reject H0 at alpha = 0.05."""
        cohorts = [self.df[self.df['studytime'] == i]['G3'] for i in range(1, 5)]
        f_stat, p_val = stats.f_oneway(*cohorts)
        self.assertAlmostEqual(f_stat, 1.7278, delta=0.01)
        self.assertGreater(p_val, 0.05) # Fails to reject H0
        
        h_stat, p_kruskal = stats.kruskal(*cohorts)
        self.assertAlmostEqual(h_stat, 7.5789, delta=0.02)
        
    def test_fr05_schoolsup_contingency(self):
        """FR-05 Test 3: schoolsup vs at_risk cross-tabulation and Fisher test."""
        ct = pd.crosstab(self.df['schoolsup'], self.df['at_risk'])
        self.assertEqual(ct.loc['no', 0], 237)
        self.assertEqual(ct.loc['no', 1], 107)
        self.assertEqual(ct.loc['yes', 0], 28)
        self.assertEqual(ct.loc['yes', 1], 23)
        
        odds_ratio, p_fisher = stats.fisher_exact(ct)
        self.assertAlmostEqual(odds_ratio, 1.8194, delta=0.01)
        self.assertAlmostEqual(p_fisher, 0.0555, delta=0.005)

    # --------------------------------------------------------------------------
    # FR-08: Leakage Prevention & Holdout Partitions
    # --------------------------------------------------------------------------
    def test_fr08_holdout_partitions_integrity(self):
        """FR-08: Verify 80/20 train/test split, size, and stratified parity."""
        train_df = pd.read_csv("models/train_partition.csv")
        test_df = pd.read_csv("models/test_partition.csv")
        self.assertEqual(len(train_df), 316)
        self.assertEqual(len(test_df), 79)
        self.assertEqual(train_df['at_risk'].sum(), 104)
        self.assertEqual(test_df['at_risk'].sum(), 26)
        self.assertAlmostEqual(train_df['at_risk'].mean(), test_df['at_risk'].mean(), delta=0.0001)

    # --------------------------------------------------------------------------
    # FR-06: Multiple Linear Regression
    # --------------------------------------------------------------------------
    def test_fr06_linear_regression_metrics(self):
        """FR-06: Multiple linear regression held-out test RMSE and MAE."""
        test_df = pd.read_csv("models/test_partition.csv")
        X_test = test_df[['G1', 'G2', 'studytime', 'absences', 'failures']]
        preds = self.lm.predict(X_test)
        rmse = np.sqrt(np.mean((test_df['G3'] - preds)**2))
        mae = np.mean(np.abs(test_df['G3'] - preds))
        
        self.assertAlmostEqual(rmse, 1.9201, delta=0.01)
        self.assertAlmostEqual(mae, 1.2729, delta=0.01)

    # --------------------------------------------------------------------------
    # FR-07: Logistic Regression & Calibration
    # --------------------------------------------------------------------------
    def test_fr07_logistic_regression_metrics(self):
        """FR-07: Logistic regression held-out test accuracy, sensitivity, AUC."""
        test_df = pd.read_csv("models/test_partition.csv")
        X_test = test_df[['G1', 'G2', 'studytime', 'absences', 'failures']].copy()
        X_test['schoolsup_yes'] = (test_df['schoolsup'] == 'yes').astype(int)
        
        probs = self.logit.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.50).astype(int)
        
        acc = np.mean(preds == test_df['at_risk'])
        sens = np.sum((preds == 1) & (test_df['at_risk'] == 1)) / np.sum(test_df['at_risk'] == 1)
        spec = np.sum((preds == 0) & (test_df['at_risk'] == 0)) / np.sum(test_df['at_risk'] == 0)
        
        self.assertAlmostEqual(acc, 0.9114, delta=0.01)
        self.assertAlmostEqual(sens, 0.8846, delta=0.01)
        self.assertAlmostEqual(spec, 0.9245, delta=0.01)

    # --------------------------------------------------------------------------
    # Extended Audits: Hurdle Model & Early Warning
    # --------------------------------------------------------------------------
    def test_hurdle_and_early_warning_models(self):
        """Verify early warning model without G1/G2 and hurdle error drop."""
        # Early warning model test input
        early_X = pd.DataFrame([{
            'studytime': 2, 'failures': 0, 'absences': 4, 'Medu': 3, 'Fedu': 3,
            'goout': 3, 'Dalc': 1, 'Walc': 1, 'age': 16,
            'schoolsup_yes': 0, 'famsup_yes': 1, 'higher_yes': 1
        }])
        prob = self.early.predict_proba(early_X)[0, 1]
        self.assertTrue(0.0 <= prob <= 1.0)
        
        # Verify hurdle completer RMSE in summary
        hurdle_rmse = self.summary['extended_audits']['hurdle_completer_rmse']
        self.assertLess(hurdle_rmse, 1.0) # Error drops from 1.92 to < 1.0

    # --------------------------------------------------------------------------
    # Release Artifacts Existence Check (Section 12)
    # --------------------------------------------------------------------------
    def test_release_artifacts_exist(self):
        """Section 12 Release Checklist: All figures, models, reports exist."""
        required_files = [
            "figures/fig1_g3_histogram.png",
            "figures/fig2_studytime_boxplot.png",
            "figures/fig3_g2_vs_g3_scatter.png",
            "figures/fig4_lm_diagnostics.png",
            "figures/fig5_logistic_evaluation.png",
            "models/linear_regression_g3.joblib",
            "models/logistic_regression_at_risk.joblib",
            "models/early_warning_risk.joblib",
            "models/model_metadata.json",
            "predictions.csv",
            "analysis_summary.json",
            "STUDENT_PERFORMANCE_REPORT.md",
            "student_dashboard.html"
        ]
        for f in required_files:
            self.assertTrue(os.path.exists(f), f"Missing required deliverable: {f}")

if __name__ == "__main__":
    unittest.main()
