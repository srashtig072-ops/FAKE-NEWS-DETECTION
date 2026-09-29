"""
Evaluation and Metrics Generation Module for Fake News Detection System.

This module implements Phase 7:
- Confusion Matrix computation and visualization
- ROC Curve and AUC calculation
- Precision-Recall Curve calculation
- Per-class Classification Report (Precision, Recall, F1-score, Support)
- Cross-validation score plotting
- Multi-algorithm benchmark comparison reporting
- Export of all metrics to CSV and PNG formats
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.feature_engineering import NewsFeaturePipeline
from src import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ModelEvaluator:
    """
    Evaluates saved champion model or candidate models on unseen test data
    and generates report artifacts.
    """

    def __init__(
        self,
        model_path: str = "models/model.pkl",
        vectorizer_path: str = "models/vectorizer.pkl",
        data_path: str = "data/processed_news.csv",
        reports_dir: str = "reports"
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.data_path = data_path
        self.reports_dir = reports_dir
        self.figures_dir = os.path.join(self.reports_dir, "figures")
        self.metrics_dir = os.path.join(self.reports_dir, "metrics")

        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

        self.model = None
        self.pipeline = None

    def load_artifacts(self):
        """Load trained model and vectorizer pipeline from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Train the model first!")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at: {self.vectorizer_path}")

        logger.info(f"Loading model from: {self.model_path}")
        self.model = joblib.load(self.model_path)
        logger.info(f"Loading feature engineering pipeline from: {self.vectorizer_path}")
        self.pipeline = NewsFeaturePipeline.load_pipeline(self.vectorizer_path)

    def evaluate_test_set(
        self,
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Evaluate champion model on held-out 20% test partition.
        """
        self.load_artifacts()

        logger.info(f"Loading dataset from: {self.data_path}")
        df = pd.read_csv(self.data_path)
        y = df["label"].values

        _, X_test_df, _, y_test = train_test_split(
            df, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"Evaluating model on {len(X_test_df):,} test articles...")

        X_test_matrix = self.pipeline.transform(X_test_df)

        # Predictions
        y_pred = self.model.predict(X_test_matrix)

        if hasattr(self.model, "predict_proba"):
            y_prob = self.model.predict_proba(X_test_matrix)[:, 1]
        elif hasattr(self.model, "decision_function"):
            dec = self.model.decision_function(X_test_matrix)
            y_prob = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
        else:
            y_prob = y_pred

        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Fake (0)", "Actual Real (1)"],
            columns=["Predicted Fake (0)", "Predicted Real (1)"]
        )
        cm_csv_path = os.path.join(self.metrics_dir, "confusion_matrix.csv")
        cm_df.to_csv(cm_csv_path)
        logger.info(f"Saved confusion matrix CSV to: {cm_csv_path}")

        cm_png_path = os.path.join(self.figures_dir, "confusion_matrix.png")
        utils.plot_confusion_matrix_figure(cm, output_path=cm_png_path, model_name="Champion Model")

        # 2. ROC Curve
        roc_png_path = os.path.join(self.figures_dir, "roc_curve.png")
        utils.plot_roc_curve_figure(y_test, y_prob, output_path=roc_png_path, model_name="Champion Model")

        # 3. Precision-Recall Curve
        pr_png_path = os.path.join(self.figures_dir, "precision_recall_curve.png")
        utils.plot_precision_recall_curve_figure(y_test, y_prob, output_path=pr_png_path, model_name="Champion Model")

        # 4. Classification Report
        report_dict = classification_report(
            y_test, y_pred,
            target_names=["Fake News (0)", "Real News (1)"],
            output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_csv_path = os.path.join(self.metrics_dir, "classification_report.csv")
        report_df.to_csv(report_csv_path)

        report_json_path = os.path.join(self.metrics_dir, "classification_report.json")
        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=4)
        logger.info(f"Saved classification report to: {report_csv_path}")

        # 5. Plot CV score visualization if CV file exists
        cv_csv_path = os.path.join(self.metrics_dir, "cross_validation_scores.csv")
        if os.path.exists(cv_csv_path):
            cv_df = pd.read_csv(cv_csv_path)
            self._plot_cv_bars(cv_df, os.path.join(self.figures_dir, "cv_scores.png"))

        metrics_summary = {
            "test_samples": int(len(y_test)),
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4)
        }

        logger.info("\n" + "="*50 + "\nCLASSIFICATION REPORT\n" + "="*50)
        logger.info("\n" + classification_report(y_test, y_pred, target_names=["Fake (0)", "Real (1)"]))
        logger.info(f"Confusion Matrix:\n{cm}")

        return metrics_summary

    def _plot_cv_bars(self, cv_df: pd.DataFrame, output_path: str):
        """Generate bar chart showing 5-fold cross validation score stability."""
        fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
        x = np.arange(len(cv_df))
        means = cv_df["CV Mean"].values
        stds = cv_df["CV Std"].values
        models = cv_df["Model"].values

        bars = ax.bar(x, means, yerr=stds, capsize=5, color=utils.PALETTE["neutral"], edgecolor="none", width=0.55)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
        ax.set_ylim(0.7, 1.05)
        ax.set_ylabel("5-Fold Mean F1 Score", fontsize=11)
        ax.set_title("Cross-Validation Stability Across Algorithms", fontsize=13, fontweight="bold", pad=12)
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold"
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved cross-validation score plot to: {output_path}")


if __name__ == "__main__":
    evaluator = ModelEvaluator()
    summary = evaluator.evaluate_test_set()
    print("\n--- Model Evaluation Summary ---")
    print(json.dumps(summary, indent=4))
