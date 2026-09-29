"""
Model Training Module for Fake News Detection System.

This module implements Phase 5:
- Stratified 80/20 Train-Test Split
- 5-Fold Cross Validation
- GridSearchCV Hyperparameter Optimization
- Multi-Model Benchmarking:
    1. Logistic Regression
    2. Linear SVM (Calibrated for probability estimation)
    3. Multinomial Naive Bayes
    4. Random Forest Classifier
    5. XGBoost Classifier
- Multi-Metric Evaluation:
    - Accuracy
    - Precision
    - Recall
    - F1 Score
    - ROC-AUC
- Automatic selection of the best performing model
- Model serialization (model.pkl, vectorizer.pkl, label_encoder.pkl)
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from src.data_loader import NewsDataLoader
from src.feature_engineering import NewsFeaturePipeline
from src import utils

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ModelTrainer:
    """
    Orchestrates ML model training, cross-validation, hyperparameter tuning,
    benchmarking, and model artifact persistence.
    """

    def __init__(
        self,
        data_path: str = "data/processed_news.csv",
        models_dir: str = "models",
        reports_dir: str = "reports",
        random_state: int = 42
    ):
        self.data_path = data_path
        self.models_dir = models_dir
        self.reports_dir = reports_dir
        self.random_state = random_state

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "figures"), exist_ok=True)
        os.makedirs(os.path.join(self.reports_dir, "metrics"), exist_ok=True)

        self.pipeline: NewsFeaturePipeline = None
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.best_score: float = -1.0
        self.results_df: pd.DataFrame = pd.DataFrame()
        self.trained_models: Dict[str, Any] = {}

    def prepare_data(
        self,
        sample_size: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load processed data and partition into 80% train and 20% test sets using Stratified Split.
        """
        if not os.path.exists(self.data_path):
            logger.info(f"{self.data_path} not found. Running data loader...")
            loader = NewsDataLoader()
            df, _ = loader.run_pipeline()
        else:
            logger.info(f"Loading processed dataset from: {self.data_path}")
            df = pd.read_csv(self.data_path)

        if sample_size and sample_size < len(df):
            logger.info(f"Sampling {sample_size:,} records for fast execution...")
            df = df.sample(n=sample_size, random_state=self.random_state).reset_index(drop=True)

        logger.info(f"Total dataset size for modeling: {len(df):,} samples.")
        
        y = df["label"].values

        # Stratified 80/20 train/test split
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            df,
            y,
            test_size=0.20,
            random_state=self.random_state,
            stratify=y
        )
        logger.info(f"Train set: {len(X_train_df):,} samples | Test set: {len(X_test_df):,} samples.")

        # Fit feature pipeline on training set only
        logger.info("Initializing and fitting Feature Engineering Pipeline (TF-IDF + Stylometrics)...")
        self.pipeline = NewsFeaturePipeline(
            max_features=15000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.pipeline.fit(X_train_df)
        self.pipeline.save_pipeline(os.path.join(self.models_dir, "vectorizer.pkl"))

        # Save label encoder
        label_encoder = LabelEncoder()
        label_encoder.fit(["FAKE", "REAL"])
        joblib.dump(label_encoder, os.path.join(self.models_dir, "label_encoder.pkl"))
        logger.info("Saved label encoder to models/label_encoder.pkl")

        return X_train_df, X_test_df, y_train, y_test

    def get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """
        Define candidate models and hyperparameter grids for GridSearchCV.
        """
        models_config = {
            "Logistic Regression": {
                "estimator": LogisticRegression(max_iter=1000, random_state=self.random_state),
                "param_grid": {
                    "C": [1.0, 5.0],
                    "solver": ["liblinear"]
                }
            },
            "Linear SVM": {
                # Wrap LinearSVC in CalibratedClassifierCV to enable probability outputs for ROC/SHAP
                "estimator": CalibratedClassifierCV(
                    estimator=LinearSVC(random_state=self.random_state, max_iter=2000, dual="auto"),
                    cv=3
                ),
                "param_grid": {
                    "estimator__C": [0.5, 1.0]
                }
            },
            "Multinomial Naive Bayes": {
                "estimator": MultinomialNB(),
                "param_grid": {
                    "alpha": [0.1, 0.5, 1.0]
                }
            },
            "Random Forest": {
                "estimator": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=25,
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                "param_grid": {
                    "n_estimators": [100],
                    "max_depth": [25]
                }
            }
        }

        if XGBOOST_AVAILABLE:
            models_config["XGBoost"] = {
                "estimator": XGBClassifier(
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=-1
                ),
                "param_grid": {
                    "n_estimators": [100],
                    "max_depth": [5],
                    "learning_rate": [0.15]
                }
            }

        return models_config

    def train_and_evaluate_all(
        self,
        sample_size: int = None
    ) -> Tuple[pd.DataFrame, Any, str]:
        """
        Train all models with 5-fold CV and GridSearchCV, compute all evaluation metrics,
        and select the best performing model.
        """
        X_train_df, X_test_df, y_train, y_test = self.prepare_data(sample_size=sample_size)

        logger.info("Transforming train and test sets with Feature Engineering Pipeline...")
        X_train_matrix = self.pipeline.transform(X_train_df)
        X_test_matrix = self.pipeline.transform(X_test_df)
        logger.info(f"Feature matrix shape: Train {X_train_matrix.shape} | Test {X_test_matrix.shape}")

        models_config = self.get_model_configurations()
        cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)

        benchmark_results = []
        cv_summary = []

        logger.info(f"=== Beginning Model Training & Hyperparameter Tuning ({len(models_config)} Algorithms) ===")

        for name, config in models_config.items():
            logger.info(f"\n--- Training {name} ---")
            grid_search = GridSearchCV(
                estimator=config["estimator"],
                param_grid=config["param_grid"],
                cv=cv_strategy,
                scoring="f1",
                n_jobs=-1,
                verbose=0
            )
            grid_search.fit(X_train_matrix, y_train)

            best_est = grid_search.best_estimator_
            self.trained_models[name] = best_est
            logger.info(f"Best Params for {name}: {grid_search.best_params_}")

            # Extract 5-Fold Cross Validation scores from best estimator in GridSearchCV
            b_idx = grid_search.best_index_
            cv_scores = [float(grid_search.cv_results_[f"split{i}_test_score"][b_idx]) for i in range(5)]
            cv_mean = float(grid_search.cv_results_["mean_test_score"][b_idx])
            cv_std = float(grid_search.cv_results_["std_test_score"][b_idx])
            logger.info(f"5-Fold CV F1 Score: {cv_mean:.4f} (+/- {cv_std:.4f})")

            # Predictions on unseen test set
            y_pred = best_est.predict(X_test_matrix)

            # Probability estimations for ROC-AUC
            if hasattr(best_est, "predict_proba"):
                y_prob = best_est.predict_proba(X_test_matrix)[:, 1]
            elif hasattr(best_est, "decision_function"):
                dec = best_est.decision_function(X_test_matrix)
                y_prob = (dec - dec.min()) / (dec.max() - dec.min() + 1e-9)
            else:
                y_prob = y_pred

            # Evaluation metrics
            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, zero_division=0))
            rec = float(recall_score(y_test, y_pred, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, zero_division=0))
            roc_auc = float(roc_auc_score(y_test, y_prob))

            logger.info(f"Test Metrics -> Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

            benchmark_results.append({
                "Model": name,
                "Accuracy": round(acc, 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4),
                "CV F1 Mean": round(cv_mean, 4),
                "CV F1 Std": round(cv_std, 4),
                "Best Hyperparameters": str(grid_search.best_params_)
            })

            cv_summary.append({
                "Model": name,
                "Fold 1": round(cv_scores[0], 4),
                "Fold 2": round(cv_scores[1], 4),
                "Fold 3": round(cv_scores[2], 4),
                "Fold 4": round(cv_scores[3], 4),
                "Fold 5": round(cv_scores[4], 4),
                "CV Mean": round(cv_mean, 4),
                "CV Std": round(cv_std, 4),
            })

            # Check if this model is the best
            if f1 > self.best_score:
                self.best_score = f1
                self.best_model_name = name
                self.best_model = best_est

        # Results compilation
        self.results_df = pd.DataFrame(benchmark_results).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
        cv_df = pd.DataFrame(cv_summary)

        # Save metrics to reports/metrics
        comp_csv_path = os.path.join(self.reports_dir, "metrics", "model_comparison.csv")
        cv_csv_path = os.path.join(self.reports_dir, "metrics", "cross_validation_scores.csv")
        self.results_df.to_csv(comp_csv_path, index=False)
        cv_df.to_csv(cv_csv_path, index=False)
        logger.info(f"Saved model comparison table to: {comp_csv_path}")
        logger.info(f"Saved cross-validation scores to: {cv_csv_path}")

        # Automatically serialize best model
        best_model_path = os.path.join(self.models_dir, "model.pkl")
        joblib.dump(self.best_model, best_model_path)
        logger.info(f"🏆 Best model '{self.best_model_name}' (F1: {self.best_score:.4f}) saved to: {best_model_path}")

        # Save metadata info
        summary_meta = {
            "best_model_name": self.best_model_name,
            "best_f1_score": round(self.best_score, 4),
            "test_accuracy": float(self.results_df[self.results_df["Model"] == self.best_model_name]["Accuracy"].values[0]),
            "models_evaluated": list(self.results_df["Model"].values)
        }
        with open(os.path.join(self.models_dir, "best_model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(summary_meta, f, indent=4)

        # Generate comparative visualization
        utils.plot_model_comparison_bar(
            self.results_df,
            output_path=os.path.join(self.reports_dir, "figures", "model_comparison.png")
        )

        return self.results_df, self.best_model, self.best_model_name


if __name__ == "__main__":
    trainer = ModelTrainer()
    results, best_model, best_name = trainer.train_and_evaluate_all()
    print("\n================ FINAL MODEL BENCHMARKING RESULTS ================")
    print(results.to_string(index=False))
    print(f"\nSelected Champion Model: {best_name}")
