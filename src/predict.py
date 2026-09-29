"""
Prediction and Explainable AI (SHAP) Module for Fake News Detection System.

This module implements Phase 6 Explainable AI:
- Inference on new/unseen news articles
- Prediction of label (REAL vs FAKE) with confidence percentage
- SHAP (SHapley Additive exPlanations) interpretability
- Extraction of top positive keywords (evidence for REAL)
- Extraction of top negative keywords (evidence for FAKE)
- Generation of local feature importance explanations
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import joblib

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shap
from src.feature_engineering import NewsFeaturePipeline

logger = logging.getLogger(__name__)


class NewsPredictor:
    """
    Handles inference and SHAP explainability for single or batch news articles.
    """

    def __init__(
        self,
        model_path: str = "models/model.pkl",
        vectorizer_path: str = "models/vectorizer.pkl",
        label_encoder_path: str = "models/label_encoder.pkl"
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.label_encoder_path = label_encoder_path

        self.model = None
        self.pipeline: NewsFeaturePipeline = None
        self.label_encoder = None
        self.explainer = None
        self._is_linear = False
        self._is_tree = False

        self._load_artifacts()

    def _load_artifacts(self):
        """Load model, vectorizer, and configure SHAP explainer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}. Please run train.py first.")
        if not os.path.exists(self.vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at: {self.vectorizer_path}")

        self.model = joblib.load(self.model_path)
        self.pipeline = NewsFeaturePipeline.load_pipeline(self.vectorizer_path)

        if os.path.exists(self.label_encoder_path):
            self.label_encoder = joblib.load(self.label_encoder_path)

        # Determine model type for SHAP
        model_cls_name = self.model.__class__.__name__
        if hasattr(self.model, "coef_"):
            self._is_linear = True
        elif hasattr(self.model, "feature_importances_") or "Forest" in model_cls_name or "XGB" in model_cls_name:
            self._is_tree = True
        elif hasattr(self.model, "calibrated_classifiers_"):
            # CalibratedClassifierCV wrapping LinearSVC
            self._is_linear = True

    def predict(self, title: str, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze an article, classify as REAL or FAKE, and generate SHAP explainability.

        Args:
            title: Headline of the news article.
            text: Body text of the news article.
            top_k: Number of positive and negative explanatory features to return.

        Returns:
            Dict containing:
                - label: 'REAL' or 'FAKE'
                - class_id: 1 or 0
                - confidence: float (0.0 to 100.0)
                - prob_fake: float (0.0 to 1.0)
                - prob_real: float (0.0 to 1.0)
                - top_positive_words: List[Tuple[feature_name, shap_value]] (supporting REAL)
                - top_negative_words: List[Tuple[feature_name, shap_value]] (supporting FAKE)
                - feature_contributions: List[Dict[str, Any]]
        """
        input_df = pd.DataFrame([{"title": title, "text": text}])
        feature_matrix = self.pipeline.transform(input_df)

        # Predict class
        raw_pred = self.model.predict(feature_matrix)[0]
        class_id = int(raw_pred)

        # Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_matrix)[0]
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(feature_matrix)[0])
            prob_real = 1.0 / (1.0 + np.exp(-score))
            prob_fake = 1.0 - prob_real
        else:
            prob_real = 1.0 if class_id == 1 else 0.0
            prob_fake = 1.0 - prob_real

        label = "REAL" if class_id == 1 else "FAKE"
        confidence = round((prob_real if class_id == 1 else prob_fake) * 100, 2)

        # SHAP Interpretability Calculation
        feature_names = self.pipeline.all_feature_names
        dense_features = feature_matrix.toarray()[0]
        active_indices = np.where(dense_features > 0)[0]

        contributions: List[Dict[str, Any]] = []

        if self._is_linear:
            # Extract weights from linear model or calibrated base estimator
            if hasattr(self.model, "coef_"):
                weights = self.model.coef_[0]
            elif hasattr(self.model, "calibrated_classifiers_"):
                # Average weights across calibrated folds
                weights = np.mean([clf.estimator.coef_[0] for clf in self.model.calibrated_classifiers_], axis=0)
            else:
                weights = np.zeros(len(feature_names))

            for idx in active_indices:
                feat_name = feature_names[idx]
                feat_val = dense_features[idx]
                impact = float(weights[idx] * feat_val)
                contributions.append({
                    "feature": feat_name,
                    "value": round(float(feat_val), 4),
                    "impact": round(impact, 4),
                    "direction": "REAL" if impact > 0 else "FAKE"
                })
        else:
            # Tree / Generic Model Attribution
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                for idx in active_indices:
                    feat_name = feature_names[idx]
                    feat_val = dense_features[idx]
                    impact = float(importances[idx] * feat_val)
                    direction = "REAL" if class_id == 1 else "FAKE"
                    contributions.append({
                        "feature": feat_name,
                        "value": round(float(feat_val), 4),
                        "impact": round(impact, 4),
                        "direction": direction
                    })

        # Sort contributions by absolute impact
        contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)

        # Separate positive (supporting REAL) and negative (supporting FAKE) features
        pos_words = [(c["feature"], c["impact"]) for c in contributions if c["impact"] > 0][:top_k]
        neg_words = [(c["feature"], abs(c["impact"])) for c in contributions if c["impact"] < 0][:top_k]

        return {
            "label": label,
            "class_id": class_id,
            "confidence": confidence,
            "prob_fake": round(prob_fake, 4),
            "prob_real": round(prob_real, 4),
            "top_positive_words": pos_words,
            "top_negative_words": neg_words,
            "feature_contributions": contributions[:top_k * 2]
        }


if __name__ == "__main__":
    sample_title = "Trump Lashes Out in Midnight Tweet Storm Targeting Intelligence Agencies"
    sample_text = (
        "In a series of late-night posts on Twitter, Donald Trump fiercely criticized "
        "the intelligence community, claiming the investigation was a total witch hunt and fake news!"
    )
    try:
        predictor = NewsPredictor()
        res = predictor.predict(sample_title, sample_text)
        print("Prediction Result:")
        print(f"Label: {res['label']} ({res['confidence']}%)")
        print(f"Probabilities -> Fake: {res['prob_fake']:.2%}, Real: {res['prob_real']:.2%}")
        print("Top Fake-Indicative Features:", res["top_negative_words"][:5])
        print("Top Real-Indicative Features:", res["top_positive_words"][:5])
    except Exception as e:
        print(f"Prediction note: {e}")
