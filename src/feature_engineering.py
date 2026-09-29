"""
Feature Engineering Module for Fake News Detection System.

This module is responsible for Phase 4 Feature Engineering:
- Method 1: TF-IDF Feature Extraction
    - max_features = 15000
    - ngram_range = (1, 2)
    - min_df = 2
    - max_df = 0.95
- Method 2: Handcrafted Linguistic Features
    - title length
    - article length
    - number of capital words
    - punctuation count
    - digit ratio
- Scaled combination via Scikit-Learn FeatureUnion / ColumnTransformer
- Feature name resolution for model interpretability (SHAP & feature weights)
"""

import os
import sys
import string
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class HandcraftedLinguisticExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts 5 domain-specific linguistic & stylometric features:
    1. Title length (characters and words)
    2. Article body length (characters)
    3. Number of capitalized words (sensationalism indicator)
    4. Punctuation count (!, ?, quotes, etc.)
    5. Digit ratio (numerical density)
    """

    def __init__(self):
        self.feature_names = [
            "title_length",
            "article_length",
            "capital_words_count",
            "punctuation_count",
            "digit_ratio"
        ]

    def fit(self, X, y=None):
        """No-op fit method for scikit-learn compatibility."""
        return self

    def _extract_single(self, title: str, text: str) -> List[float]:
        """Compute the 5 stylometric features for a single sample."""
        title_str = str(title) if title is not None and not pd.isna(title) else ""
        text_str = str(text) if text is not None and not pd.isna(text) else ""
        combined = f"{title_str} {text_str}".strip()

        # 1. Title length in characters
        title_len = float(len(title_str))

        # 2. Article length in characters
        article_len = float(len(text_str))

        # 3. Number of capitalized words (words with >= 2 characters all uppercase)
        words = combined.split()
        cap_words_count = float(sum(1 for w in words if w.isupper() and len(w) > 1 and w.isalpha()))

        # 4. Punctuation count (supporting both ASCII and CJK full-width punctuation)
        cjk_punct = "！？。，、“”‘’《》（）…—：；【】—～·"
        punct_set = set(string.punctuation + cjk_punct)
        punct_count = float(sum(1 for ch in combined if ch in punct_set))

        # 5. Digit ratio
        total_chars = max(1, len(combined))
        digit_count = sum(1 for ch in combined if ch.isdigit())
        digit_ratio = float(digit_count / total_chars)

        return [title_len, article_len, cap_words_count, punct_count, digit_ratio]

    def transform(self, X: Union[pd.DataFrame, List[Tuple[str, str]], List[Dict[str, str]]]) -> np.ndarray:
        """
        Extract numerical features across all samples in X.

        Args:
            X: DataFrame with 'title' and 'text' columns, or list of (title, text) tuples.

        Returns:
            np.ndarray of shape (n_samples, 5).
        """
        features_list = []
        if isinstance(X, pd.DataFrame):
            titles = X["title"].tolist() if "title" in X.columns else [""] * len(X)
            texts = X["text"].tolist() if "text" in X.columns else X.iloc[:, 0].tolist()
            for t, b in zip(titles, texts):
                features_list.append(self._extract_single(t, b))
        elif isinstance(X, (list, tuple)):
            for item in X:
                if isinstance(item, (tuple, list)) and len(item) >= 2:
                    features_list.append(self._extract_single(item[0], item[1]))
                elif isinstance(item, dict):
                    features_list.append(self._extract_single(item.get("title", ""), item.get("text", "")))
                else:
                    features_list.append(self._extract_single("", str(item)))
        else:
            raise ValueError(f"Unsupported data type for LinguisticExtractor: {type(X)}")

        return np.array(features_list, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)


class NewsFeaturePipeline:
    """
    Complete Feature Engineering Pipeline combining:
    - 15,000 TF-IDF features (n-grams 1-2, min_df=2, max_df=0.95)
    - 5 Handcrafted Stylometric Features (MinMax Scaled for Naive Bayes & Linear compatibility)
    """

    def __init__(
        self,
        max_features: int = 15000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df

        self.preprocessor = TextPreprocessor()
        self.tfidf = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b"
        )
        self.linguistic_extractor = HandcraftedLinguisticExtractor()
        self.scaler = MinMaxScaler()
        self.all_feature_names: List[str] = []

    def fit(self, df: pd.DataFrame):
        """
        Fit TF-IDF vectorizer and numerical feature scaler on training dataset.

        Args:
            df: DataFrame containing 'title' and 'text'.
        """
        logger.info("Fitting NLP Preprocessing and TF-IDF Vectorizer...")
        raw_combined = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
        cleaned_texts = self.preprocessor.transform(raw_combined)

        n_samples = len(df)
        effective_min_df = self.min_df
        effective_max_df = self.max_df

        # Guard against small testing subsets where max_df < min_df
        if n_samples < 10:
            effective_min_df = 1
            effective_max_df = 1.0

        self.tfidf.set_params(min_df=effective_min_df, max_df=effective_max_df)
        self.tfidf.fit(cleaned_texts)
        tfidf_names = self.tfidf.get_feature_names_out().tolist()
        logger.info(f"Fitted TF-IDF with {len(tfidf_names):,} features.")

        logger.info("Fitting Handcrafted Linguistic Feature Scaler...")
        ling_features = self.linguistic_extractor.transform(df)
        self.scaler.fit(ling_features)

        self.all_feature_names = tfidf_names + self.linguistic_extractor.feature_names
        logger.info(f"Total Combined Features: {len(self.all_feature_names):,}")
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        """
        Transform raw news articles into combined sparse feature matrix.

        Args:
            df: DataFrame containing 'title' and 'text'.

        Returns:
            csr_matrix: Unified feature matrix (TF-IDF + Scaled Linguistic).
        """
        raw_combined = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()
        cleaned_texts = self.preprocessor.transform(raw_combined)
        tfidf_sparse = self.tfidf.transform(cleaned_texts)

        ling_features = self.linguistic_extractor.transform(df)
        ling_scaled = self.scaler.transform(ling_features)
        ling_sparse = csr_matrix(ling_scaled)

        combined_matrix = hstack([tfidf_sparse, ling_sparse]).tocsr()
        return combined_matrix

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        """Fit on df and return transformed feature matrix."""
        self.fit(df)
        return self.transform(df)

    def save_pipeline(self, output_path: str = "models/vectorizer.pkl"):
        """Serialize complete feature pipeline to disk using joblib."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(self, output_path)
        logger.info(f"Saved feature engineering pipeline to: {output_path}")

    @classmethod
    def load_pipeline(cls, model_path: str = "models/vectorizer.pkl") -> "NewsFeaturePipeline":
        """Deserialize feature engineering pipeline from disk."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Feature pipeline file not found at: {model_path}")
        pipeline = joblib.load(model_path)
        return pipeline


if __name__ == "__main__":
    sample_df = pd.DataFrame([
        {
            "title": "NASA Launches Breakthrough Mars Exploration Mission",
            "text": "Scientists at NASA confirmed the successful launch of the new Mars rover today."
        },
        {
            "title": "SHOCKING PROOF: Aliens Captured Alive In Secret Facility!!!",
            "text": "Leaked documents show government officials hiding 100% verified alien creatures!"
        }
    ])
    pipeline = NewsFeaturePipeline(max_features=50)
    features = pipeline.fit_transform(sample_df)
    print("Combined matrix shape:", features.shape)
    print("Features extracted:", len(pipeline.all_feature_names))
    print("Linguistic features:", pipeline.all_feature_names[-5:])
