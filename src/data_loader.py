"""
Data Loader Module for Fake News Detection System.

This module is responsible for Phase 1 Data Engineering:
- Loading raw datasets across multiple benchmark formats:
    1. Weibo Benchmark (train.pkl, val.pkl, test.pkl) - Chinese Social News
    2. FakeNewsNet GossipCop (gossipcop_fake.csv, gossipcop_real.csv) - English
    3. FakeNewsNet PolitiFact (politifact_fake.csv, politifact_real.csv) - English
    4. ISOT Benchmark (Fake.csv, True.csv) - English Full Articles
- Normalizing schema into: ['title', 'text', 'label', 'subject']
- Standardizing ground-truth binary labels:
    - Fake = 0
    - Real = 1
- Handling missing values and duplicates
- Shuffling the dataset with a reproducible seed
- Generating and persisting an exploratory summary report (data_summary.json)
- Saving the cleaned and processed dataset for downstream pipelines (processed_news.csv)
"""

import os
import sys
import re
import json
import pickle
import logging
import argparse
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

# Configure logging for transparent tracing
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_title_from_content(content: str) -> str:
    """
    Extract a clean headline/title from post content.
    Prioritizes brackets 【...】 or hashtags #...#, or falls back to first sentence.
    """
    s = str(content).strip()
    # Check for 【title】
    m = re.search(r"【(.*?)】", s)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # Check for #topic#
    m2 = re.search(r"#(.*?)#", s)
    if m2 and m2.group(1).strip():
        return m2.group(1).strip()
    # Fallback: first sentence
    parts = re.split(r"[。！？!?\n]", s)
    if parts and parts[0].strip():
        return parts[0].strip()[:70]
    return s[:70]


class NewsDataLoader:
    """
    Handles data ingestion, validation, cleaning, and preprocessing export
    for Fake News Detection benchmarks (Weibo, GossipCop, PolitiFact, ISOT).
    """

    def __init__(
        self,
        data_dir: str = "data",
        reports_dir: str = "reports/metrics",
        dataset_type: str = "auto"
    ):
        """
        Initialize directory paths and dataset type.

        Args:
            data_dir: Path to directory containing raw data files.
            reports_dir: Path to directory where metrics summary will be stored.
            dataset_type: 'auto', 'weibo', 'gossipcop', 'politifact', or 'isot'.
        """
        self.data_dir = data_dir
        self.reports_dir = reports_dir
        self.dataset_type = dataset_type.lower()
        self.processed_path = os.path.join(self.data_dir, "processed_news.csv")
        self.summary_report_path = os.path.join(self.reports_dir, "data_summary.json")

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    def detect_dataset_type(self) -> str:
        """Auto-detect available dataset in data directory."""
        weibo_train = os.path.join(self.data_dir, "train.pkl")
        gossipcop_fake = os.path.join(self.data_dir, "gossipcop_fake.csv")
        politifact_fake = os.path.join(self.data_dir, "politifact_fake.csv")
        isot_fake = os.path.join(self.data_dir, "Fake.csv")

        if os.path.exists(weibo_train):
            return "weibo"
        elif os.path.exists(gossipcop_fake):
            return "gossipcop"
        elif os.path.exists(politifact_fake):
            return "politifact"
        elif os.path.exists(isot_fake):
            return "isot"
        else:
            raise FileNotFoundError(
                f"No recognized dataset files found in {self.data_dir}. "
                "Expected either train.pkl/val.pkl/test.pkl (Weibo), "
                "gossipcop_*.csv, politifact_*.csv, or Fake.csv/True.csv."
            )

    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw dataset files from disk and normalize schema to:
        ['title', 'text', 'label', 'subject']
        """
        target_type = self.dataset_type
        if target_type == "auto":
            target_type = self.detect_dataset_type()

        logger.info(f"Target dataset type selected: '{target_type.upper()}'")

        if target_type == "weibo":
            return self._load_weibo()
        elif target_type == "gossipcop":
            return self._load_fakenewsnet("gossipcop")
        elif target_type == "politifact":
            return self._load_fakenewsnet("politifact")
        elif target_type == "isot":
            return self._load_isot()
        else:
            raise ValueError(f"Unsupported dataset type: {target_type}")

    def _load_weibo(self) -> pd.DataFrame:
        """Load Weibo Chinese benchmark from train.pkl, val.pkl, and test.pkl."""
        train_path = os.path.join(self.data_dir, "train.pkl")
        val_path = os.path.join(self.data_dir, "val.pkl")
        test_path = os.path.join(self.data_dir, "test.pkl")

        dfs = []
        for p, split_name in [(train_path, "train"), (val_path, "val"), (test_path, "test")]:
            if os.path.exists(p):
                with open(p, "rb") as f:
                    part_df = pickle.load(f)
                    part_df["split"] = split_name
                    dfs.append(part_df)
                    logger.info(f"Loaded Weibo {split_name} split: {len(part_df):,} samples.")

        if not dfs:
            raise FileNotFoundError(f"No Weibo pkl files found in {self.data_dir}")

        raw_df = pd.concat(dfs, ignore_index=True)

        # Standardize columns:
        # In original Weibo benchmark: label 0 = Real, label 1 = Fake (Rumor).
        # Standardize to project convention: Fake = 0, Real = 1.
        raw_df["label"] = raw_df["label"].apply(lambda x: 0 if int(x) == 1 else 1)
        raw_df["text"] = raw_df["content"].astype(str)
        raw_df["title"] = raw_df["content"].apply(extract_title_from_content)
        raw_df["subject"] = raw_df.get("category", "General")
        
        logger.info(f"Loaded total {len(raw_df):,} Weibo samples. Label distribution (0=Fake, 1=Real): "
                    f"{dict(raw_df['label'].value_counts())}")
        return raw_df

    def _load_fakenewsnet(self, name: str) -> pd.DataFrame:
        """Load FakeNewsNet dataset (gossipcop or politifact)."""
        fake_path = os.path.join(self.data_dir, f"{name}_fake.csv")
        real_path = os.path.join(self.data_dir, f"{name}_real.csv")

        if not os.path.exists(fake_path) or not os.path.exists(real_path):
            raise FileNotFoundError(f"Missing {fake_path} or {real_path}")

        fake_df = pd.read_csv(fake_path)
        real_df = pd.read_csv(real_path)

        fake_df["label"] = 0
        real_df["label"] = 1

        combined = pd.concat([fake_df, real_df], ignore_index=True)
        combined["text"] = combined["title"].astype(str)
        combined["subject"] = "Entertainment" if name == "gossipcop" else "Politics"
        logger.info(f"Loaded total {len(combined):,} FakeNewsNet {name} samples.")
        return combined

    def _load_isot(self) -> pd.DataFrame:
        """Load ISOT Fake.csv and True.csv."""
        fake_path = os.path.join(self.data_dir, "Fake.csv")
        true_path = os.path.join(self.data_dir, "True.csv")

        if not os.path.exists(fake_path) or not os.path.exists(true_path):
            raise FileNotFoundError(f"Missing {fake_path} or {true_path}")

        fake_df = pd.read_csv(fake_path)
        true_df = pd.read_csv(true_path)

        fake_df["label"] = 0
        true_df["label"] = 1

        combined = pd.concat([fake_df, true_df], ignore_index=True)
        if "subject" not in combined.columns:
            combined["subject"] = "News"
        return combined

    def process_and_clean(
        self,
        raw_df: pd.DataFrame,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Deduplicate, clean nulls, shuffle, and generate data summary report.
        """
        initial_total = len(raw_df)
        initial_fake = int((raw_df["label"] == 0).sum())
        initial_real = int((raw_df["label"] == 1).sum())

        initial_nulls = int(raw_df[["title", "text"]].isnull().sum().sum())

        # Drop missing title or text
        cleaned_df = raw_df.dropna(subset=["title", "text"]).copy()

        # Count duplicates
        duplicate_count = int(cleaned_df.duplicated(subset=["title", "text"]).sum())
        cleaned_df = cleaned_df.drop_duplicates(subset=["title", "text"]).copy()

        # Remove blank text
        cleaned_df = cleaned_df[
            (cleaned_df["text"].str.strip() != "") | (cleaned_df["title"].str.strip() != "")
        ].copy()

        # Shuffle
        shuffled_df = cleaned_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        final_total = len(shuffled_df)
        final_fake = int((shuffled_df["label"] == 0).sum())
        final_real = int((shuffled_df["label"] == 1).sum())

        # Standardize columns
        final_cols = ["title", "text", "label", "subject"]
        if "split" in shuffled_df.columns:
            final_cols.append("split")
        shuffled_df = shuffled_df[final_cols]

        summary_report: Dict[str, Any] = {
            "dataset_type": self.dataset_type,
            "initial_samples": {
                "fake_samples": initial_fake,
                "real_samples": initial_real,
                "total_samples": initial_total,
            },
            "data_quality": {
                "missing_values_detected": initial_nulls,
                "duplicate_count_removed": duplicate_count,
            },
            "cleaned_samples": {
                "final_fake_samples": final_fake,
                "final_real_samples": final_real,
                "final_total_samples": final_total,
                "fake_percentage": round((final_fake / max(1, final_total)) * 100, 2),
                "real_percentage": round((final_real / max(1, final_total)) * 100, 2),
            },
            "features": list(shuffled_df.columns),
        }

        # Persist summary report
        with open(self.summary_report_path, "w", encoding="utf-8") as f:
            json.dump(summary_report, f, indent=4, ensure_ascii=False)
        logger.info(f"Persisted data summary report to: {self.summary_report_path}")

        return shuffled_df, summary_report

    def save_processed_data(self, df: pd.DataFrame, sample_size: int = None) -> str:
        """Save cleaned dataset to CSV."""
        output_df = df if sample_size is None else df.iloc[:sample_size]
        output_df.to_csv(self.processed_path, index=False, encoding="utf-8")
        logger.info(f"Saved processed dataset ({len(output_df):,} rows) to: {self.processed_path}")
        return self.processed_path

    def run_pipeline(self, sample_size: int = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Run complete Phase 1 ingestion & normalization."""
        logger.info("=== Starting Phase 1: Data Engineering Pipeline ===")
        raw_df = self.load_raw_data()
        df, report = self.process_and_clean(raw_df)
        self.save_processed_data(df, sample_size=sample_size)
        logger.info("=== Phase 1 Pipeline Completed Successfully ===")
        return df, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and preprocess dataset.")
    parser.add_argument("--dataset", type=str, default="auto",
                        choices=["auto", "weibo", "gossipcop", "politifact", "isot"],
                        help="Dataset to load")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size limit")
    args = parser.parse_args()

    loader = NewsDataLoader(dataset_type=args.dataset)
    df, report = loader.run_pipeline(sample_size=args.sample)
    print("\n--- Data Engineering Summary Report ---")
    print(json.dumps(report, indent=4, ensure_ascii=False))
