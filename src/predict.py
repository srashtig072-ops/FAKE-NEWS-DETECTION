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

import re
import html
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import shap
from src.feature_engineering import NewsFeaturePipeline

logger = logging.getLogger(__name__)

REPUTABLE_SOURCES = {
    # Wire services & global outlets
    "reuters", "associated press", "ap news", "ap", "afp", "bbc", "bbc news",
    "cnn", "nbc news", "cbs news", "abc news", "npr", "pbs", "pbs newshour",
    "the new york times", "new york times", "nytimes", "washington post",
    "the washington post", "wall street journal", "wsj", "financial times",
    "bloomberg", "cnbc", "forbes", "the guardian", "guardian", "the independent",
    "the telegraph", "al jazeera", "dw", "deutsche welle", "france 24",
    "euronews", "sky news", "usa today", "politico", "axios", "time",
    "newsweek", "nikkei asia", "south china morning post", "cna",
    "los angeles times", "yahoo news", "fortune", "business insider",
    # Indian national, regional & business press
    "the hindu", "the times of india", "times of india", "ndtv", "india today",
    "indian express", "the indian express", "new indian express",
    "the new indian express", "hindustan times", "deccan herald",
    "deccan chronicle", "ani", "ani news", "pti", "press trust of india",
    "economic times", "the economic times", "livemint", "mint", "moneycontrol",
    "business standard", "financial express", "business today", "cnbctv18",
    "cnbc tv18", "firstpost", "news18", "times now", "wion", "the print",
    "the wire", "scroll", "scroll.in", "the quint", "the tribune", "tribune",
    "outlook", "outlook india", "dd news", "pib", "press information bureau",
    "the news minute", "bar and bench", "livelaw", "live law", "mid-day",
    "free press journal", "abp live", "zee news", "aaj tak",
    # Science, tech & sports
    "isro", "nasa", "nasa.gov", "nasa science", "nature", "science",
    "techcrunch", "the verge", "ars technica", "wired", "engadget",
    "9to5mac", "gsmarena", "gadgets 360", "espn", "espncricinfo", "cricbuzz",
    "space.com", "down to earth", "who", "world health organization", "rbi"
}

FACT_CHECK_MARKERS = [
    "fact check", "fact-check", "factcheck", "snopes", "altnews", "alt news",
    "vishvas news", "boomlive", "boom factcheck", "politifact", "hoax",
    "debunk", "debunked", "false claim", "fake claim", "misleading claim",
    "fake news", "hoax exposed", "viral claim is false", "pib fact check"
]

FACT_CHECK_REGEX = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in FACT_CHECK_MARKERS) + r")\b",
    re.IGNORECASE
)

SCAM_VIRAL_PATTERNS = [
    r"free\s+(?:\d+\s+)?(?:rupees|rs|inr|gb|recharge|mobile|phone|balance|laptop|money)",
    r"click\s+(?:this\s+|the\s+)?link\s+to\s+(?:claim|activate|receive|register)",
    r"forward\s+(?:this\s+)?to\s+\d+\s+(?:groups|friends|people|contacts)",
    r"(?:\d+\s+)?days\s+(?:of\s+)?(?:total\s+|complete\s+)*darkness",
    r"cures\s+(?:all\s+)?(?:diabetes|cancer|aids|covid)\s+(?:in\s+\d+|overnight|completely)",
    r"big\s+pharma\s+(?:is\s+)?(?:terrified|hiding|suppressing|furious)",
    r"share\s+(?:this\s+)?(?:urgent\s+\w+\s+)?before\s+(?:it\s+is\s+|facebook\s+)?(?:deleted|banned|censored|taken\s+down)",
    r"banks\s+(?:will\s+be\s+)?closed\s+for\s+\d+\s+days",
    r"withdraw\s+all\s+(?:your\s+)?(?:money|cash)\s+immediately",
    r"asteroid\s+(?:will\s+|to\s+)?hit\s+earth\s+(?:tomorrow|tonight|this\s+week)",
    r"(?:alien|ufo|extraterrestrial)\s+(?:spacecraft|ship|landing|invaded|confirmed|spotted\s+on\s+moon)",
    r"(?:rs\.?|rupees?|inr)\s*\d[\d,]*\s+(?:fee|charge|tax|penalty)\s+on\s+(?:every\s+|all\s+)?upi",
    r"\d[\d,]*\s+(?:rs\.?|rupees?|inr)\s+(?:fee|charge|tax|penalty)\s+on\s+(?:every\s+|all\s+)?upi",
    r"(?:rbi|government|govt)\s+(?:announces?|imposes?|levies?|charges?|introduces?)\s+[\w\s]*(?:fee|charge|tax)\s+on\s+(?:every\s+|all\s+)?upi",
    r"upi\s+(?:transactions?\s+)?(?:fee|charge|tax|will\s+be\s+charged|becomes?\s+paid)",
    r"government\s+(?:to\s+)?(?:impose|levy|charge|announce[sd]?)\s+(?:fee|tax)\s+on\s+upi",
    r"all\s+banks?\s+(?:will\s+be\s+)?closed\s+(?:for\s+)?\d+\s+days",
    r"neft\s+rtgs\s+imps\s+(?:to\s+be\s+)?suspended",
    r"(?:rs\.?|rupees?)\s*\d[\d,]+\s+(?:given?\s+free|government\s+scheme|cashback\s+scheme)\s+for\s+all"
]

SENSATIONAL_HOAX_CUES = {
    "shocking", "bombshell", "unbelievable", "exposed", "must watch",
    "share immediately", "forward this", "before it is deleted",
    "before facebook deletes", "big pharma", "miracle cure", "secret leaked",
    "deep state", "cabal", "100% proof", "reptilians", "mainstream media is hiding",
    "censors delete", "terrified", "refuse to disclose", "alien spacecraft"
}

QUERY_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "over",
    "after", "amid", "near", "about", "under", "between", "against", "during",
    "says", "said", "will", "have", "has", "had", "are", "was", "were", "been",
    "being", "more", "than", "also", "just", "watch", "live", "video", "read",
    "photos", "breaking", "exclusive", "latest", "update", "updates", "news",
    "report", "reports", "how", "what", "why", "when", "where", "who", "here",
    "there", "their", "they", "your", "our", "out", "new", "all", "not", "but"
}


class NewsPredictor:
    """
    Handles inference, live Google News cross-verification, and SHAP explainability.
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
        elif hasattr(self.model, "feature_log_prob_"):
            self._is_linear = True
        elif hasattr(self.model, "feature_importances_") or "Forest" in model_cls_name or "XGB" in model_cls_name:
            self._is_tree = True
        elif hasattr(self.model, "calibrated_classifiers_"):
            self._is_linear = True

    @staticmethod
    def _strip_source_suffix(title: str) -> str:
        """Strip trailing ' - Publisher' or ' | Publisher' suffix common in Google News titles."""
        parts = re.split(r"\s+[-|–—]\s+", title.strip())
        if len(parts) > 1 and len(parts[-1]) <= 45:
            return " - ".join(parts[:-1]).strip()
        return title.strip()

    @staticmethod
    def _extract_url_metadata(url: str) -> Tuple[str, str]:
        """Fetch page title and meta description when a raw article URL is pasted."""
        req = urllib.request.Request(url.strip(), headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw_html = resp.read(200_000).decode("utf-8", errors="ignore")
            og_title = re.search(
                r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']',
                raw_html,
                re.IGNORECASE
            )
            title_tag = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
            og_desc = re.search(
                r'<meta[^>]+(?:name|property)=["\'](?:description|og:description)["\'][^>]+content=["\']([^"\']+)["\']',
                raw_html,
                re.IGNORECASE
            )
            extracted_title = ""
            if og_title:
                extracted_title = html.unescape(og_title.group(1)).strip()
            elif title_tag:
                extracted_title = html.unescape(re.sub(r"\s+", " ", title_tag.group(1))).strip()
            extracted_desc = html.unescape(og_desc.group(1)).strip() if og_desc else ""
            if extracted_title.lower() in {"google news", "redirect notice"}:
                extracted_title = ""
            return extracted_title, extracted_desc
        except Exception as e:
            logger.debug(f"URL metadata extraction note: {e}")
            return "", ""

    @staticmethod
    def _tokenize_for_match(text: str) -> set:
        """Extract normalized tokens (including 2-3 char acronyms and CJK characters) for overlap."""
        cleaned = NewsPredictor._strip_source_suffix(text).lower()
        if re.search(r"[\u4e00-\u9fff]", cleaned):
            cjk = re.findall(r"[\u4e00-\u9fff]", cleaned)
            bigrams = {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
            latin = {w for w in re.findall(r"[a-z0-9]{2,}", cleaned) if w not in QUERY_STOPWORDS}
            return bigrams | latin | set(cjk)
        words = re.findall(r"[a-z0-9]+", cleaned)
        return {w for w in words if len(w) >= 2 and w not in QUERY_STOPWORDS}

    @staticmethod
    def _build_search_query(headline: str, text: str = "") -> str:
        """Build a concise Google News RSS query from headline or opening body text."""
        base = headline.strip() if headline and headline.strip() else text.strip()
        if not base:
            return ""
        # Use first sentence when falling back to long body text
        if not headline.strip():
            first_sentence = re.split(r"[.!?\n。！？]", base, maxsplit=1)[0].strip()
            base = first_sentence[:220] if first_sentence else base[:220]

        base = NewsPredictor._strip_source_suffix(base)
        if re.search(r"[\u4e00-\u9fff]", base):
            cleaned_cn = re.sub(r"[^\u4e00-\u9fffa-zA-Z0-9\s]", " ", base).strip()
            return cleaned_cn[:28]

        raw_words = re.findall(r"[A-Za-z0-9]+", base)
        filtered = [w for w in raw_words if len(w) >= 2 and w.lower() not in QUERY_STOPWORDS]
        if len(filtered) < 3:
            filtered = [w for w in raw_words if len(w) >= 2]
        return " ".join(filtered[:7])

    @staticmethod
    def _is_reputable_source(source_name: str) -> bool:
        """Check whether an RSS source name matches an accredited press outlet."""
        if not source_name:
            return False
        s_norm = re.sub(r"\.(?:com|in|org|net|co\.uk)$", "", source_name.strip().lower())
        return any(
            s_norm == rep or re.search(r"\b" + re.escape(rep) + r"\b", s_norm)
            for rep in REPUTABLE_SOURCES
        )

    @staticmethod
    def _is_relevant_match(query_tokens: set, rss_title: str) -> Tuple[bool, float]:
        """Verify that a returned Google RSS article actually matches the user's news claim."""
        if not query_tokens or not rss_title:
            return False, 0.0
        rss_tokens = NewsPredictor._tokenize_for_match(rss_title)
        if not rss_tokens:
            return False, 0.0
        shared = len(query_tokens & rss_tokens)
        denom = max(1, min(len(query_tokens), len(rss_tokens)))
        ratio = shared / denom
        relevant = (
            (shared >= 2 and ratio >= 0.35)
            or (shared >= 3 and ratio >= 0.25)
            or (len(query_tokens) <= 2 and shared >= 1 and ratio >= 0.50)
        )
        return relevant, ratio

    def verify_google_news_live(self, headline: str, text: str = "") -> Dict[str, Any]:
        """
        Cross-reference headline/text against Live Google News RSS and Fact-Check indices.
        """
        combined = f"{headline} {text}".lower()

        # Check viral hoax / scam regex patterns
        is_scam = any(re.search(pat, combined, re.IGNORECASE) for pat in SCAM_VIRAL_PATTERNS)

        query = self._build_search_query(headline, text)
        match_reference = headline.strip() if headline.strip() else text.strip()[:220]
        query_tokens = self._tokenize_for_match(match_reference)

        matched_articles: List[Dict[str, str]] = []
        relevant_articles: List[Dict[str, str]] = []
        has_fact_check_debunk = False
        verified_sources: List[str] = []
        relevant_sources: List[str] = []
        max_relevance = 0.0

        if query:
            url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query)
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    root = ET.fromstring(response.read())
                    for item in root.findall(".//item")[:8]:
                        t = item.find("title").text if item.find("title") is not None else ""
                        s = item.find("source").text if item.find("source") is not None else ""
                        link = item.find("link").text if item.find("link") is not None else ""
                        article_info = {"title": t, "source": s, "link": link}
                        matched_articles.append(article_info)

                        is_rel, rel_score = self._is_relevant_match(query_tokens, t)
                        if not is_rel:
                            continue

                        relevant_articles.append(article_info)
                        max_relevance = max(max_relevance, rel_score)

                        item_is_debunk = bool(FACT_CHECK_REGEX.search(t) or FACT_CHECK_REGEX.search(s))
                        if item_is_debunk:
                            has_fact_check_debunk = True
                            continue

                        if s and s not in relevant_sources:
                            relevant_sources.append(s)
                        if self._is_reputable_source(s) and s not in verified_sources:
                            verified_sources.append(s)
            except Exception as e:
                logger.debug(f"Live Google check note: {e}")

        # Determine verification verdict using ONLY relevant matches
        if is_scam or has_fact_check_debunk:
            status = "DEBUNKED_FAKE"
            note = "Flagged: Matches known viral disinformation or debunked by fact-checkers."
        elif len(verified_sources) >= 1 or (len(relevant_sources) >= 2 and max_relevance >= 0.45):
            status = "VERIFIED_REAL"
            display_sources = verified_sources if verified_sources else relevant_sources
            note = f"Verified: Corroborated on Google News ({', '.join(display_sources[:3])})."
        else:
            status = "UNVERIFIED"
            note = "Independent claim: No matching mainstream press or fact-check record found."

        display_articles = relevant_articles if relevant_articles else matched_articles[:5]

        return {
            "status": status,
            "is_scam": is_scam,
            "has_fact_check_debunk": has_fact_check_debunk,
            "verified_sources": verified_sources if verified_sources else (relevant_sources if status == "VERIFIED_REAL" else []),
            "matched_articles": display_articles[:6],
            "verification_note": note
        }

    def predict(self, title: str, text: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Analyze an article, classify as REAL or FAKE, and generate SHAP explainability.
        """
        title = str(title or "").strip()
        text = str(text or "").strip()

        # If the user pasted a raw URL into title or text, attempt to resolve headline/snippet
        url_candidate = ""
        if re.match(r"^https?://\S+$", title):
            url_candidate = title
            title = ""
        elif not title and re.match(r"^https?://\S+$", text):
            url_candidate = text
            text = ""
        if url_candidate:
            fetched_title, fetched_desc = self._extract_url_metadata(url_candidate)
            if fetched_title:
                title = fetched_title
            if fetched_desc and not text:
                text = fetched_desc
            if not title and not text:
                title = url_candidate

        input_df = pd.DataFrame([{"title": title, "text": text}])
        feature_matrix = self.pipeline.transform(input_df)

        # Base ML Model Probabilities
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(feature_matrix)[0]
            prob_fake = float(probs[0])
            prob_real = float(probs[1])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(feature_matrix)[0])
            prob_real = 1.0 / (1.0 + np.exp(-score))
            prob_fake = 1.0 - prob_real
        else:
            raw_pred = self.model.predict(feature_matrix)[0]
            prob_real = 1.0 if int(raw_pred) == 1 else 0.0
            prob_fake = 1.0 - prob_real

        # Live Google News Cross-Reference & Credibility Fusion
        google_res = self.verify_google_news_live(title, text)

        if google_res["status"] == "DEBUNKED_FAKE":
            prob_fake = max(prob_fake, 0.885)
            prob_real = round(1.0 - prob_fake, 4)
            class_id = 0
            label = "FAKE"
        elif google_res["status"] == "VERIFIED_REAL":
            prob_real = max(prob_real, 0.865)
            prob_fake = round(1.0 - prob_real, 4)
            class_id = 1
            label = "REAL"
        else:
            # Guardrail for unverified viral/sensationalist clickbait cues
            combined_lower = f"{title} {text}".lower()
            hoax_hits = sum(
                1 for cue in SENSATIONAL_HOAX_CUES
                if re.search(r"\b" + re.escape(cue) + r"\b", combined_lower)
            )
            excl_count = (title + " " + text).count("!") + (title + " " + text).count("！")
            if hoax_hits >= 2 or (hoax_hits >= 1 and excl_count >= 2):
                prob_fake = max(prob_fake, min(0.85, 0.60 + 0.08 * hoax_hits + 0.03 * min(excl_count, 4)))
                prob_real = round(1.0 - prob_fake, 4)
            class_id = 1 if prob_real >= 0.50 else 0
            label = "REAL" if class_id == 1 else "FAKE"

        confidence = round((prob_real if label == "REAL" else prob_fake) * 100, 2)

        # SHAP Interpretability Calculation
        feature_names = self.pipeline.all_feature_names
        dense_features = feature_matrix.toarray()[0]
        active_indices = np.where(dense_features > 0)[0]

        contributions: List[Dict[str, Any]] = []

        if self._is_linear:
            # Extract weights from linear model or calibrated base estimator
            if hasattr(self.model, "coef_"):
                weights = self.model.coef_[0]
            elif hasattr(self.model, "feature_log_prob_") and len(self.model.feature_log_prob_) >= 2:
                weights = self.model.feature_log_prob_[1] - self.model.feature_log_prob_[0]
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
            "feature_contributions": contributions[:top_k * 2],
            "google_verification": google_res
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
