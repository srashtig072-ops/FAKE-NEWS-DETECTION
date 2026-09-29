"""
NLP Preprocessing Module for Fake News Detection System.

This module is responsible for Phase 2 NLP Preprocessing:
- Multilingual capability: Supports both Chinese (via Jieba segmentation) and English (via NLTK)
- Lowercase conversion
- URL removal
- HTML tag removal
- Emoji removal
- Number removal
- Punctuation removal (ASCII and full-width CJK punctuation)
- Word segmentation & Tokenization
- Stopword removal (English + Chinese common stopwords)
- Lemmatization (English WordNet)
- Extra whitespace normalization

Compatible with both Chinese Weibo benchmarks and English datasets, fully integrated with Scikit-Learn pipelines.
"""

import re
import string
import logging
from typing import List, Union, Iterable, Set
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import jieba
    JIEBA_AVAILABLE = True
    # Suppress jieba verbose logging in normal runs
    jieba.setLogLevel(logging.INFO)
except ImportError:
    JIEBA_AVAILABLE = False

logger = logging.getLogger(__name__)

# Ensure NLTK resources are available
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
    LEMMATIZER.lemmatize("testing")
except Exception:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    LEMMATIZER = WordNetLemmatizer()

# Comprehensive Chinese stopword set (from classic Harbin Institute & Baidu stopword lists)
CHINESE_STOPWORDS: Set[str] = {
    "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很",
    "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这", "与", "及", "等", "其",
    "对", "从", "被", "由", "把", "向", "为", "以", "但", "而", "并", "因", "所以", "如果", "关于",
    "各", "只", "又", "更", "做", "让", "给", "但", "我们", "你们", "他们", "她们", "这个", "那个",
    "这些", "那些", "什么", "怎么", "怎样", "哪儿", "哪里", "为什么", "而且", "并且", "或者", "因为"
}

# Full-width Chinese and half-width English punctuation
FULL_PUNCTUATION = string.punctuation + "！？。，、“”‘’《》（）…—：；【】—～·"


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Production-grade text cleaning and NLP normalization pipeline.
    Follows standard Scikit-Learn transformer protocol (fit, transform).
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        remove_numbers: bool = True,
        remove_urls: bool = True,
        remove_html: bool = True,
        remove_emojis: bool = True,
        remove_punctuations: bool = True,
        custom_stopwords: Iterable[str] = None
    ):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.remove_numbers = remove_numbers
        self.remove_urls = remove_urls
        self.remove_html = remove_html
        self.remove_emojis = remove_emojis
        self.remove_punctuations = remove_punctuations

        self.stopwords_en = set(STOPWORDS)
        self.stopwords_cn = set(CHINESE_STOPWORDS)
        if custom_stopwords:
            self.stopwords_en.update(custom_stopwords)
            self.stopwords_cn.update(custom_stopwords)

        self.lemmatizer = LEMMATIZER

        # Precompiled regex patterns for maximum execution efficiency
        self.url_pattern = re.compile(r"https?://\S+|www\.\S+|t\.cn/\S+|sinaurl\.cn/\S+")
        self.html_pattern = re.compile(r"<.*?>")
        self.emoji_pattern = re.compile(
            r"\[.*?\]|[\U00010000-\U0010ffff]|[\u2600-\u27BF]|[\uD83C-\uDBFF\uDC00-\uDFFF]",
            flags=re.UNICODE
        )
        self.number_pattern = re.compile(r"\d+")
        self.punctuation_pattern = re.compile(f"[{re.escape(FULL_PUNCTUATION)}]")
        self.whitespace_pattern = re.compile(r"\s+")
        self.chinese_char_pattern = re.compile(r"[\u4e00-\u9fff]")

    def is_chinese(self, text: str) -> bool:
        """Check if text contains Chinese ideographs."""
        return bool(self.chinese_char_pattern.search(text))

    def clean_text(self, text: Union[str, float, None]) -> str:
        """
        Apply language-aware NLP cleaning pipeline to a single text string.
        """
        if text is None or pd.isna(text):
            return ""

        text = str(text)

        # 1. URL removal
        if self.remove_urls:
            text = self.url_pattern.sub(" ", text)

        # 2. HTML tag removal
        if self.remove_html:
            text = self.html_pattern.sub(" ", text)

        # 3. Emoji removal
        if self.remove_emojis:
            text = self.emoji_pattern.sub(" ", text)

        # 4. Number removal
        if self.remove_numbers:
            text = self.number_pattern.sub(" ", text)

        # Check language
        if self.is_chinese(text) and JIEBA_AVAILABLE:
            return self._clean_chinese(text)
        else:
            return self._clean_english(text)

    def _clean_chinese(self, text: str) -> str:
        """Clean and segment Chinese text using Jieba tokenization."""
        # Remove punctuation
        if self.remove_punctuations:
            text = self.punctuation_pattern.sub(" ", text)

        words = jieba.cut(text)
        tokens = []
        for w in words:
            w_str = w.strip()
            if not w_str:
                continue
            if self.remove_stopwords and w_str in self.stopwords_cn:
                continue
            if len(w_str) <= 1 and self.is_chinese(w_str):
                # Optionally filter single-char Chinese functional words
                continue
            if not re.match(r"^[\W_]+$", w_str):
                tokens.append(w_str)

        cleaned = " ".join(tokens)
        return self.whitespace_pattern.sub(" ", cleaned).strip()

    def _clean_english(self, text: str) -> str:
        """Clean and normalize English text using NLTK."""
        text = text.lower()

        if self.remove_punctuations:
            text = self.punctuation_pattern.sub(" ", text)

        tokens = text.split()
        processed_tokens = []
        for token in tokens:
            if self.remove_stopwords and token in self.stopwords_en:
                continue
            if len(token) <= 1:
                continue

            if self.lemmatize:
                lemmatized_token = self.lemmatizer.lemmatize(token, pos="v")
                lemmatized_token = self.lemmatizer.lemmatize(lemmatized_token, pos="n")
                processed_tokens.append(lemmatized_token)
            else:
                processed_tokens.append(token)

        cleaned = " ".join(processed_tokens)
        return self.whitespace_pattern.sub(" ", cleaned).strip()

    def fit(self, X, y=None):
        """No-op fit method for scikit-learn pipeline compatibility."""
        return self

    def transform(self, X: Union[pd.Series, List[str], Iterable[str]]) -> List[str]:
        """
        Transform a collection of texts.
        """
        if isinstance(X, pd.Series):
            return X.fillna("").apply(self.clean_text).tolist()
        elif isinstance(X, (list, tuple)):
            return [self.clean_text(item) for item in X]
        else:
            return [self.clean_text(str(item)) for item in X]


if __name__ == "__main__":
    import sys
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    en_news = "BREAKING: <b>NASA</b> lands rover on Mars at 10:45 AM! Check it out https://nasa.gov/mars Incredible!!"
    cn_news = "【国际物理学十大突破公布中国科研成果居榜首[good]】欧洲物理学会新闻网站“物理世界”昨天公布了2015年度国际物理学领域的十项重大突破。"

    preprocessor = TextPreprocessor()
    print("--- English Verification ---")
    print(preprocessor.clean_text(en_news))
    print("\n--- Chinese Verification ---")
    print(preprocessor.clean_text(cn_news))
