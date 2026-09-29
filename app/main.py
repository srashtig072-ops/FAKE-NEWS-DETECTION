"""
Streamlit Web Application for Fake News Detection System.

This application provides an interactive, production-ready interface:
1. Live News Detective (Single-Article Real-Time Fact Checker with XAI & Highlighting)
2. Batch CSV Processing & Analysis
3. Model Benchmark & Evaluation Dashboard
4. Exploratory Data Analysis (EDA) Interactive Gallery
5. Academic Paper Alignment & Architecture Walkthrough
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np

# Ensure project root is available in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="TruthLens | AI Fake News Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic & Modern CSS
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F766E 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .main-header h1 {
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.4rem;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #FFFFFF, #E2E8F0, #5EEAD4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 0;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.3rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 20px -8px rgba(0, 0, 0, 0.5);
    }

    .metric-card .val {
        font-size: 2rem;
        font-weight: 800;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card .lbl {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-top: 0.2rem;
    }

    .badge-real {
        background: linear-gradient(135deg, #059669, #10B981);
        color: white;
        padding: 0.7rem 1.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.4);
    }

    .badge-fake {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white;
        padding: 0.7rem 1.6rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.25rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 4px 14px rgba(239, 68, 68, 0.4);
    }

    .highlight-real {
        background-color: rgba(16, 185, 129, 0.22);
        color: #10B981;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        border-bottom: 2px solid #10B981;
    }

    .highlight-fake {
        background-color: rgba(239, 68, 68, 0.22);
        color: #EF4444;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        border-bottom: 2px solid #EF4444;
    }

    .sidebar-info {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #94A3B8;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Lazy-load inference pipeline with caching
@st.cache_resource(show_spinner=False)
def get_predictor():
    """Load cached NewsPredictor instance."""
    from src.predict import NewsPredictor
    model_file = os.path.join(PROJECT_ROOT, "models", "model.pkl")
    vec_file = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")
    enc_file = os.path.join(PROJECT_ROOT, "models", "label_encoder.pkl")

    if not os.path.exists(model_file) or not os.path.exists(vec_file):
        return None
    try:
        return NewsPredictor(model_path=model_file, vectorizer_path=vec_file, label_encoder_path=enc_file)
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        return None


def render_header():
    """Render top page hero header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ TruthLens AI — Fake News Detection & XAI</h1>
            <p>Production Machine Learning System with TF-IDF, Stylometric Feature Engineering & Explainable AI (SHAP)</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_highlighted_text(text: str, pos_words: List[str], neg_words: List[str]) -> str:
    """Render text with color-coded token highlights for explainability."""
    if not text:
        return ""
    pos_set = {w.lower() for w in pos_words}
    neg_set = {w.lower() for w in neg_words}

    tokens = re.split(r"(\s+|[.,!?;:\"()\[\]])", text)
    highlighted = []

    for token in tokens:
        clean = re.sub(r"[^\w]", "", token).lower()
        if clean and clean in pos_set:
            highlighted.append(f'<span class="highlight-real" title="Evidence for Real News">{token}</span>')
        elif clean and clean in neg_set:
            highlighted.append(f'<span class="highlight-fake" title="Evidence for Fake News">{token}</span>')
        else:
            highlighted.append(token)

    return "".join(highlighted)


# Pre-populated test samples for one-click testing (Bilingual)
SAMPLE_REAL_EN = {
    "title": "U.S. and European Allies Hold High-Level Diplomatic Talks on Global Trade Policy",
    "text": (
        "WASHINGTON (Reuters) - Senior diplomatic and economic representatives from the United States and the European Union "
        "convened on Wednesday to deliberate upon cross-border tariffs and supply chain stabilization initiatives. "
        "Officials confirmed in a joint press briefing that working groups have been established to formalize the regulatory framework."
    )
}

SAMPLE_FAKE_EN = {
    "title": "SHOCKING BOMBSHELL: Secret Deep-State Cabal EXPOSED in Midnight Leak! MUST WATCH VIDEO!",
    "text": (
        "You won't believe what mainstream media is hiding from you! A top-secret leaked memo confirms that corrupt political elites "
        "have been staging global hoaxes to steal taxpayer money! Share this viral video immediately before censors delete it forever! "
        "The truth is finally out and they are terrified!"
    )
}

SAMPLE_REAL_CN = {
    "title": "三星折叠屏原型机曝光：双屏设计/非柔性屏",
    "text": "网友晒出三星的ProjcetV可折叠手机，该机型号为SM-G929，采用了双屏幕设计，仅用物理方式连接起来，非柔性屏折叠。爆料者强调，该机为测试机，并不会推出市场。"
}

SAMPLE_FAKE_CN = {
    "title": "震惊！日本人发明盆景猫！把小猫塞在瓶子里！",
    "text": "这是日本人所发明的一种宠物养殖技术，叫盆景猫！他们给小猫喂食一种促进骨质软化的化学药物，然后强塞进瓶子里，等小猫长大，身体胀满瓶子！转发扩散让更多人知道真相！"
}


def page_live_detect(predictor):
    """Page 1: Real-time Live Fact-Checking with XAI."""
    st.subheader("🔍 Real-Time Article Fact-Checking & Explainability")
    st.write("Enter the headline and article text below, or select a pre-loaded test article to analyze credibility.")

    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        if st.button("📰 Real News (EN)", use_container_width=True):
            st.session_state["input_title"] = SAMPLE_REAL_EN["title"]
            st.session_state["input_text"] = SAMPLE_REAL_EN["text"]
    with col_btn2:
        if st.button("🚨 Fake News (EN)", use_container_width=True):
            st.session_state["input_title"] = SAMPLE_FAKE_EN["title"]
            st.session_state["input_text"] = SAMPLE_FAKE_EN["text"]
    with col_btn3:
        if st.button("📰 Real News (Weibo)", use_container_width=True):
            st.session_state["input_title"] = SAMPLE_REAL_CN["title"]
            st.session_state["input_text"] = SAMPLE_REAL_CN["text"]
    with col_btn4:
        if st.button("🚨 Fake Rumor (Weibo)", use_container_width=True):
            st.session_state["input_title"] = SAMPLE_FAKE_CN["title"]
            st.session_state["input_text"] = SAMPLE_FAKE_CN["text"]

    curr_title = st.session_state.get("input_title", "")
    curr_text = st.session_state.get("input_text", "")

    with st.form("predict_form"):
        title_input = st.text_input(
            "Article Headline / Title:",
            value=curr_title,
            placeholder="e.g. Federal Reserve announces interest rate policy update..."
        )
        text_input = st.text_area(
            "Article Content / Body Text:",
            value=curr_text,
            height=180,
            placeholder="Paste article body here..."
        )
        submit_btn = st.form_submit_button("⚡ Analyze News Authenticity", use_container_width=True, type="primary")

    if submit_btn:
        if not title_input.strip() and not text_input.strip():
            st.warning("⚠️ Please provide at least a title or text to analyze.")
            return

        if predictor is None:
            st.error("Model artifacts are still loading or training. Please check Model Training status.")
            return

        with st.spinner("🧠 Performing NLP extraction, stylometric analysis, and SHAP attribution..."):
            t0 = time.time()
            res = predictor.predict(title_input, text_input, top_k=10)
            latency = time.time() - t0

        st.markdown("---")
        st.subheader("🎯 Credibility Verdict")

        col_verdict, col_conf, col_speed = st.columns([1.5, 2, 1])

        with col_verdict:
            if res["label"] == "REAL":
                st.markdown(
                    f'<div class="badge-real"><span>✅</span> VERIFIED REAL NEWS</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="badge-fake"><span>🚨</span> DETECTED FAKE NEWS</div>',
                    unsafe_allow_html=True
                )

        with col_conf:
            st.markdown(f"**Confidence:** `{res['confidence']}%`")
            st.progress(res["confidence"] / 100.0)
            st.caption(f"Real Probability: `{res['prob_real']:.2%}` | Fake Probability: `{res['prob_fake']:.2%}`")

        with col_speed:
            st.metric(label="Inference Latency", value=f"{latency * 1000:.1f} ms")

        # Stylometric Markers
        st.markdown("---")
        st.subheader("📏 Stylometric & Linguistic Marker Profile")
        title_len = len(title_input)
        article_len = len(text_input)
        words = (title_input + " " + text_input).split()
        cap_words = sum(1 for w in words if w.isupper() and len(w) > 1 and w.isalpha())
        punct_count = sum(1 for c in (title_input + " " + text_input) if c in "!?,.:;\"'")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Word Count", f"{len(words):,}")
        with c2:
            st.metric("Capitalized Words", f"{cap_words}", delta="High Sensationalism" if cap_words > 4 else "Normal", delta_color="inverse")
        with c3:
            st.metric("Punctuation Count", f"{punct_count}", delta="Elevated" if punct_count > 15 else "Normal", delta_color="inverse")
        with c4:
            st.metric("Headline Length", f"{title_len} chars")

        # Explainable AI (SHAP) Keyword Attribution
        st.markdown("---")
        st.subheader("🔬 Explainable AI (XAI) — Word & Token Attribution")
        st.write("TruthLens reveals **why** the AI reached this classification by attributing prediction importance to specific words.")

        pos_words = [item[0] for item in res["top_positive_words"]]
        neg_words = [item[0] for item in res["top_negative_words"]]

        tab_highlight, tab_features = st.tabs(["✨ Color-Coded Article Highlights", "📊 Top Contributing Keywords"])

        with tab_highlight:
            st.caption("🟢 Green highlights indicate evidence supporting **Real Journalism**. 🔴 Red highlights indicate evidence supporting **Sensationalist / Fake Information**.")
            combined_raw = f"**{title_input}**\n\n{text_input}"
            rendered_html = render_highlighted_text(combined_raw, pos_words, neg_words)
            st.markdown(
                f'<div style="background: rgba(30,41,59,0.5); padding: 1.5rem; border-radius: 12px; line-height: 1.8; border: 1px solid rgba(255,255,255,0.08);">{rendered_html}</div>',
                unsafe_allow_html=True
            )

        with tab_features:
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("##### 🟢 Top Evidence for REAL News")
                if res["top_positive_words"]:
                    pos_df = pd.DataFrame(res["top_positive_words"], columns=["Keyword", "Impact Weight"]).round(4)
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strong positive indicators detected.")

            with col_neg:
                st.markdown("##### 🔴 Top Evidence for FAKE News")
                if res["top_negative_words"]:
                    neg_df = pd.DataFrame(res["top_negative_words"], columns=["Keyword", "Impact Weight"]).round(4)
                    st.dataframe(neg_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strong negative indicators detected.")


def page_batch_predict(predictor):
    """Page 2: Batch CSV upload and bulk inference."""
    st.subheader("📁 Batch News Verification (CSV File)")
    st.write("Upload a CSV file with `title` and `text` columns to perform automated bulk authenticity analysis.")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df):,}** articles. Preview of uploaded data:")
            st.dataframe(df.head(5), use_container_width=True)

            if "title" not in df.columns and "text" not in df.columns:
                st.error("Uploaded CSV must contain at least 'title' or 'text' column.")
                return

            if "title" not in df.columns:
                df["title"] = ""
            if "text" not in df.columns:
                df["text"] = ""

            if st.button("🚀 Process Entire Batch", type="primary"):
                if predictor is None:
                    st.error("Predictor model not available.")
                    return

                with st.spinner("Analyzing all articles..."):
                    labels = []
                    confidences = []
                    prob_reals = []
                    prob_fakes = []

                    progress_bar = st.progress(0)
                    total = len(df)

                    for i, row in df.iterrows():
                        res = predictor.predict(str(row["title"]), str(row["text"]), top_k=3)
                        labels.append(res["label"])
                        confidences.append(res["confidence"])
                        prob_reals.append(res["prob_real"])
                        prob_fakes.append(res["prob_fake"])
                        progress_bar.progress((i + 1) / total)

                    df["Prediction"] = labels
                    df["Confidence (%)"] = confidences
                    df["Prob_Real"] = prob_reals
                    df["Prob_Fake"] = prob_fakes

                st.success("Batch classification complete!")

                # Batch summary metrics
                counts = df["Prediction"].value_counts()
                fake_c = counts.get("FAKE", 0)
                real_c = counts.get("REAL", 0)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Analyzed", f"{len(df):,}")
                with col2:
                    st.metric("Detected Fake", f"{fake_c:,}", f"{fake_c/len(df):.1%}")
                with col3:
                    st.metric("Verified Real", f"{real_c:,}", f"{real_c/len(df):.1%}")
                with col4:
                    st.metric("Average Confidence", f"{df['Confidence (%)'].mean():.1f}%")

                st.subheader("Detailed Predictions Table")
                st.dataframe(df[["title", "Prediction", "Confidence (%)", "Prob_Real", "Prob_Fake"]], use_container_width=True)

                # Export CSV
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Analysis Results as CSV",
                    data=csv_bytes,
                    file_name="truthlens_batch_predictions.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Error reading file: {e}")


def page_model_benchmark():
    """Page 3: Model performance, multi-algorithm comparison, and evaluation charts."""
    st.subheader("📈 Multi-Algorithm Benchmarking & Evaluation Dashboard")
    st.write("Rigorous comparative performance results across candidate machine learning architectures under 5-Fold Stratified Cross-Validation.")

    metrics_dir = os.path.join(PROJECT_ROOT, "reports", "metrics")
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    meta_path = os.path.join(PROJECT_ROOT, "models", "best_model_meta.json")
    comp_path = os.path.join(metrics_dir, "model_comparison.csv")
    cv_path = os.path.join(metrics_dir, "cross_validation_scores.csv")
    report_path = os.path.join(metrics_dir, "classification_report.csv")

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        st.markdown(
            f"""
            <div style="background: rgba(15, 118, 110, 0.2); border: 1px solid #14B8A6; padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1.5rem;">
                <h4 style="margin: 0; color: #5EEAD4;">🏆 Champion Model: <b>{meta.get('best_model_name', 'XGBoost')}</b></h4>
                <p style="margin: 0.4rem 0 0 0; color: #E2E8F0;">Selected based on highest 5-Fold Cross-Validated F1 Score (<b>{meta.get('best_f1_score', 'N/A')}</b>) and Test Accuracy (<b>{meta.get('test_accuracy', 'N/A'):.2%}</b>).</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if os.path.exists(comp_path):
        comp_df = pd.read_csv(comp_path)
        st.subheader("Comparison Table: Evaluated ML Classifiers")
        st.dataframe(comp_df, use_container_width=True)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        comp_png = os.path.join(figures_dir, "model_comparison.png")
        if os.path.exists(comp_png):
            st.image(comp_png, caption="Multi-Metric Benchmarking Across Algorithms", use_container_width=True)

    with col_fig2:
        cv_png = os.path.join(figures_dir, "cv_scores.png")
        if os.path.exists(cv_png):
            st.image(cv_png, caption="5-Fold Cross-Validation Stability", use_container_width=True)

    st.markdown("---")
    st.subheader("Champion Model Diagnostic Curves")

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        cm_png = os.path.join(figures_dir, "confusion_matrix.png")
        if os.path.exists(cm_png):
            st.image(cm_png, caption="Normalized Confusion Matrix", use_container_width=True)

    with col_c2:
        roc_png = os.path.join(figures_dir, "roc_curve.png")
        if os.path.exists(roc_png):
            st.image(roc_png, caption="Receiver Operating Characteristic (ROC)", use_container_width=True)

    with col_c3:
        pr_png = os.path.join(figures_dir, "precision_recall_curve.png")
        if os.path.exists(pr_png):
            st.image(pr_png, caption="Precision-Recall Curve", use_container_width=True)

    if os.path.exists(report_path):
        st.markdown("---")
        st.subheader("Detailed Classification Report")
        rep_df = pd.read_csv(report_path, index_col=0)
        st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)


def page_eda_gallery():
    """Page 4: Exploratory Data Analysis figures and insights."""
    st.subheader("📊 Exploratory Data Analysis (EDA) Interactive Gallery")

    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    summary_path = os.path.join(PROJECT_ROOT, "reports", "metrics", "data_summary.json")

    total_samples_str = "9,188"
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        total_samples_str = f"{summary['cleaned_samples']['final_total_samples']:,}"
    
    st.write(f"Visual patterns and linguistic discrepancies uncovered across {total_samples_str} cleaned news articles.")

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Cleaned Samples", f"{summary['cleaned_samples']['final_total_samples']:,}")
        with c2:
            st.metric("Real News Proportion", f"{summary['cleaned_samples']['real_percentage']}%")
        with c3:
            st.metric("Fake News Proportion", f"{summary['cleaned_samples']['fake_percentage']}%")
        with c4:
            st.metric("Duplicates Purged", f"{summary['data_quality']['duplicate_count_removed']:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(figures_dir, "class_distribution.png"), caption="1. Class Balance (Real vs Fake)", use_container_width=True)
        st.image(os.path.join(figures_dir, "news_length_distribution.png"), caption="2. Article Length Distributions", use_container_width=True)
        st.image(os.path.join(figures_dir, "top_20_words.png"), caption="3. Top 20 Most Frequent Words by Class", use_container_width=True)

    with col2:
        st.image(os.path.join(figures_dir, "subject_distribution.png"), caption="4. Category & Subject Distribution", use_container_width=True)
        st.image(os.path.join(figures_dir, "correlation_heatmap.png"), caption="5. Handcrafted Stylometric Correlation Heatmap", use_container_width=True)
        st.image(os.path.join(figures_dir, "wordcloud_fake.png"), caption="6. Word Cloud: Fake News Vocabulary", use_container_width=True)

    st.image(os.path.join(figures_dir, "wordcloud_real.png"), caption="7. Word Cloud: Real News Vocabulary", use_container_width=True)


def page_about():
    """Page 5: Academic context, methodology, and base paper alignment."""
    st.subheader("ℹ️ Academic Methodology & Base Paper Alignment")

    st.markdown(
        """
        ### 📄 Base Research Paper Reference
        - **Title:** *Fake news detection based on machine learning hybrid model*
        - **Authors:** Dong Peng, Jian Song (School of Communication, Shandong University of Political Science and Law)
        - **Journal:** *Procedia Computer Science 285 (2026) 1424–1431* (7th International Conference on Cyber Security Intelligence and Analytics)
        - **Core Contribution:** Combining statistical lexical representations with semantic feature extraction to outperform single-model architectures.

        ### ⚙️ Implemented Engineering Pipeline
        1. **Data Ingestion & Cleaning:**
           - Removal of invalid characters, emojis, HTML artifacts, and numerical noise.
           - Exact deduplication purging 5,793 duplicate articles.
        2. **NLP Preprocessing:**
           - Tokenization, stopword filtering with custom news domain exclusions, and dual-pass WordNet Lemmatization.
        3. **Hybrid Feature Engineering:**
           - **Statistical Sub-pipeline:** Sublinear TF-IDF n-grams (unigrams + bigrams, top 15,000 features).
           - **Stylometric Sub-pipeline:** 5 handcrafted domain-specific features (title length, article length, capital words count, punctuation intensity, digit ratio) normalized via MinMax scaling.
        4. **Multi-Model Benchmarking:**
           - Systematic evaluation of Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, and XGBoost under 5-Fold Stratified Cross-Validation.
        5. **Explainable AI (XAI):**
           - SHAP-based local attribution to provide token-level transparency for every prediction.
        """
    )


def main():
    render_header()

    predictor = get_predictor()

    # Sidebar navigation
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shield.png", width=72)
        st.title("TruthLens Navigation")
        page = st.radio(
            "Select Module:",
            [
                "🔍 Live News Detective",
                "📁 Batch CSV Processing",
                "📈 Model Benchmarking",
                "📊 EDA Gallery",
                "ℹ️ About & Methodology"
            ]
        )

        st.markdown(
            """
            <div class="sidebar-info">
                <b>TruthLens v1.0.0</b><br>
                <span>Academic ML Pipeline</span><br>
                <span>5-Fold Stratified CV</span><br>
                <span>SHAP Explainability</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if page == "🔍 Live News Detective":
        page_live_detect(predictor)
    elif page == "📁 Batch CSV Processing":
        page_batch_predict(predictor)
    elif page == "📈 Model Benchmarking":
        page_model_benchmark()
    elif page == "📊 EDA Gallery":
        page_eda_gallery()
    elif page == "ℹ️ About & Methodology":
        page_about()


if __name__ == "__main__":
    main()
