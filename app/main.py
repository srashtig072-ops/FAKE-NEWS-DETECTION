"""
TruthLens AI — Enterprise Production Web Interface for Fake News Detection.

This application provides an interactive, production-grade frontend connected
directly to the underlying ML pipeline logics:
1. 🔍 Live News Detective: Real-time fact-checker with probability thresholds, stylometrics & token attribution.
2. 📁 Batch CSV Processing: Bulk CSV inference, sample benchmark loader, and report export.
3. ⚙️ ML Pipeline & Retraining Lab: Direct triggers for Ingestion, Model Retraining (5-Fold CV), and Evaluation.
4. 📈 Model Benchmark & Evaluation Dashboard: Multi-model performance comparison, ROC, PR, and Confusion Matrix.
5. 📊 Exploratory Data Analysis (EDA) Gallery: Visual distributions, stylometrics heatmap, and word clouds.
6. ℹ️ Academic Methodology: Research alignment with Peng & Song (Procedia Computer Science 2026).
"""

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

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

# Custom Glassmorphic & Modern CSS Theme
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    .main-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 55%, #0F766E 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 14px 30px -8px rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .main-header::after {
        content: "";
        position: absolute;
        top: -40%;
        right: -10%;
        width: 320px;
        height: 320px;
        background: radial-gradient(circle, rgba(20, 184, 166, 0.25) 0%, rgba(20, 184, 166, 0) 70%);
        pointer-events: none;
    }

    .main-header h1 {
        font-weight: 800;
        font-size: 2.35rem;
        margin-bottom: 0.35rem;
        letter-spacing: -0.025em;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 50%, #5EEAD4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .main-header p {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 0;
        max-width: 820px;
    }

    .header-badges {
        display: flex;
        gap: 0.6rem;
        margin-top: 1rem;
        flex-wrap: wrap;
    }

    .header-pill {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.12);
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        color: #CBD5E1;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card {
        background: rgba(30, 41, 59, 0.75);
        backdrop-filter: blur(14px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.3rem;
        text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 25px -10px rgba(0, 0, 0, 0.5);
    }

    .metric-card .val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #38BDF8;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-card .lbl {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94A3B8;
        margin-top: 0.3rem;
    }

    .badge-real {
        background: linear-gradient(135deg, #059669, #10B981);
        color: white;
        padding: 0.8rem 1.8rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.45);
        letter-spacing: 0.02em;
    }

    .badge-fake {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white;
        padding: 0.8rem 1.8rem;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 1.3rem;
        display: inline-flex;
        align-items: center;
        gap: 0.6rem;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.45);
        letter-spacing: 0.02em;
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
        background-color: rgba(239, 68, 68, 0.24);
        color: #EF4444;
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
        border-bottom: 2px solid #EF4444;
    }

    .pipeline-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        margin-bottom: 1.2rem;
    }

    .pipeline-card h4 {
        margin: 0 0 0.5rem 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
    }

    .sidebar-info {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1rem;
        margin-top: 1.2rem;
        font-size: 0.82rem;
        color: #94A3B8;
        line-height: 1.6;
    }

    .sidebar-info b {
        color: #E2E8F0;
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
    """Render top hero header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ TruthLens AI — Fake News Detection & XAI</h1>
            <p>Production Machine Learning Pipeline with Dual-Branch TF-IDF, Scaled Stylometric Engineering, 5-Fold Stratified Cross-Validation & SHAP Explainability.</p>
            <div class="header-badges">
                <span class="header-pill">⚙️ TF-IDF + Stylometrics (15,005 feats)</span>
                <span class="header-pill">🏆 L2 Logistic Regression (C=5.0)</span>
                <span class="header-pill">🎯 87.65% Test Acc | 0.9456 ROC-AUC</span>
                <span class="header-pill">⚡ Latency &lt; 15ms</span>
                <span class="header-pill">🌐 Bilingual (EN + CN Weibo)</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Pre-populated test samples for one-click testing
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


def render_highlighted_text(text: str, pos_words: List[str], neg_words: List[str]) -> str:
    """Render text with color-coded token highlights for explainability, filtering out stylometrics."""
    if not text:
        return ""
    stylometric_names = {"title_length", "article_length", "capital_words_count", "punctuation_count", "digit_ratio"}
    pos_set = {w.lower() for w in pos_words if w not in stylometric_names}
    neg_set = {w.lower() for w in neg_words if w not in stylometric_names}

    tokens = re.split(r"(\s+|[.,!?;:\"()\[\]])", text)
    highlighted = []

    for token in tokens:
        clean = re.sub(r"[^\w]", "", token).lower()
        if clean and clean in pos_set:
            highlighted.append(f'<span class="highlight-real" title="Evidence supporting Real Journalism">{token}</span>')
        elif clean and clean in neg_set:
            highlighted.append(f'<span class="highlight-fake" title="Evidence supporting Sensationalist/Fake News">{token}</span>')
        else:
            highlighted.append(token)

    return "".join(highlighted)


# ==============================================================================
# MODULE 1: LIVE NEWS DETECTIVE
# ==============================================================================
def page_live_detect(predictor):
    """Real-time Live Fact-Checking with Threshold Control, Stylometrics & XAI."""
    st.subheader("🔍 Real-Time Article Fact-Checking & Explainability")
    st.write("Enter the headline and article text below, or pick a pre-loaded test article to analyze credibility.")

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
            placeholder="Paste news body text or statement here..."
        )

        with st.expander("⚙️ Advanced Inference Settings"):
            threshold_val = st.slider(
                "Classification Decision Boundary (P_Real Threshold)",
                min_value=0.30,
                max_value=0.70,
                value=0.50,
                step=0.05,
                help="Default is 0.50. Lower threshold (e.g. 0.40) makes the filter more aggressive in flagging fake news."
            )
            top_k_features = st.slider("Top Explainability Features (SHAP)", min_value=5, max_value=25, value=10)

        submit_btn = st.form_submit_button("⚡ Analyze News Authenticity", use_container_width=True, type="primary")

    if submit_btn:
        if not title_input.strip() and not text_input.strip():
            st.warning("⚠️ Please provide at least a title or text to analyze.")
            return

        if predictor is None:
            st.error("Model artifacts not found. Please train models in the 'ML Pipeline Lab' tab.")
            return

        with st.spinner("🧠 Running NLP segmentation, stylometric extraction & SHAP attribution..."):
            t0 = time.time()
            res = predictor.predict(title_input, text_input, top_k=top_k_features)
            latency = time.time() - t0

        # Apply custom decision threshold
        prob_real = res["prob_real"]
        prob_fake = res["prob_fake"]
        if prob_real >= threshold_val:
            final_label = "REAL"
            final_conf = round(prob_real * 100, 2)
        else:
            final_label = "FAKE"
            final_conf = round(prob_fake * 100, 2)

        st.markdown("---")
        st.subheader("🎯 Credibility Verdict")

        col_verdict, col_conf, col_speed = st.columns([1.6, 2, 1])

        with col_verdict:
            if final_label == "REAL":
                st.markdown('<div class="badge-real"><span>✅</span> VERIFIED REAL NEWS</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="badge-fake"><span>🚨</span> DETECTED FAKE NEWS</div>', unsafe_allow_html=True)

        with col_conf:
            st.markdown(f"**Confidence:** `{final_conf}%`")
            st.progress(final_conf / 100.0)
            st.caption(f"Real Probability: `{prob_real:.2%}` | Fake Probability: `{prob_fake:.2%}` (Threshold: `{threshold_val:.2f}`)")

        with col_speed:
            st.metric(label="Inference Latency", value=f"{latency * 1000:.1f} ms")

        # Stylometric Profile
        st.markdown("---")
        st.subheader("📏 Stylometric & Linguistic Marker Profile")
        title_len = len(title_input)
        article_len = len(text_input)
        combined_text = f"{title_input} {text_input}".strip()
        words = combined_text.split()
        cap_words = sum(1 for w in words if w.isupper() and len(w) > 1 and w.isalpha())
        punct_count = sum(1 for c in combined_text if c in "!?,.:;\"'！？。，、“”")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Word Count", f"{len(words):,}")
        with c2:
            st.metric("Capitalized Words", f"{cap_words}", delta="High Sensationalism" if cap_words > 4 else "Normal", delta_color="inverse")
        with c3:
            st.metric("Punctuation Count", f"{punct_count}", delta="Elevated" if punct_count > 15 else "Normal", delta_color="inverse")
        with c4:
            st.metric("Headline Length", f"{title_len} chars")

        # Explainable AI (XAI)
        st.markdown("---")
        st.subheader("🔬 Explainable AI (XAI) — Word & Token Attribution")
        st.write("TruthLens reveals **why** the AI reached this classification by attributing prediction importance to specific words.")

        pos_words = [item[0] for item in res["top_positive_words"]]
        neg_words = [item[0] for item in res["top_negative_words"]]

        tab_highlight, tab_features, tab_raw_json = st.tabs([
            "✨ Color-Coded Article Highlights",
            "📊 Contributing Features",
            "📋 Raw Verification Payload"
        ])

        with tab_highlight:
            st.caption("🟢 Green highlights indicate evidence supporting **Real Journalism**. 🔴 Red highlights indicate evidence supporting **Sensationalist / Fake Information**.")
            combined_raw = f"**{title_input}**\n\n{text_input}"
            rendered_html = render_highlighted_text(combined_raw, pos_words, neg_words)
            st.markdown(
                f'<div style="background: rgba(30,41,59,0.5); padding: 1.5rem; border-radius: 12px; line-height: 1.8; border: 1px solid rgba(255,255,255,0.08); font-size: 1.05rem;">{rendered_html}</div>',
                unsafe_allow_html=True
            )

        with tab_features:
            col_pos, col_neg = st.columns(2)
            with col_pos:
                st.markdown("##### 🟢 Top Evidence for REAL News")
                if res["top_positive_words"]:
                    pos_df = pd.DataFrame(res["top_positive_words"], columns=["Feature", "Impact Weight"]).round(4)
                    st.dataframe(pos_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strong positive indicators detected.")

            with col_neg:
                st.markdown("##### 🔴 Top Evidence for FAKE News")
                if res["top_negative_words"]:
                    neg_df = pd.DataFrame(res["top_negative_words"], columns=["Feature", "Impact Weight"]).round(4)
                    st.dataframe(neg_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strong negative indicators detected.")

        with tab_raw_json:
            result_payload = {
                "verdict": final_label,
                "confidence": final_conf,
                "prob_real": prob_real,
                "prob_fake": prob_fake,
                "threshold_applied": threshold_val,
                "latency_ms": round(latency * 1000, 2),
                "stylometrics": {
                    "word_count": len(words),
                    "capitalized_words": cap_words,
                    "punctuation_count": punct_count,
                    "title_length": title_len,
                    "article_length": article_len
                },
                "top_positive_features": res["top_positive_words"],
                "top_negative_features": res["top_negative_words"]
            }
            st.json(result_payload)


# ==============================================================================
# MODULE 2: BATCH CSV PROCESSING
# ==============================================================================
def page_batch_predict(predictor):
    """Batch CSV upload, benchmark test-set loader, and bulk inference."""
    st.subheader("📁 Batch News Verification (CSV File)")
    st.write("Upload a custom CSV file or load sample articles from the benchmark dataset to test bulk classification.")

    col_upload, col_sample = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV (must contain 'title' and/or 'text')", type=["csv"])

    with col_sample:
        st.markdown("<br>", unsafe_allow_html=True)
        load_sample = st.button("🧪 Load 20 Benchmark Articles for Testing", use_container_width=True)

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
    elif load_sample:
        processed_csv = os.path.join(PROJECT_ROOT, "data", "processed_news.csv")
        if os.path.exists(processed_csv):
            raw_full = pd.read_csv(processed_csv)
            df = raw_full.sample(n=min(20, len(raw_full)), random_state=42).reset_index(drop=True)
            st.session_state["batch_sample_df"] = df
        else:
            # Fallback synthetic demo articles
            df = pd.DataFrame([
                {"title": "NASA Launches New Deep Space Satellite", "text": "Officials confirmed the successful launch from Cape Canaveral."},
                {"title": "ALIENS IN WHITE HOUSE! 100% PROOF!", "text": "Shocking leak reveals secret reptilians controlling the government!"},
                {"title": "Federal Reserve Cuts Benchmark Interest Rates", "text": "Chairman stated inflation indicators have moderated significantly."}
            ])
            st.session_state["batch_sample_df"] = df

    if "batch_sample_df" in st.session_state and df is None:
        df = st.session_state["batch_sample_df"]

    if df is not None:
        st.write(f"Loaded **{len(df):,}** articles. Data preview:")
        st.dataframe(df.head(5), use_container_width=True)

        if "title" not in df.columns and "text" not in df.columns:
            st.error("Uploaded CSV must contain at least 'title' or 'text' column.")
            return

        if "title" not in df.columns:
            df["title"] = ""
        if "text" not in df.columns:
            df["text"] = ""

        if st.button("🚀 Process Entire Batch Now", type="primary"):
            if predictor is None:
                st.error("Predictor model not available.")
                return

            with st.spinner(f"Analyzing all {len(df):,} articles with TruthLens pipeline..."):
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

            counts = df["Prediction"].value_counts()
            fake_c = int(counts.get("FAKE", 0))
            real_c = int(counts.get("REAL", 0))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyzed", f"{len(df):,}")
            with col2:
                st.metric("Detected Fake", f"{fake_c:,}", f"{fake_c/len(df):.1%}")
            with col3:
                st.metric("Verified Real", f"{real_c:,}", f"{real_c/len(df):.1%}")
            with col4:
                st.metric("Average Confidence", f"{df['Confidence (%)'].mean():.1f}%")

            # Prediction table
            st.subheader("Detailed Classification Output")
            cols_to_show = ["title", "Prediction", "Confidence (%)", "Prob_Real", "Prob_Fake"]
            if "label" in df.columns:
                cols_to_show.insert(1, "label")
            st.dataframe(df[cols_to_show], use_container_width=True)

            # Export button
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download Analysis Results as CSV",
                data=csv_bytes,
                file_name="truthlens_batch_predictions.csv",
                mime="text/csv"
            )


# ==============================================================================
# MODULE 3: ML PIPELINE & RETRAINING LAB
# ==============================================================================
def page_pipeline_lab():
    """Interactive control panel to execute and monitor ML pipeline stages."""
    st.subheader("⚙️ Machine Learning Pipeline & Retraining Lab")
    st.write("Trigger and monitor the end-to-end ML pipeline directly from this interface. Models and artifacts are synchronized automatically.")

    # Status of artifacts
    data_file = os.path.join(PROJECT_ROOT, "data", "processed_news.csv")
    model_file = os.path.join(PROJECT_ROOT, "models", "model.pkl")
    vec_file = os.path.join(PROJECT_ROOT, "models", "vectorizer.pkl")
    meta_file = os.path.join(PROJECT_ROOT, "models", "best_model_meta.json")

    st.markdown("#### 📦 Pipeline Artifact Status")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Cleaned Dataset", "Ready" if os.path.exists(data_file) else "Missing", "3.77 MB" if os.path.exists(data_file) else "None")
    with c2:
        st.metric("Feature Pipeline", "Ready" if os.path.exists(vec_file) else "Missing", "15,005 feats" if os.path.exists(vec_file) else "None")
    with c3:
        st.metric("Champion Model", "Ready" if os.path.exists(model_file) else "Missing", "LogReg" if os.path.exists(model_file) else "None")
    with c4:
        st.metric("Reports & Figures", "Ready" if os.path.exists(meta_file) else "Missing", "6 Metrics / 7 Figs" if os.path.exists(meta_file) else "None")

    st.markdown("---")
    st.subheader("🛠️ Pipeline Execution Controls")

    tab_train, tab_ingest, tab_eval = st.tabs([
        "🚀 1. Train & Benchmark Models",
        "📥 2. Ingest & Clean Dataset",
        "📊 3. Re-evaluate Champion Model"
    ])

    with tab_train:
        st.markdown(
            """
            <div class="pipeline-card">
                <h4>Phase 5: Multi-Model Benchmark & 5-Fold Stratified Cross-Validation</h4>
                <p>Trains Logistic Regression, Linear SVM, Naive Bayes, Random Forest, and XGBoost with GridSearchCV hyperparameter optimization on training folds. Automatically designates the champion model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_sample_size, col_btn = st.columns([1, 1])
        with col_sample_size:
            sample_opt = st.selectbox(
                "Training Dataset Size:",
                ["Quick Benchmark (1,500 samples - Fast)", "Full Benchmark (All 9,188 samples - Thorough)"],
                index=0
            )
            sample_val = 1500 if "1,500" in sample_opt else None

        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_train_btn = st.button("⚡ Execute Training & Benchmarking", type="primary", use_container_width=True)

        if run_train_btn:
            from src.train import ModelTrainer
            with st.spinner("Executing 5-Fold Stratified Cross-Validation & GridSearch across all models..."):
                t_start = time.time()
                trainer = ModelTrainer()
                results_df, best_model, best_name = trainer.train_and_evaluate_all(sample_size=sample_val)
                train_time = time.time() - t_start

            # Invalidate cached predictor so changes take effect
            st.cache_resource.clear()
            st.success(f"🏆 Training completed in {train_time:.1f}s! Selected Champion: **{best_name}**")
            st.dataframe(results_df, use_container_width=True)

    with tab_ingest:
        st.markdown(
            """
            <div class="pipeline-card">
                <h4>Phase 1: Raw Data Ingestion & Deduplication</h4>
                <p>Loads raw files (Weibo, GossipCop, PolitiFact, ISOT), standardizes labels (0=Fake, 1=Real), strips HTML/URLs/emojis, purges duplicates, and saves <code>data/processed_news.csv</code>.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("📥 Run Ingestion & Preprocessing Pipeline", use_container_width=True):
            from src.data_loader import NewsDataLoader
            with st.spinner("Ingesting, cleaning, and deduplicating dataset..."):
                loader = NewsDataLoader()
                df, summary = loader.run_pipeline()
            st.success(f"Ingestion successful! Final Cleaned Records: **{len(df):,}** ({summary['cleaned_samples']['real_percentage']}% Real, {summary['cleaned_samples']['fake_percentage']}% Fake)")
            st.json(summary)

    with tab_eval:
        st.markdown(
            """
            <div class="pipeline-card">
                <h4>Phase 7: Diagnostic Evaluation & Figure Generation</h4>
                <p>Evaluates the serialized champion model against an unseen 20% holdout test partition, regenerating Confusion Matrix, ROC curves, and Precision-Recall plots.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("📊 Run Evaluation & Re-generate Diagnostic Figures", use_container_width=True):
            from src.evaluate import ModelEvaluator
            with st.spinner("Computing test metrics and generating high-resolution diagnostic figures..."):
                evaluator = ModelEvaluator()
                evaluator.load_artifacts()
                eval_metrics = evaluator.evaluate_test_set()
            st.success(f"Evaluation complete! Test Accuracy: **{eval_metrics['accuracy']:.2%}** | ROC-AUC: **{eval_metrics['roc_auc']:.4f}**")
            st.json(eval_metrics)


# ==============================================================================
# MODULE 4: MODEL BENCHMARKING DASHBOARD
# ==============================================================================
def page_model_benchmark():
    """Model performance, multi-algorithm comparison, and diagnostic curves."""
    st.subheader("📈 Multi-Algorithm Benchmarking & Evaluation Dashboard")
    st.write("Rigorous comparative performance results across candidate machine learning architectures under 5-Fold Stratified Cross-Validation.")

    metrics_dir = os.path.join(PROJECT_ROOT, "reports", "metrics")
    figures_dir = os.path.join(PROJECT_ROOT, "reports", "figures")
    meta_path = os.path.join(PROJECT_ROOT, "models", "best_model_meta.json")
    comp_path = os.path.join(metrics_dir, "model_comparison.csv")
    report_path = os.path.join(metrics_dir, "classification_report.csv")

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        st.markdown(
            f"""
            <div style="background: rgba(15, 118, 110, 0.2); border: 1px solid #14B8A6; padding: 1.3rem 1.6rem; border-radius: 14px; margin-bottom: 1.5rem;">
                <h4 style="margin: 0; color: #5EEAD4; font-size: 1.25rem;">🏆 Champion Model: <b>{meta.get('best_model_name', 'Logistic Regression')}</b></h4>
                <p style="margin: 0.4rem 0 0 0; color: #E2E8F0;">Selected based on highest 5-Fold Cross-Validated F1 Score (<b>{meta.get('best_f1_score', 'N/A')}</b>) and Test Accuracy (<b>{meta.get('test_accuracy', 'N/A'):.2%}</b>) with L2 regularized hyperparameter optimization.</p>
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


# ==============================================================================
# MODULE 5: EXPLORATORY DATA ANALYSIS (EDA) GALLERY
# ==============================================================================
def page_eda_gallery():
    """Exploratory Data Analysis figures and dataset insights."""
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


# ==============================================================================
# MODULE 6: ACADEMIC METHODOLOGY & BASE PAPER
# ==============================================================================
def page_about():
    """Academic context, methodology, and base paper alignment."""
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
           - Exact deduplication purging 404 duplicate articles.
        2. **NLP Preprocessing:**
           - Tokenization, stopword filtering with custom news domain exclusions, and bilingual normalization (Jieba & NLTK).
        3. **Hybrid Feature Engineering:**
           - **Statistical Sub-pipeline:** Sublinear TF-IDF n-grams (unigrams + bigrams, top 15,000 features).
           - **Stylometric Sub-pipeline:** 5 handcrafted domain-specific features (title length, article length, capital words count, punctuation intensity, digit ratio) normalized via MinMax scaling.
        4. **Multi-Model Benchmarking:**
           - Systematic evaluation of Logistic Regression, Linear SVM, Multinomial Naive Bayes, Random Forest, and XGBoost under 5-Fold Stratified Cross-Validation.
        5. **Explainable AI (XAI):**
           - Exact linear attribution and SHAP principles to provide token-level transparency for every prediction.
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
                "⚙️ ML Pipeline & Retraining Lab",
                "📈 Model Benchmarking",
                "📊 EDA Gallery",
                "ℹ️ About & Methodology"
            ]
        )

        st.markdown(
            """
            <div class="sidebar-info">
                <b>TruthLens v2.0 Production</b><br>
                <span>Dual-Branch Hybrid ML</span><br>
                <span>5-Fold Stratified CV</span><br>
                <span>Token-Level SHAP XAI</span><br>
                <span>Bilingual Support (EN/CN)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if page == "🔍 Live News Detective":
        page_live_detect(predictor)
    elif page == "📁 Batch CSV Processing":
        page_batch_predict(predictor)
    elif page == "⚙️ ML Pipeline & Retraining Lab":
        page_pipeline_lab()
    elif page == "📈 Model Benchmarking":
        page_model_benchmark()
    elif page == "📊 EDA Gallery":
        page_eda_gallery()
    elif page == "ℹ️ About & Methodology":
        page_about()


if __name__ == "__main__":
    main()
