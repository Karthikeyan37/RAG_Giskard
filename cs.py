import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from sentence_transformers import SentenceTransformer

# General setup
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Statement Similarity", layout="wide")

# -------------------------------------------------------------------
# 1️⃣  Model selection UI
# -------------------------------------------------------------------
MODEL_OPTIONS = {
    # name shown in the dropdown : HF model hub identifier
    "all-MiniLM‑L6‑v2 (very fast, 384‑d)":      "all-MiniLM-L6-v2",
    "all-mpnet-base‑v2 (general‑purpose, 768‑d)": "all-mpnet-base-v2",
    "stsb-mpnet-base‑v2 (STS‑tuned, 768‑d)":      "stsb-mpnet-base-v2"
 #   "text-embedding‑3‑small (OpenAI replica)":    "openai-community/text-embedding-3-small",
 #   "text-embedding‑3‑large (OpenAI replica)":    "openai-community/text-embedding-3-large",
}
model_choice = st.selectbox(
    "🔎 Choose an embedding model",
    list(MODEL_OPTIONS.keys()),
    index=0,
    help="MiniLM is fastest. MPNet models are larger/more accurate. "
         "The text‑embedding‑3 replicas follow OpenAI’s latest embedding spec."
)

# -------------------------------------------------------------------
# 2️⃣  Lazy‑load / cache the chosen model
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model…")
def load_model(hf_name: str):
    """Download & cache a SentenceTransformer model."""
    return SentenceTransformer(hf_name)

model = load_model(MODEL_OPTIONS[model_choice])

# -------------------------------------------------------------------
# 3️⃣  Similarity calculation helper
# -------------------------------------------------------------------
def cosine_sim(s1: str, s2: str):
    emb = model.encode([s1, s2], normalize_embeddings=True)
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0]), emb[0], emb[1]

# -------------------------------------------------------------------
# 4️⃣  UI – input & output
# -------------------------------------------------------------------
st.title("📝 Statement Similarity Calculator")
st.markdown("""
Enter two statements to compute their **semantic cosine similarity**.
The score ranges from **‑1 (very different)** to **+1 (almost identical)**.
""")

col1, col2 = st.columns(2)
with col1:
    stmt1 = st.text_area(
        "First Statement",
        "The quick brown fox jumps over the lazy dog",
        height=150
    )
with col2:
    stmt2 = st.text_area(
        "Second Statement",
        "A fast brown fox leaps over a sleepy canine",
        height=150
    )

if st.button("Calculate Similarity", type="primary"):
    if not stmt1 or not stmt2:
        st.warning("Please enter both statements.")
    else:
        with st.spinner("Embedding & comparing…"):
            score, e1, e2 = cosine_sim(stmt1, stmt2)

        # ─ Results ────────────────────────────────────────────────
        st.subheader("Results")
        st.metric("Cosine Similarity Score", f"{score:+.4f}")  # show sign
        st.progress(max(min((score + 1) / 2, 1), 0))           # map ‑1..1 → 0..1

        st.subheader("Interpretation")
        if score > 0.8:
            st.success("✅ The statements are very similar in meaning.")
        elif score > 0.6:
            st.info("ℹ️ The statements are similar.")
        elif score > 0.4:
            st.warning("⚠️ The statements are somewhat related.")
        else:
            st.error("❌ The statements appear unrelated.")

        with st.expander("Show Embedding Details"):
            st.write("First 10 dimensions of each embedding:")
            d1, d2 = np.round(e1[:10], 4), np.round(e2[:10], 4)
            st.table(
                {
                    "Dim": list(range(10)),
                    "Stmt 1": d1,
                    "Stmt 2": d2
                }
            )

# -------------------------------------------------------------------
# 5️⃣  OneDNN / TensorFlow noise mute (optional)
# -------------------------------------------------------------------
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
