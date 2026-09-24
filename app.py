import os
import sys
import time
import streamlit as st
from PIL import Image

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Milvus Semantic Image Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Design & High-Contrast Typography ─────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global Font & Theme */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* Background */
.stApp {
    background: radial-gradient(circle at 15% 15%, #181935 0%, #0d0f1d 45%, #080912 100%);
    color: #f1f5f9;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── SIDEBAR STYLING ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #131528 0%, #0d0e1c 100%) !important;
    border-right: 1px solid rgba(168, 85, 247, 0.25) !important;
    padding: 1.5rem 1rem !important;
}

/* Sidebar Title & Section Headers */
.sidebar-header {
    font-size: 1.15rem;
    font-weight: 800;
    color: #ffffff;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(168, 85, 247, 0.25);
    letter-spacing: -0.3px;
}
.sidebar-header span.icon {
    font-size: 1.3rem;
}

.sidebar-label {
    font-size: 0.88rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-top: 1rem;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* Fix Streamlit default text in sidebar */
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #e2e8f0 !important;
    font-weight: 500;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #94a3b8 !important;
    font-size: 0.8rem;
}

/* Multiselect & Selectbox inside Sidebar */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] {
    background-color: rgba(26, 29, 54, 0.95) !important;
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    border-radius: 10px !important;
}

[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.4), rgba(59, 130, 246, 0.3)) !important;
    border: 1px solid rgba(168, 85, 247, 0.5) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
}

/* Multiselect placeholder */
[data-testid="stSidebar"] [data-baseweb="select"] div[aria-hidden="true"] {
    color: #94a3b8 !important;
}

/* Slider value display */
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: #94a3b8 !important;
    font-weight: 600 !important;
}

/* Filter code box */
.filter-box {
    background: rgba(18, 20, 39, 0.85);
    border: 1px solid rgba(59, 130, 246, 0.35);
    border-radius: 10px;
    padding: 0.65rem 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #60a5fa;
    word-break: break-all;
    line-height: 1.4;
    margin-top: 0.3rem;
}

/* Info card in sidebar */
.sidebar-info-card {
    background: linear-gradient(145deg, rgba(30, 27, 75, 0.6), rgba(15, 23, 42, 0.8));
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 12px;
    padding: 0.9rem;
    margin-top: 1.2rem;
}
.sidebar-info-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.25rem 0;
    font-size: 0.82rem;
    color: #94a3b8;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}
.sidebar-info-row:last-child {
    border-bottom: none;
}
.sidebar-info-row strong {
    color: #e2e8f0;
    font-weight: 600;
}

/* ── HERO BANNER ── */
.hero-card {
    background: linear-gradient(135deg, rgba(37, 30, 92, 0.85) 0%, rgba(26, 38, 84, 0.8) 50%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(168, 85, 247, 0.3);
    border-radius: 20px;
    padding: 2.2rem 2.2rem 1.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 10px 35px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(10px);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #c084fc 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero-desc {
    color: #cbd5e1;
    font-size: 0.98rem;
    margin: 0;
    line-height: 1.5;
}
.hero-pills {
    display: flex;
    gap: 0.5rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}
.hero-pill {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: #e2e8f0;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}
.hero-pill-purple { border-color: rgba(168, 85, 247, 0.4); color: #c084fc; background: rgba(168, 85, 247, 0.12); }
.hero-pill-blue   { border-color: rgba(59, 130, 246, 0.4); color: #93c5fd; background: rgba(59, 130, 246, 0.12); }
.hero-pill-green  { border-color: rgba(52, 211, 153, 0.4); color: #6ee7b7; background: rgba(52, 211, 153, 0.12); }

/* ── SEARCH BAR & FORM ── */
.search-container {
    background: rgba(22, 25, 48, 0.75);
    border: 1px solid rgba(168, 85, 247, 0.35);
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
}

/* Style Primary Buttons (Search Button) */
button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 18px rgba(139, 92, 246, 0.4) !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(139, 92, 246, 0.6) !important;
}

/* Style Secondary Buttons (Example Chips) */
button[kind="secondary"] {
    background: rgba(30, 34, 66, 0.85) !important;
    border: 1px solid rgba(168, 85, 247, 0.35) !important;
    border-radius: 20px !important;
    color: #e2e8f0 !important;
    font-weight: 500 !important;
    font-size: 0.83rem !important;
    padding: 0.35rem 0.9rem !important;
    transition: all 0.2s ease !important;
}
button[kind="secondary"]:hover {
    background: rgba(55, 48, 114, 0.9) !important;
    border-color: #a855f7 !important;
    color: #ffffff !important;
    transform: translateY(-1px) !important;
}

/* Input box */
.stTextInput > div > div > input {
    background: rgba(18, 20, 39, 0.9) !important;
    border: 1px solid rgba(168, 85, 247, 0.4) !important;
    border-radius: 12px !important;
    color: #ffffff !important;
    font-size: 1.02rem !important;
    padding: 0.75rem 1.1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #a855f7 !important;
    box-shadow: 0 0 0 3px rgba(168, 85, 247, 0.25) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #94a3b8 !important;
}

/* ── STATS CARDS ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0 1.8rem 0;
}
.stat-box {
    background: linear-gradient(145deg, rgba(28, 31, 61, 0.8), rgba(18, 20, 42, 0.9));
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 14px;
    padding: 0.9rem 1rem;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}
.stat-box-num {
    font-size: 1.6rem;
    font-weight: 800;
    color: #c084fc;
    line-height: 1.2;
}
.stat-box-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.25rem;
}

/* ── RESULT CARDS ── */
.image-result-card {
    background: linear-gradient(160deg, rgba(28, 31, 62, 0.9) 0%, rgba(16, 18, 36, 0.95) 100%);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 16px;
    overflow: hidden;
    margin-bottom: 1.4rem;
    transition: all 0.25s ease;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}
.image-result-card:hover {
    transform: translateY(-4px);
    border-color: rgba(168, 85, 247, 0.6);
    box-shadow: 0 12px 30px rgba(124, 58, 237, 0.25);
}
.card-header-bar {
    padding: 0.75rem 1rem;
    background: linear-gradient(90deg, rgba(124, 58, 237, 0.25), rgba(59, 130, 246, 0.15));
    border-bottom: 1px solid rgba(168, 85, 247, 0.2);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card-rank {
    background: linear-gradient(135deg, #8b5cf6, #3b82f6);
    color: #ffffff;
    font-weight: 800;
    font-size: 0.82rem;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
}
.card-percentage {
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.card-body-content {
    padding: 0.9rem 1rem 1rem;
}
.card-filename {
    font-weight: 700;
    color: #ffffff;
    font-size: 0.92rem;
    margin-bottom: 0.5rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.card-tags {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
    margin-bottom: 0.6rem;
}
.card-tag {
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #cbd5e1;
    font-size: 0.74rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 6px;
}
.card-tag span {
    color: #c084fc;
}
.bar-track {
    background: rgba(255, 255, 255, 0.08);
    height: 6px;
    border-radius: 999px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.bar-value {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #8b5cf6, #34d399);
}

/* ── EMPTY & NOT FOUND STATES ── */
.placeholder-box {
    text-align: center;
    padding: 3.5rem 1.5rem;
    background: rgba(22, 25, 48, 0.4);
    border: 2px dashed rgba(168, 85, 247, 0.25);
    border-radius: 20px;
    color: #94a3b8;
    margin: 2rem 0;
}
.placeholder-box .icon {
    font-size: 3.2rem;
    margin-bottom: 0.8rem;
    display: block;
}
.placeholder-box h3 {
    color: #e2e8f0;
    font-size: 1.2rem;
    margin-bottom: 0.4rem;
}

/* Section Dividers */
.styled-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.6rem 0 1rem;
}
.styled-header h2 {
    font-size: 1.25rem;
    font-weight: 800;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.3px;
}
.styled-header .line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(168, 85, 247, 0.5), transparent);
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ─────────────────────────────────────────────────────────────────
def calibrate_score(raw_cosine: float) -> float:
    """Transforms raw CLIP cosine similarities into an intuitive percentage."""
    min_thresh, max_thresh = 0.15, 0.35
    clipped = max(min_thresh, min(raw_cosine, max_thresh))
    return ((clipped - min_thresh) / (max_thresh - min_thresh)) * 100.0


def build_filter_expr(image_classes: list, cameras: list) -> str | None:
    """Constructs a valid Milvus scalar boolean filter string."""
    parts = []
    if image_classes:
        quoted = [f'"{c}"' for c in image_classes]
        parts.append(f"image_class in [{', '.join(quoted)}]")
    if cameras:
        quoted = [f'"{c}"' for c in cameras]
        parts.append(f"camera in [{', '.join(quoted)}]")
    return " and ".join(parts) if parts else None


@st.cache_resource(show_spinner=False)
def load_model_and_db():
    """Loads CLIP model and Milvus client once and caches for fast subsequent queries."""
    from models import HFCLIPModel
    from vectordb import Project5VectorStore
    model = HFCLIPModel()
    vdb = Project5VectorStore()
    return model, vdb


# ── Constants ───────────────────────────────────────────────────────────────
AVAILABLE_CAMERAS = ["gate-01", "gate-02", "perimeter-east", "lobby-main", "warehouse-north"]
AVAILABLE_CLASSES = ["animal", "person", "vehicle", "nature", "general"]
CLASS_ICONS = {
    "animal": "🐾", "person": "👤", "vehicle": "🚗",
    "nature": "🌿", "general": "📷"
}
CAMERA_ICONS = {
    "gate-01": "🚪", "gate-02": "🚪", "perimeter-east": "🔭",
    "lobby-main": "🏛️", "warehouse-north": "🏭"
}

# Ensure session state variables
if "search_query" not in st.session_state:
    st.session_state["search_query"] = ""
if "trigger_search" not in st.session_state:
    st.session_state["trigger_search"] = False


# ═══════════════════════════════════════════════════════════════════════════
# SIDEBAR: SEARCH FILTERS & CONTROLS
# ═══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <span class="icon">⚙️</span>
        <span>Search Controls</span>
    </div>
    """, unsafe_allow_html=True)

    # Top-K Slider
    st.markdown('<div class="sidebar-label">🎯 Results Count (Top-K)</div>', unsafe_allow_html=True)
    top_k = st.slider(
        label="Top-K count",
        min_value=1,
        max_value=24,
        value=6,
        step=1,
        label_visibility="collapsed",
        help="Adjust the number of top matching images to retrieve from Milvus"
    )

    # Class Filter
    st.markdown('<div class="sidebar-label">🏷️ Filter by Class</div>', unsafe_allow_html=True)
    selected_classes = st.multiselect(
        label="Select Image Classes",
        options=AVAILABLE_CLASSES,
        default=[],
        format_func=lambda c: f"{CLASS_ICONS.get(c, '🏷️')} {c.capitalize()}",
        placeholder="All classes (no filter)",
        label_visibility="collapsed"
    )

    # Camera Filter
    st.markdown('<div class="sidebar-label">📸 Filter by Camera</div>', unsafe_allow_html=True)
    selected_cameras = st.multiselect(
        label="Select Cameras",
        options=AVAILABLE_CAMERAS,
        default=[],
        format_func=lambda c: f"{CAMERA_ICONS.get(c, '📸')} {c}",
        placeholder="All cameras (no filter)",
        label_visibility="collapsed"
    )

    # Filter Expression Display
    filter_expr = build_filter_expr(selected_classes, selected_cameras)
    st.markdown('<div class="sidebar-label">⚡ Active Milvus Filter</div>', unsafe_allow_html=True)
    if filter_expr:
        st.markdown(f'<div class="filter-box">{filter_expr}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="filter-box" style="color:#94a3b8;font-style:italic">None (searching entire dataset)</div>', unsafe_allow_html=True)

    # System & Database Info Card
    st.markdown("""
    <div class="sidebar-info-card">
        <div style="font-weight:700;color:#c084fc;font-size:0.86rem;margin-bottom:0.5rem">
            📊 System Specs
        </div>
        <div class="sidebar-info-row">
            <span>Model</span>
            <strong>CLIP ViT-B/32</strong>
        </div>
        <div class="sidebar-info-row">
            <span>Embedding Dim</span>
            <strong>512-D</strong>
        </div>
        <div class="sidebar-info-row">
            <span>Vector Engine</span>
            <strong>Milvus Lite</strong>
        </div>
        <div class="sidebar-info-row">
            <span>Index / Metric</span>
            <strong>FLAT / Cosine</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT AREA
# ═══════════════════════════════════════════════════════════════════════════

# Hero Header
st.markdown("""
<div class="hero-card">
    <div class="hero-title">🔍 Milvus Hybrid Image Retrieval Engine</div>
    <div class="hero-desc">
        Perform lightning-fast semantic image search using deep vision-language embeddings paired with Milvus vector indexing and scalar filtering.
    </div>
    <div class="hero-pills">
        <span class="hero-pill hero-pill-purple">🧠 OpenAI CLIP ViT-B/32</span>
        <span class="hero-pill hero-pill-blue">⚡ Milvus Vector DB</span>
        <span class="hero-pill hero-pill-green">🎯 Hybrid Scalar Search</span>
        <span class="hero-pill">📊 512-D Cosine Metric</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Pre-load Models & DB Connection
with st.spinner("Connecting to Milvus vector database and CLIP encoder..."):
    try:
        model, vdb = load_model_and_db()
        ready = True
    except Exception as e:
        st.error(f"❌ Failed to load model or Milvus vector database: {e}")
        ready = False

# Search Form (Supports Enter Key & Button Click natively)
st.markdown("""
<div class="styled-header">
    <h2>💬 Enter Natural Language Query</h2>
    <div class="line"></div>
</div>
""", unsafe_allow_html=True)

with st.form("search_form", clear_on_submit=False):
    col_input, col_submit = st.columns([5, 1], vertical_alignment="center")
    with col_input:
        query_val = st.text_input(
            label="Search query",
            value=st.session_state["search_query"],
            placeholder="Describe any scene, e.g., 'a boy skateboard trick', 'children playing in garden', 'black dog running'...",
            label_visibility="collapsed",
            key="text_input_box"
        )
    with col_submit:
        submitted = st.form_submit_button("🔍 Search", type="primary", use_container_width=True)

# Update state if form submitted
if submitted:
    st.session_state["search_query"] = query_val
    st.session_state["trigger_search"] = True

# Quick Suggestion Chips (Styled as subtle, clean badges)
st.markdown('<div style="font-size:0.83rem;color:#94a3b8;font-weight:600;margin-top:0.2rem;margin-bottom:0.4rem">💡 QUICK SUGGESTIONS:</div>', unsafe_allow_html=True)
examples = [
    ("🛹 Skateboard trick", "a boy skateboard trick off a metal plank"),
    ("🧒 Children playing", "children playing in garden"),
    ("🐕 Dog in snow", "black dog running after a white dog in snow"),
    ("🚗 Vehicle at gate", "vehicle near gate entrance"),
    ("🧍 Person walking", "person walking near building"),
]

chip_cols = st.columns(len(examples))
for idx, (label, ex_query) in enumerate(examples):
    with chip_cols[idx]:
        if st.button(label, key=f"chip_{idx}", type="secondary", use_container_width=True):
            st.session_state["search_query"] = ex_query
            st.session_state["trigger_search"] = True
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# SEARCH EXECUTION & RESULTS PRESENTATION
# ═══════════════════════════════════════════════════════════════════════════
active_query = st.session_state.get("search_query", "").strip()

if active_query and ready and (st.session_state.get("trigger_search", False) or submitted):
    with st.spinner(f"🔍 Searching database for '{active_query}'..."):
        try:
            # 1. Encode text query to 512-D vector
            query_vec = model.encode_text(active_query)
            
            # 2. ANN Vector search in Milvus with hybrid scalar filter & latency telemetry
            active_filter = filter_expr
            t0 = time.perf_counter()
            hits = vdb.search(query_vec, scalar_filter=active_filter, top_k=top_k)
            latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as err:
            st.error(f"Search query error: {err}")
            hits = []
            latency_ms = 0.0

    # ── Summary Metrics Bar ──
    if hits:
        best_score = calibrate_score(hits[0]["score"])
        avg_score = sum(calibrate_score(h["score"]) for h in hits) / len(hits)
        unique_classes = len(set(h.get("image_class") for h in hits if h.get("image_class")))

        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-box-num">{len(hits)}</div>
                <div class="stat-box-title">Retrieved</div>
            </div>
            <div class="stat-box">
                <div class="stat-box-num" style="color:#34d399">{best_score:.1f}%</div>
                <div class="stat-box-title">Best Match</div>
            </div>
            <div class="stat-box">
                <div class="stat-box-num" style="color:#60a5fa">{avg_score:.1f}%</div>
                <div class="stat-box-title">Average Match</div>
            </div>
            <div class="stat-box">
                <div class="stat-box-num" style="color:#e879f9">{unique_classes}</div>
                <div class="stat-box-title">Unique Classes</div>
            </div>
            <div class="stat-box">
                <div class="stat-box-num" style="color:#facc15">{latency_ms:.2f} ms</div>
                <div class="stat-box-title">Query Latency</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Results Header
    filter_label_display = f"<code style='color:#a855f7'>{filter_expr}</code>" if filter_expr else "<span style='color:#94a3b8'>None</span>"
    st.markdown(f"""
    <div class="styled-header">
        <h2>Results for <span style="color:#c084fc">"{active_query}"</span></h2>
        <div class="line"></div>
    </div>
    <div style="font-size:0.83rem;color:#94a3b8;margin-bottom:1.2rem">
        Active Filter: {filter_label_display} &nbsp;•&nbsp; Limit: <strong style="color:#ffffff">{top_k}</strong>
    </div>
    """, unsafe_allow_html=True)

    if not hits:
        st.markdown("""
        <div class="placeholder-box">
            <span class="icon">🔍</span>
            <h3>No Matching Images Found</h3>
            <p>No vectors matched your query within the selected scalar filters.<br>Try clearing some filters or refining your query keywords.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display results in 3-column responsive grid
        cols_per_row = 3
        for start_idx in range(0, len(hits), cols_per_row):
            current_row = hits[start_idx : start_idx + cols_per_row]
            row_columns = st.columns(cols_per_row)

            for col, (hit_item, offset) in zip(row_columns, ((h, start_idx + i + 1) for i, h in enumerate(current_row))):
                with col:
                    file_path = hit_item.get("file_path", "")
                    file_name = os.path.basename(file_path) if file_path else "unknown.jpg"
                    calibrated = calibrate_score(hit_item["score"])
                    raw_score = hit_item["score"]
                    img_class = hit_item.get("image_class", "general")
                    camera = hit_item.get("camera", "unknown")
                    date_str = hit_item.get("capture_date", "—")

                    # Dynamic score color
                    if calibrated >= 75:
                        score_color = "#34d399"
                    elif calibrated >= 45:
                        score_color = "#fbbf24"
                    else:
                        score_color = "#f87171"

                    # Card container
                    st.markdown(f"""
                    <div class="image-result-card">
                        <div class="card-header-bar">
                            <span class="card-rank">#{offset}</span>
                            <span class="card-percentage" style="color:{score_color}">{calibrated:.1f}%</span>
                        </div>
                        <div class="card-body-content">
                            <div class="card-filename" title="{file_name}">📄 {file_name}</div>
                            <div class="card-tags">
                                <span class="card-tag">{CLASS_ICONS.get(img_class, '🏷️')} <span>{img_class}</span></span>
                                <span class="card-tag">{CAMERA_ICONS.get(camera, '📸')} <span>{camera}</span></span>
                                <span class="card-tag">📅 <span>{date_str}</span></span>
                            </div>
                            <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#94a3b8;margin-top:0.4rem">
                                <span>Cosine Score</span>
                                <span style="font-family:'JetBrains Mono',monospace;color:#e2e8f0">{raw_score:.4f}</span>
                            </div>
                            <div class="bar-track">
                                <div class="bar-value" style="width:{min(100, max(0, int(calibrated)))}%"></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Image rendering
                    if file_path and os.path.exists(file_path):
                        try:
                            pil_img = Image.open(file_path)
                            st.image(pil_img, use_container_width=True)
                        except Exception as img_err:
                            st.warning(f"Unable to render image: {img_err}")
                    else:
                        st.info(f"📁 Image file path not found on disk:\n`{file_path}`")

elif not active_query:
    st.markdown("""
    <div class="placeholder-box">
        <span class="icon">🖼️</span>
        <h3>Ready to Search</h3>
        <p>Type a description in the query box above or select one of the quick suggestions to explore matching images.</p>
    </div>
    """, unsafe_allow_html=True)
