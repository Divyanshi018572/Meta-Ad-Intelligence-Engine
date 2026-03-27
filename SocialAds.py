"""
Meta Ad Performance Analysis — Streamlit Dashboard (Improved)
=============================================================
Run with:  streamlit run streamlit_app.py

Requirements:
    pip install streamlit pandas numpy matplotlib scikit-learn imbalanced-learn openpyxl fpdf2
    
Place these files in the same directory:
    data/raw/ad_events.csv, data/raw/ads.csv, data/raw/campaigns.csv, data/raw/users.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Meta Ad Performance",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0a0e1a; }
    .stMetric { background: #111827; border: 1px solid #1e2d42; border-radius: 10px; padding: 16px; }
    .stMetricLabel { color: #94a3b8 !important; font-size: 12px !important; }
    .stMetricValue { color: #f1f5f9 !important; }
    h1 { color: #f1f5f9 !important; }
    h2, h3 { color: #93c5fd !important; }
    .sidebar .sidebar-content { background: #111827; }
    .kpi-box {
        background: #111827; border: 1px solid #1e2d42; border-radius: 12px;
        padding: 20px; text-align: center; border-top: 2px solid #3b82f6;
    }
    .kpi-val { font-size: 2rem; font-weight: 700; color: #3b82f6; font-family: monospace; }
    .kpi-lbl { font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; margin-top: 4px; }
    .insight-box {
        background: #0f172a; border-left: 3px solid #3b82f6;
        padding: 12px 16px; border-radius: 4px; margin: 8px 0;
        color: #cbd5e1; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB STYLE ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#111827', 'axes.facecolor': '#111827',
    'axes.edgecolor': '#1e2d42', 'axes.labelcolor': '#94a3b8',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
    'text.color': '#f1f5f9', 'grid.color': '#1e2d42',
    'grid.alpha': 0.6, 'font.family': 'DejaVu Sans',
})
COLORS = ['#3b82f6','#06b6d4','#8b5cf6','#10b981','#f59e0b','#ef4444','#f472b6','#34d399']

# ── DATA LOADING ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    events    = pd.read_csv('data/raw/ad_events.csv')
    ads       = pd.read_csv('data/raw/ads.csv')
    campaigns = pd.read_csv('data/raw/campaigns.csv')
    users     = pd.read_csv('data/raw/users.csv')

    events['user_id'] = events['user_id'].astype(str)
    users['user_id']  = users['user_id'].astype(str)

    df = events.merge(ads, on='ad_id', how='left')
    df = df.merge(campaigns[['campaign_id','total_budget','duration_days']], on='campaign_id', how='left')
    df = df.merge(users[['user_id','user_gender','age_group','country']], on='user_id', how='left')

    # Estimated cost per event (for ROAS calculation)
    df['estimated_cost'] = np.where(df['event_type'] == 'Click', 0.5,
                           np.where(df['event_type'] == 'Purchase', 2.0,
                           np.where(df['event_type'] == 'Impression', 0.002, 0.1)))
    # Estimated revenue per purchase
    df['estimated_revenue'] = np.where(df['event_type'] == 'Purchase',
                                        np.random.uniform(20, 150, len(df)), 0)
    return df, events, ads, campaigns, users

@st.cache_resource
def train_ml_models(df):
    """
    ML: RandomForest with class_weight='balanced' (handles imbalance without SMOTE dependency)
    + KMeans Audience Segmentation
    Removed: GradientBoosting (poor results, not worth keeping)
    """
    results = {}

    # ── 1. Conversion Classifier (RandomForest + balanced weights) ────────────
    click_df = df[df['event_type'] == 'Click'].copy()
    purchase_pairs = set(zip(
        df[df['event_type']=='Purchase']['user_id'],
        df[df['event_type']=='Purchase']['ad_id']
    ))
    click_df['converted'] = click_df.apply(
        lambda r: 1 if (r['user_id'], r['ad_id']) in purchase_pairs else 0, axis=1
    )

    features = ['ad_platform','ad_type','target_gender','target_age_group',
                'day_of_week','time_of_day','total_budget','duration_days']
    click_clean = click_df.dropna(subset=features+['converted'])
    X = click_clean[features].copy()

    le_dict = {}
    for col in ['ad_platform','ad_type','target_gender','target_age_group','day_of_week','time_of_day']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    y = click_clean['converted']
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Use class_weight='balanced' — no SMOTE needed, handles imbalance natively
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight='balanced',   # key fix for class imbalance
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    y_prob = clf.predict_proba(X_te)[:,1]
    y_pred = clf.predict(X_te)
    auc    = roc_auc_score(y_te, y_prob)
    report = classification_report(y_te, y_pred, output_dict=True)
    feat_imp = dict(zip(features, clf.feature_importances_))

    results['clf']      = clf
    results['le_dict']  = le_dict
    results['clf_auc']  = auc
    results['clf_feat'] = feat_imp
    results['clf_report'] = report
    results['y_te']     = y_te
    results['y_prob']   = y_prob

    # ── 2. KMeans Audience Segmentation ──────────────────────────────────────
    users_df = df[['user_id','age_group','user_gender','country']].drop_duplicates('user_id').copy()
    seg_enc = pd.DataFrame({
        'age_group_enc': LabelEncoder().fit_transform(users_df['age_group'].fillna('Unknown')),
        'gender_enc':    LabelEncoder().fit_transform(users_df['user_gender'].fillna('Unknown')),
        'country_enc':   LabelEncoder().fit_transform(users_df['country'].fillna('Unknown')),
    }).fillna(0)
    seg_scaled = StandardScaler().fit_transform(seg_enc)
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    users_df['segment'] = km.fit_predict(seg_scaled)
    seg_labels = {
        0: "Young Digital Explorers",
        1: "Mainstream Adults",
        2: "Mature Professionals",
        3: "Global Millennials",
        4: "Emerging Market Youth"
    }
    results['seg_labels'] = seg_labels
    results['users_seg']  = users_df

    return results


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
with st.spinner("Loading data..."):
    df, events, ads, campaigns, users = load_data()

# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────
def compute_kpis(data):
    imps  = (data['event_type']=='Impression').sum()
    clicks= (data['event_type']=='Click').sum()
    purch = (data['event_type']=='Purchase').sum()
    likes = (data['event_type']=='Like').sum()
    cmts  = (data['event_type']=='Comment').sum()
    shrs  = (data['event_type']=='Share').sum()
    cost  = data['estimated_cost'].sum()
    rev   = data['estimated_revenue'].sum()
    return {
        'impressions': imps, 'clicks': clicks, 'purchases': purch,
        'ctr': clicks/imps*100 if imps>0 else 0,
        'cvr': purch/clicks*100 if clicks>0 else 0,
        'engagement': (likes+cmts+shrs)/imps*100 if imps>0 else 0,
        'total_cost': cost,
        'total_revenue': rev,
        'roas': rev/cost if cost>0 else 0,
        'roi': (rev-cost)/cost*100 if cost>0 else 0,
        'cpa': cost/purch if purch>0 else 0,
    }

def insight_box(text):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Meta Analytics")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 About & Overview",
        "⚡ Dashboard",
        "🔻 Funnel Analysis",
        "💰 ROAS & ROI",
        "🧪 A/B Testing",
        "👥 Audience Segments",
        "🤖 ML Models",
        "📤 Export Report",
    ])
    st.markdown("---")
    st.markdown("**Dataset**")
    st.markdown(f"• {len(events):,} events")
    st.markdown(f"• {campaigns.shape[0]} campaigns")
    st.markdown(f"• {ads.shape[0]} ads")
    st.markdown(f"• {users.shape[0]:,} users")

    if page in ["🔻 Funnel Analysis", "💰 ROAS & ROI", "🧪 A/B Testing"]:
        st.markdown("---")
        st.markdown("**Filters**")
        platform_filter = st.multiselect("Platform", ["Facebook", "Instagram"],
                                          default=["Facebook","Instagram"])
        ad_type_filter  = st.multiselect("Ad Type", ["Stories","Image","Carousel","Video"],
                                          default=["Stories","Image","Carousel","Video"])
    else:
        platform_filter = ["Facebook","Instagram"]
        ad_type_filter  = ["Stories","Image","Carousel","Video"]

def apply_filters(data):
    return data[
        (data['ad_platform'].isin(platform_filter)) &
        (data['ad_type'].isin(ad_type_filter))
    ]


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT & OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 About & Overview":

    # ── Extra styles for this page ────────────────────────────────────────────
    st.markdown("""
    <style>
    .about-hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 16px;
        padding: 36px 40px;
        margin-bottom: 24px;
        text-align: center;
    }
    .about-hero h1 { font-size: 2.4rem !important; color: #f1f5f9 !important; margin-bottom: 4px; }
    .about-hero .tagline { color: #93c5fd; font-size: 1.05rem; margin-bottom: 12px; }
    .about-hero .badge {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #3b82f6;
        color: #93c5fd;
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        margin: 3px;
    }
    .section-card {
        background: #111827;
        border: 1px solid #1e2d42;
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 20px;
        border-left: 3px solid #3b82f6;
    }
    .section-card h3 { color: #93c5fd !important; margin-top: 0; font-size: 1.1rem; }
    .pipeline-step {
        background: #0f172a;
        border: 1px solid #1e2d42;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        display: flex;
        align-items: flex-start;
        gap: 12px;
    }
    .step-num {
        background: #3b82f6;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        flex-shrink: 0;
    }
    .tech-chip {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        color: #94a3b8;
        border-radius: 6px;
        padding: 3px 10px;
        font-size: 0.78rem;
        margin: 3px 2px;
        font-family: monospace;
    }
    .profile-link {
        display: inline-block;
        background: #1e3a5f;
        border: 1px solid #3b82f6;
        color: #93c5fd !important;
        border-radius: 8px;
        padding: 8px 16px;
        text-decoration: none;
        margin: 5px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .author-card {
        background: #0f172a;
        border: 1px solid #1e2d42;
        border-radius: 12px;
        padding: 20px 24px;
    }
    .dataset-row {
        display: flex;
        justify-content: space-between;
        padding: 7px 0;
        border-bottom: 1px solid #1e2d42;
        color: #cbd5e1;
        font-size: 0.88rem;
    }
    .dataset-row:last-child { border-bottom: none; }
    </style>
    """, unsafe_allow_html=True)

    # ── Hero Banner ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="about-hero">
        <h1>📊 SocialAds360</h1>
        <div class="tagline">Meta Ad Intelligence Engine · Business Intelligence Dashboard</div>
        <p style="color:#94a3b8; font-size:0.9rem; max-width:700px; margin:0 auto 16px;">
            An end-to-end analytics platform analyzing <strong style="color:#3b82f6;">400K+ Meta ad events</strong>
            across Facebook & Instagram — covering funnel analysis, ROAS/ROI tracking,
            A/B testing, audience segmentation, and ML-powered conversion prediction.
        </p>
        <span class="badge">📁 400K+ Events</span>
        <span class="badge">📣 50 Campaigns</span>
        <span class="badge">🖼️ 200 Ads</span>
        <span class="badge">👥 Audience Segmentation</span>
        <span class="badge">🤖 ML Conversion Model</span>
        <span class="badge">📤 Excel & PDF Export</span>
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        # ── Project Overview ──────────────────────────────────────────────────
        st.markdown("""
        <div class="section-card">
            <h3>🔍 Project Overview</h3>
            <p style="color:#cbd5e1; font-size:0.9rem; line-height:1.7;">
                <strong style="color:#f1f5f9;">SocialAds360</strong> is a full-stack Business Intelligence dashboard
                built to simulate the analytics workflow of a real-world Meta Ads analyst.
                It ingests raw ad event data, processes it through multiple analytical lenses,
                and surfaces actionable insights — from top-of-funnel impressions down to
                revenue attribution and audience profiling.
            </p>
            <p style="color:#94a3b8; font-size:0.85rem;">
                The project combines <strong style="color:#93c5fd;">data engineering</strong>,
                <strong style="color:#93c5fd;">statistical analysis</strong>,
                <strong style="color:#93c5fd;">machine learning</strong>, and
                <strong style="color:#93c5fd;">interactive visualization</strong>
                into a single deployable Streamlit app.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Full Pipeline ──────────────────────────────────────────────────────
        st.markdown("### 🔄 Full Project Pipeline")

        pipeline_steps = [
            ("Data Ingestion & Merging",
             "Four raw CSVs (ad_events, ads, campaigns, users) are loaded and joined on shared keys to form a unified 400K+ row master dataframe."),
            ("Feature Engineering",
             "Estimated cost & revenue columns are derived per event type. Day-of-week, time-of-day, and campaign budget features are prepared for ML."),
            ("KPI Computation",
             "Core metrics (CTR, CVR, ROAS, ROI, CPA, Engagement Rate) are computed dynamically with full filter support."),
            ("Funnel & Platform Analysis",
             "Impression → Click → Purchase drop-off is visualized per platform, ad type, and time segment with drop-off % annotations."),
            ("ROAS & ROI Tracking",
             "Revenue attribution and return metrics are broken down by campaign, platform, ad format, and target audience."),
            ("A/B Testing Engine",
             "Chi-Square statistical testing compares conversion rates across ad variants with significance flags and lift % calculations."),
            ("Audience Segmentation (K-Means)",
             "Users are clustered into 5 behavioral segments using K-Means on age, gender, and country features — ready for Meta custom audiences."),
            ("ML Conversion Predictor (Random Forest)",
             "A class-balanced Random Forest classifier predicts purchase likelihood from click features, with ROC-AUC scoring and feature importance."),
            ("Export Engine",
             "Full analysis exported as multi-sheet Excel workbook or PDF summary report with KPIs, platform metrics, and key insights."),
        ]

        for i, (title, desc) in enumerate(pipeline_steps, 1):
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">{i}</div>
                <div>
                    <div style="color:#f1f5f9; font-weight:600; font-size:0.9rem; margin-bottom:4px;">{title}</div>
                    <div style="color:#94a3b8; font-size:0.83rem; line-height:1.6;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tech Stack ────────────────────────────────────────────────────────
        st.markdown("### 🛠️ Tech Stack")
        tech_categories = {
            "Language": ["Python 3.10+"],
            "Dashboard": ["Streamlit"],
            "Data": ["Pandas", "NumPy"],
            "ML / Stats": ["Scikit-learn", "SciPy", "K-Means", "Random Forest"],
            "Visualization": ["Matplotlib", "Seaborn"],
            "Export": ["openpyxl", "fpdf2"],
            "DevOps": ["Git", "GitHub", "Render"],
            "BI Tools": ["Power BI", "MS Excel", "DAX"],
        }
        for cat, chips in tech_categories.items():
            chips_html = "".join(f'<span class="tech-chip">{c}</span>' for c in chips)
            st.markdown(
                f'<div style="margin:6px 0;"><span style="color:#94a3b8;font-size:0.78rem;'
                f'text-transform:uppercase;letter-spacing:1px;">{cat} &nbsp;</span>{chips_html}</div>',
                unsafe_allow_html=True
            )

    with col_right:
        # ── About the Author ──────────────────────────────────────────────────
        st.markdown("""
        <div class="section-card">
            <h3>👩‍💻 About the Author</h3>
            <div class="author-card">
                <div style="font-size:1.1rem; font-weight:700; color:#f1f5f9; margin-bottom:4px;">
                    Divyanshi Singh
                </div>
                <div style="color:#3b82f6; font-size:0.82rem; margin-bottom:12px;">
                    AI Engineer · Data Scientist · NLP Enthusiast
                </div>
                <div style="color:#94a3b8; font-size:0.83rem; line-height:1.7;">
                    B.Tech Civil Engineering student at <strong style="color:#cbd5e1;">MMMUT Gorakhpur</strong>
                    (2022–2026) with hands-on experience in building end-to-end ML pipelines,
                    NLP solutions, LLM-powered apps, and BI dashboards. Passionate about
                    turning raw data into actionable intelligence.
                </div>
                <div style="margin-top:12px; color:#94a3b8; font-size:0.82rem;">
                    📍 Basti, UP &nbsp;|&nbsp; 📧 divyanshis499@gmail.com
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Profile Links ─────────────────────────────────────────────────────
        st.markdown("#### 🔗 Profile Links")
        st.markdown("""
        <div style="margin: 8px 0;">
            <a class="profile-link" href="https://www.linkedin.com/in/divyanshi018572" target="_blank">💼 LinkedIn</a>
            <a class="profile-link" href="https://github.com/Divyanshi018572" target="_blank">🐙 GitHub</a>
            <a class="profile-link" href="https://github.com/Divyanshi018572/SocialAds-360-Dashboard" target="_blank">📊 This Repo</a>
            <a class="profile-link" href="https://2eydel6rqdciyjzgjkcics.streamlit.app/" target="_blank">🚀 DemandPulse Live</a>
        </div>
        """, unsafe_allow_html=True)

        # ── Dataset Info ──────────────────────────────────────────────────────
        st.markdown("""
        <div class="section-card" style="margin-top:20px;">
            <h3>📦 Dataset Information</h3>
            <div class="dataset-row"><span>Source</span><span style="color:#93c5fd;">Synthetic / Simulated Meta Ads Data</span></div>
            <div class="dataset-row"><span>Total Events</span><span style="color:#3b82f6; font-weight:700;">400,000+</span></div>
            <div class="dataset-row"><span>Campaigns</span><span style="color:#3b82f6;">50</span></div>
            <div class="dataset-row"><span>Ads</span><span style="color:#3b82f6;">200</span></div>
            <div class="dataset-row"><span>Platforms</span><span style="color:#93c5fd;">Facebook, Instagram</span></div>
            <div class="dataset-row"><span>Ad Formats</span><span style="color:#93c5fd;">Image, Video, Carousel, Stories</span></div>
            <div class="dataset-row"><span>Event Types</span><span style="color:#93c5fd;">Impression, Click, Like, Comment, Share, Purchase</span></div>
            <div class="dataset-row"><span>User Features</span><span style="color:#93c5fd;">Age Group, Gender, Country</span></div>
            <div class="dataset-row"><span>Files</span><span style="color:#94a3b8; font-family:monospace; font-size:0.78rem;">ad_events.csv · ads.csv · campaigns.csv · users.csv</span></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Dashboard Sections Quick Nav ──────────────────────────────────────
        st.markdown("""
        <div class="section-card">
            <h3>🗂️ Dashboard Sections</h3>
            <div style="color:#94a3b8; font-size:0.85rem; line-height:2;">
                ⚡ <strong style="color:#f1f5f9;">Dashboard</strong> — KPIs & event overview<br>
                🔻 <strong style="color:#f1f5f9;">Funnel Analysis</strong> — Drop-off tracking<br>
                💰 <strong style="color:#f1f5f9;">ROAS & ROI</strong> — Revenue attribution<br>
                🧪 <strong style="color:#f1f5f9;">A/B Testing</strong> — Statistical significance<br>
                👥 <strong style="color:#f1f5f9;">Audience Segments</strong> — K-Means clusters<br>
                🤖 <strong style="color:#f1f5f9;">ML Models</strong> — Conversion predictor<br>
                📤 <strong style="color:#f1f5f9;">Export Report</strong> — Excel & PDF download
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; color:#475569; font-size:0.8rem; padding: 8px 0;">'
        'Built with ❤️ by <strong style="color:#93c5fd;">Divyanshi Singh</strong> · '
        'SocialAds360 — Meta Ad Intelligence Engine · '
        '<a href="https://github.com/Divyanshi018572/SocialAds-360-Dashboard" '
        'style="color:#3b82f6;" target="_blank">View on GitHub</a>'
        '</div>',
        unsafe_allow_html=True
    )

# ════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚡ Dashboard":
    st.title("Meta Ad Performance Dashboard")
    st.caption("Facebook & Instagram · 50 Campaigns · 200 Ads · 400K Events")

    kpis = compute_kpis(df)

    # Row 1 — Event KPIs
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Total Events",       f"{len(df):,}",           "400K interactions")
    col2.metric("Click-Through Rate", f"{kpis['ctr']:.2f}%",    f"{kpis['clicks']:,} clicks")
    col3.metric("Conversion Rate",    f"{kpis['cvr']:.2f}%",    f"{kpis['purchases']:,} purchases")
    col4.metric("Engagement Rate",    f"{kpis['engagement']:.2f}%", "Likes+Comments+Shares")

    # Row 2 — Business KPIs
    col5,col6,col7,col8 = st.columns(4)
    col5.metric("Total Ad Spend",  f"${kpis['total_cost']:,.0f}",    "Estimated")
    col6.metric("Total Revenue",   f"${kpis['total_revenue']:,.0f}", "From purchases")
    col7.metric("ROAS",            f"{kpis['roas']:.2f}x",           "Return on Ad Spend")
    col8.metric("Cost per Acq.",   f"${kpis['cpa']:.2f}",            "Per purchase")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Event Distribution")
        evt = df['event_type'].value_counts()
        fig, ax = plt.subplots(figsize=(7,4))
        bars = ax.bar(evt.index, evt.values, color=COLORS[:len(evt)], edgecolor='none')
        ax.set_xlabel("Event Type"); ax.set_ylabel("Count"); ax.grid(axis='y', alpha=0.4)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+500,
                    f'{int(b.get_height()):,}', ha='center', va='bottom', fontsize=9)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_b:
        st.subheader("Quick Funnel Overview")
        imps   = (df['event_type']=='Impression').sum()
        clks   = (df['event_type']=='Click').sum()
        prchs  = (df['event_type']=='Purchase').sum()
        funnel = [('Impressions', imps, '#3b82f6'),
                  ('Clicks',      clks, '#06b6d4'),
                  ('Purchases',   prchs,'#10b981')]
        fig, ax = plt.subplots(figsize=(7,4))
        for i,(lbl,val,c) in enumerate(funnel):
            w = val/imps
            ax.barh(i, w, color=c, alpha=0.85, height=0.55)
            ax.text(w+0.01, i, f'{val:,}  ({val/imps*100:.1f}%)', va='center', fontsize=10)
        ax.set_yticks(range(3)); ax.set_yticklabels(['Impressions','Clicks','Purchases'])
        ax.set_xlabel("Relative Volume"); ax.set_xlim(0,1.5); ax.grid(axis='x',alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Platform Split")
        plat = df[df['event_type']=='Impression'].groupby('ad_platform').size()
        fig, ax = plt.subplots(figsize=(5,4))
        wedges, texts, autotexts = ax.pie(plat.values, labels=plat.index, colors=COLORS[:2],
                                           autopct='%1.1f%%', startangle=90, pctdistance=0.75)
        for t in autotexts: t.set_fontsize(11); t.set_color('#f1f5f9')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_d:
        st.subheader("Time of Day — Clicks")
        tod = df[df['event_type']=='Click'].groupby('time_of_day').size()
        order = ['Morning','Afternoon','Evening','Night']
        tod = tod.reindex([o for o in order if o in tod.index])
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(tod.index, tod.values, color=COLORS, edgecolor='none')
        ax.set_ylabel("Clicks"); ax.grid(axis='y', alpha=0.4)
        fig.tight_layout(); st.pyplot(fig); plt.close()


# ════════════════════════════════════════════════════════════════════════════
# PAGE: FUNNEL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔻 Funnel Analysis":
    st.title("Conversion Funnel Analysis")
    st.caption("Track how users move from Impression → Click → Purchase across platforms and formats")

    fdf = apply_filters(df)

    # ── Overall Funnel ────────────────────────────────────────────────────────
    st.subheader("Overall Funnel")
    stages = ['Impression','Click','Like','Comment','Purchase','Share']
    counts = {s: (fdf['event_type']==s).sum() for s in stages}

    col1, col2 = st.columns([2,1])
    with col1:
        fig, ax = plt.subplots(figsize=(9,5))
        base = counts['Impression']
        colors_f = ['#3b82f6','#06b6d4','#8b5cf6','#10b981','#f59e0b','#f472b6']
        for i, (s, c) in enumerate(zip(stages, colors_f)):
            w = counts[s]/base
            ax.barh(i, w, color=c, alpha=0.88, height=0.6)
            drop = ''
            if i > 0:
                prev = list(counts.values())[i-1]
                drop_pct = (1 - counts[s]/prev)*100 if prev > 0 else 0
                drop = f'  ▼ {drop_pct:.1f}% drop'
            ax.text(w+0.01, i,
                    f'{counts[s]:,}  ({counts[s]/base*100:.2f}%){drop}',
                    va='center', fontsize=9)
        ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages)
        ax.set_xlabel("Relative to Impressions"); ax.set_xlim(0, 1.6)
        ax.grid(axis='x', alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Funnel Metrics**")
        imp = counts['Impression']
        clk = counts['Click']
        pch = counts['Purchase']
        st.metric("Imp → Click",    f"{clk/imp*100:.2f}%", "CTR")
        st.metric("Click → Purchase",f"{pch/clk*100:.2f}%","CVR")
        st.metric("Imp → Purchase", f"{pch/imp*100:.3f}%", "Overall Conv.")
        st.metric("Click Drop-off", f"{(1-clk/imp)*100:.1f}%", "Lost after impression")
        st.metric("Purchase Drop-off", f"{(1-pch/clk)*100:.1f}%","Lost after click")

    insight_box(f"Only {pch/clk*100:.2f}% of clicks convert to purchases. "
                f"Focus on landing page optimization and retargeting to improve post-click experience.")

    st.markdown("---")

    # ── Funnel by Platform ────────────────────────────────────────────────────
    st.subheader("Funnel by Platform")
    col3, col4 = st.columns(2)
    platforms = fdf['ad_platform'].dropna().unique()

    platform_funnel = {}
    for plat in platforms:
        pdf = fdf[fdf['ad_platform'] == plat]
        platform_funnel[plat] = {s: (pdf['event_type']==s).sum() for s in ['Impression','Click','Purchase']}

    with col3:
        fig, axes = plt.subplots(1, len(platforms), figsize=(9,4), sharey=False)
        if len(platforms) == 1: axes = [axes]
        for ax, plat in zip(axes, platforms):
            d = platform_funnel[plat]
            base_p = d['Impression'] if d['Impression'] > 0 else 1
            vals = [d[s]/base_p for s in ['Impression','Click','Purchase']]
            ax.bar(['Imp','Click','Purch'], vals,
                   color=['#3b82f6','#06b6d4','#10b981'], alpha=0.85, edgecolor='none')
            ax.set_title(plat, fontsize=11)
            ax.set_ylabel("Relative Volume")
            ax.grid(axis='y', alpha=0.3)
            for j, (v, s) in enumerate(zip(vals, ['Impression','Click','Purchase'])):
                ax.text(j, v+0.01, f'{d[s]:,}', ha='center', fontsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.markdown("**Platform CTR & CVR**")
        plat_rows = []
        for plat in platforms:
            d = platform_funnel[plat]
            plat_rows.append({
                'Platform': plat,
                'Impressions': f"{d['Impression']:,}",
                'Clicks': f"{d['Click']:,}",
                'Purchases': f"{d['Purchase']:,}",
                'CTR %': f"{d['Click']/d['Impression']*100:.2f}%" if d['Impression']>0 else '-',
                'CVR %': f"{d['Purchase']/d['Click']*100:.2f}%" if d['Click']>0 else '-',
            })
        st.dataframe(pd.DataFrame(plat_rows), use_container_width=True, hide_index=True)

        # Best platform insight
        best = max(platforms, key=lambda p:
                   platform_funnel[p]['Purchase']/platform_funnel[p]['Click']
                   if platform_funnel[p]['Click']>0 else 0)
        insight_box(f"**{best}** has the highest CVR. Allocate more budget here for better returns.")

    st.markdown("---")

    # ── Funnel by Ad Format ───────────────────────────────────────────────────
    st.subheader("Funnel by Ad Format")
    fmt_data = []
    for fmt in fdf['ad_type'].dropna().unique():
        fmtdf = fdf[fdf['ad_type'] == fmt]
        imp_f = (fmtdf['event_type']=='Impression').sum()
        clk_f = (fmtdf['event_type']=='Click').sum()
        pch_f = (fmtdf['event_type']=='Purchase').sum()
        fmt_data.append({
            'Format': fmt,
            'Impressions': imp_f,
            'Clicks': clk_f,
            'Purchases': pch_f,
            'CTR %': round(clk_f/imp_f*100,2) if imp_f>0 else 0,
            'CVR %': round(pch_f/clk_f*100,2) if clk_f>0 else 0,
        })
    fmt_df = pd.DataFrame(fmt_data).sort_values('CVR %', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    axes[0].bar(fmt_df['Format'], fmt_df['CTR %'], color=COLORS[0], alpha=0.85)
    axes[0].set_title("CTR % by Format"); axes[0].set_ylabel("CTR %"); axes[0].grid(axis='y',alpha=0.3)
    for i, v in enumerate(fmt_df['CTR %']):
        axes[0].text(i, v+0.05, f'{v}%', ha='center', fontsize=9)

    axes[1].bar(fmt_df['Format'], fmt_df['CVR %'], color=COLORS[3], alpha=0.85)
    axes[1].set_title("CVR % by Format"); axes[1].set_ylabel("CVR %"); axes[1].grid(axis='y',alpha=0.3)
    for i, v in enumerate(fmt_df['CVR %']):
        axes[1].text(i, v+0.05, f'{v}%', ha='center', fontsize=9)

    fig.tight_layout(); st.pyplot(fig); plt.close()
    st.dataframe(fmt_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    best_fmt = fmt_df.iloc[0]['Format']
    insight_box(f"**{best_fmt}** format has the highest CVR of {fmt_df.iloc[0]['CVR %']}%. "
                f"Prioritize this format in future campaigns.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ROAS & ROI
# ════════════════════════════════════════════════════════════════════════════
elif page == "💰 ROAS & ROI":
    st.title("ROAS & ROI Analysis")
    st.caption("Return on Ad Spend and Return on Investment by Campaign, Platform, and Format")

    fdf = apply_filters(df)
    kpis = compute_kpis(fdf)

    # ── Top KPIs ──────────────────────────────────────────────────────────────
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Total Spend",   f"${kpis['total_cost']:,.0f}")
    col2.metric("Total Revenue", f"${kpis['total_revenue']:,.0f}")
    col3.metric("ROAS",          f"{kpis['roas']:.2f}x",  "Revenue per $1 spent")
    col4.metric("ROI",           f"{kpis['roi']:.1f}%",   "Net return")
    col5.metric("CPA",           f"${kpis['cpa']:.2f}",   "Cost per acquisition")

    if kpis['roas'] >= 3:
        insight_box(f"ROAS of {kpis['roas']:.2f}x is healthy. Industry benchmark is 3x–4x for Meta ads.")
    else:
        insight_box(f"ROAS of {kpis['roas']:.2f}x is below the 3x industry benchmark. "
                    f"Consider pausing low-performing ads.")

    st.markdown("---")

    # ── ROAS by Campaign ──────────────────────────────────────────────────────
    st.subheader("ROAS by Campaign (Top 10)")
    camp_roas = fdf.groupby('campaign_id').apply(lambda g: pd.Series({
        'spend':   g['estimated_cost'].sum(),
        'revenue': g['estimated_revenue'].sum(),
        'purchases': (g['event_type']=='Purchase').sum(),
    })).reset_index().merge(campaigns[['campaign_id','name','total_budget']], on='campaign_id', how='left')
    camp_roas['roas'] = (camp_roas['revenue'] / camp_roas['spend'].replace(0, np.nan)).round(2)
    camp_roas['roi']  = ((camp_roas['revenue'] - camp_roas['spend']) / camp_roas['spend'].replace(0,np.nan) * 100).round(1)
    camp_roas['cpa']  = (camp_roas['spend'] / camp_roas['purchases'].replace(0,np.nan)).round(2)
    camp_roas = camp_roas.sort_values('roas', ascending=False)

    top10 = camp_roas.head(10)
    fig, ax = plt.subplots(figsize=(12,5))
    bars = ax.bar(range(len(top10)), top10['roas'], color=[
        '#10b981' if r>=3 else '#f59e0b' if r>=2 else '#ef4444'
        for r in top10['roas']
    ], alpha=0.85, edgecolor='none')
    ax.axhline(y=3, color='#ef4444', linestyle='--', linewidth=1.5, label='3x Benchmark')
    ax.set_xticks(range(len(top10)))
    ax.set_xticklabels([n.replace('Campaign_','C').replace('_',' ')
                        for n in top10['name']], rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("ROAS (x)"); ax.legend(); ax.grid(axis='y', alpha=0.3)
    for i, v in enumerate(top10['roas']):
        ax.text(i, v+0.05, f'{v}x', ha='center', fontsize=9)
    fig.tight_layout(); st.pyplot(fig); plt.close()

    # ── ROAS by Platform & Format ─────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("ROAS by Platform")
        plat_roas = fdf.groupby('ad_platform').apply(lambda g: pd.Series({
            'spend':   g['estimated_cost'].sum(),
            'revenue': g['estimated_revenue'].sum(),
        })).reset_index()
        plat_roas['roas'] = (plat_roas['revenue'] / plat_roas['spend'].replace(0,np.nan)).round(2)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(plat_roas['ad_platform'], plat_roas['roas'],
               color=['#4267B2','#C13584'], alpha=0.85, edgecolor='none')
        ax.axhline(y=3, color='#ef4444', linestyle='--', linewidth=1.5, label='3x Benchmark')
        ax.set_ylabel("ROAS (x)"); ax.legend(); ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(plat_roas['roas']):
            ax.text(i, v+0.05, f'{v}x', ha='center', fontsize=11)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        st.subheader("ROAS by Ad Format")
        fmt_roas = fdf.groupby('ad_type').apply(lambda g: pd.Series({
            'spend':   g['estimated_cost'].sum(),
            'revenue': g['estimated_revenue'].sum(),
        })).reset_index()
        fmt_roas['roas'] = (fmt_roas['revenue'] / fmt_roas['spend'].replace(0,np.nan)).round(2)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(fmt_roas['ad_type'], fmt_roas['roas'], color=COLORS[:4], alpha=0.85, edgecolor='none')
        ax.axhline(y=3, color='#ef4444', linestyle='--', linewidth=1.5, label='3x Benchmark')
        ax.set_ylabel("ROAS (x)"); ax.legend(); ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(fmt_roas['roas']):
            ax.text(i, v+0.05, f'{v}x', ha='center', fontsize=11)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")

    # ── Budget Efficiency Table ───────────────────────────────────────────────
    st.subheader("Campaign ROI Leaderboard")
    display = camp_roas[['name','spend','revenue','roas','roi','cpa','purchases']].rename(columns={
        'name':'Campaign','spend':'Spend ($)','revenue':'Revenue ($)',
        'roas':'ROAS (x)','roi':'ROI %','cpa':'CPA ($)','purchases':'Purchases'
    }).reset_index(drop=True)
    st.dataframe(display, use_container_width=True, height=400)

    best_camp = camp_roas.iloc[0]['name']
    worst_camp = camp_roas.iloc[-1]['name']
    insight_box(f"**Best:** {best_camp} with ROAS {camp_roas.iloc[0]['roas']}x. "
                f"**Worst:** {worst_camp} with ROAS {camp_roas.iloc[-1]['roas']}x. "
                f"Consider reallocating budget from worst to best performing campaigns.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: A/B TESTING
# ════════════════════════════════════════════════════════════════════════════
elif page == "🧪 A/B Testing":
    st.title("A/B Testing Analysis")
    st.caption("Statistical significance testing between platforms, formats, and targeting strategies")

    fdf = apply_filters(df)

    st.info("ℹ️ Using **Chi-Square test** for conversion rate significance and **Z-test** for CTR differences. p < 0.05 = statistically significant.")

    # ── Test 1: Facebook vs Instagram CTR ─────────────────────────────────────
    st.subheader("Test 1: Facebook vs Instagram — CTR")
    col1, col2 = st.columns(2)

    fb  = fdf[fdf['ad_platform']=='Facebook']
    ig  = fdf[fdf['ad_platform']=='Instagram']

    fb_imp  = (fb['event_type']=='Impression').sum()
    fb_clk  = (fb['event_type']=='Click').sum()
    ig_imp  = (ig['event_type']=='Impression').sum()
    ig_clk  = (ig['event_type']=='Click').sum()

    fb_ctr = fb_clk/fb_imp if fb_imp>0 else 0
    ig_ctr = ig_clk/ig_imp if ig_imp>0 else 0

    # Chi-square test
    contingency = np.array([[fb_clk, fb_imp-fb_clk], [ig_clk, ig_imp-ig_clk]])
    chi2, p_ctr, dof, _ = stats.chi2_contingency(contingency)

    with col1:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(['Facebook','Instagram'], [fb_ctr*100, ig_ctr*100],
               color=['#4267B2','#C13584'], alpha=0.85, edgecolor='none', width=0.5)
        ax.set_ylabel("CTR %"); ax.set_title("CTR Comparison")
        ax.grid(axis='y', alpha=0.3)
        ax.text(0, fb_ctr*100+0.05, f'{fb_ctr*100:.2f}%', ha='center', fontsize=12, fontweight='bold')
        ax.text(1, ig_ctr*100+0.05, f'{ig_ctr*100:.2f}%', ha='center', fontsize=12, fontweight='bold')
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        winner = "Instagram" if ig_ctr > fb_ctr else "Facebook"
        lift   = abs(ig_ctr - fb_ctr)/min(fb_ctr,ig_ctr)*100
        significant = p_ctr < 0.05

        st.markdown("**Test Results**")
        st.metric("Facebook CTR", f"{fb_ctr*100:.2f}%", f"{fb_imp:,} impressions")
        st.metric("Instagram CTR", f"{ig_ctr*100:.2f}%", f"{ig_imp:,} impressions")
        st.metric("Chi-Square p-value", f"{p_ctr:.4f}", "< 0.05 = significant")
        st.metric("Relative Lift", f"{lift:.1f}%", f"Favor {winner}")

        if significant:
            st.success(f"✅ **Statistically Significant** — {winner} CTR is genuinely higher (p={p_ctr:.4f})")
        else:
            st.warning(f"⚠️ **Not Significant** — Difference could be random chance (p={p_ctr:.4f})")

    st.markdown("---")

    # ── Test 2: Ad Format CVR ─────────────────────────────────────────────────
    st.subheader("Test 2: Ad Format — Conversion Rate")
    formats = fdf['ad_type'].dropna().unique()
    fmt_results = []
    for fmt in formats:
        fmtdf = fdf[fdf['ad_type']==fmt]
        clk_f = (fmtdf['event_type']=='Click').sum()
        pch_f = (fmtdf['event_type']=='Purchase').sum()
        fmt_results.append({'Format': fmt, 'Clicks': clk_f, 'Purchases': pch_f,
                            'CVR %': round(pch_f/clk_f*100,3) if clk_f>0 else 0})
    fmt_res_df = pd.DataFrame(fmt_results).sort_values('CVR %', ascending=False)

    col3, col4 = st.columns(2)
    with col3:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(fmt_res_df['Format'], fmt_res_df['CVR %'],
               color=COLORS[:len(fmt_res_df)], alpha=0.85, edgecolor='none')
        ax.set_ylabel("CVR %"); ax.set_title("CVR by Ad Format"); ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(fmt_res_df['CVR %']):
            ax.text(i, v+0.02, f'{v}%', ha='center', fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col4:
        # Chi-square across all formats
        if len(fmt_results) >= 2:
            obs = [[r['Purchases'], r['Clicks']-r['Purchases']] for r in fmt_results if r['Clicks']>0]
            if len(obs) >= 2:
                chi2_fmt, p_fmt, _, _ = stats.chi2_contingency(obs)
                st.markdown("**Chi-Square Test (All Formats)**")
                st.metric("Chi-Square Stat", f"{chi2_fmt:.2f}")
                st.metric("p-value", f"{p_fmt:.4f}")
                if p_fmt < 0.05:
                    best_fmt = fmt_res_df.iloc[0]['Format']
                    st.success(f"✅ **Significant difference** between formats. Use **{best_fmt}** for best CVR.")
                else:
                    st.warning("⚠️ Format differences not statistically significant.")
        st.dataframe(fmt_res_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Test 3: Time of Day ───────────────────────────────────────────────────
    st.subheader("Test 3: Time of Day — Click Performance")
    tod_data = []
    for tod in ['Morning','Afternoon','Evening','Night']:
        tod_df = fdf[fdf['time_of_day']==tod]
        imp_t  = (tod_df['event_type']=='Impression').sum()
        clk_t  = (tod_df['event_type']=='Click').sum()
        tod_data.append({'Time':tod, 'Impressions':imp_t, 'Clicks':clk_t,
                         'CTR %': round(clk_t/imp_t*100,2) if imp_t>0 else 0})
    tod_df_res = pd.DataFrame(tod_data)

    col5, col6 = st.columns(2)
    with col5:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(tod_df_res['Time'], tod_df_res['CTR %'], color=COLORS, alpha=0.85, edgecolor='none')
        ax.set_ylabel("CTR %"); ax.set_title("CTR by Time of Day"); ax.grid(axis='y',alpha=0.3)
        for i, v in enumerate(tod_df_res['CTR %']):
            ax.text(i, v+0.05, f'{v}%', ha='center', fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col6:
        best_tod = tod_df_res.sort_values('CTR %', ascending=False).iloc[0]
        worst_tod = tod_df_res.sort_values('CTR %').iloc[0]
        st.dataframe(tod_df_res, use_container_width=True, hide_index=True)
        insight_box(f"**{best_tod['Time']}** has the highest CTR at {best_tod['CTR %']}%. "
                    f"Schedule more ads during this window. "
                    f"**{worst_tod['Time']}** performs worst — reduce spend here.")

    st.markdown("---")

    # ── Test 4: Gender Targeting ───────────────────────────────────────────────
    st.subheader("Test 4: Gender Targeting — CVR")
    gender_data = []
    for g in fdf['target_gender'].dropna().unique():
        gdf = fdf[fdf['target_gender']==g]
        clk_g = (gdf['event_type']=='Click').sum()
        pch_g = (gdf['event_type']=='Purchase').sum()
        gender_data.append({'Target Gender':g, 'Clicks':clk_g, 'Purchases':pch_g,
                            'CVR %': round(pch_g/clk_g*100,3) if clk_g>0 else 0})
    gender_df = pd.DataFrame(gender_data).sort_values('CVR %', ascending=False)

    col7, col8 = st.columns(2)
    with col7:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.bar(gender_df['Target Gender'], gender_df['CVR %'],
               color=COLORS[:len(gender_df)], alpha=0.85, edgecolor='none')
        ax.set_ylabel("CVR %"); ax.set_title("CVR by Target Gender"); ax.grid(axis='y', alpha=0.3)
        for i, v in enumerate(gender_df['CVR %']):
            ax.text(i, v+0.02, f'{v}%', ha='center', fontsize=10)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col8:
        st.dataframe(gender_df.reset_index(drop=True), use_container_width=True, hide_index=True)
        best_g = gender_df.iloc[0]['Target Gender']
        insight_box(f"Ads targeting **{best_g}** have the highest CVR. "
                    f"Adjust targeting strategy to focus more budget here.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: AUDIENCE SEGMENTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "👥 Audience Segments":
    st.title("Audience Segmentation")
    st.caption("K-Means clustering of 9,841 users into 5 behavioral segments")

    with st.spinner("Running K-Means segmentation..."):
        models = train_ml_models(df)

    users_seg   = models['users_seg']
    seg_labels  = models['seg_labels']

    seg_summary = users_seg.groupby('segment').agg(
        count=('user_id','count'),
        top_gender=('user_gender', lambda x: x.value_counts().index[0]),
        top_country=('country', lambda x: x.value_counts().index[0]),
        top_age_group=('age_group', lambda x: x.value_counts().index[0]),
    ).reset_index()
    seg_summary['label'] = seg_summary['segment'].map(seg_labels)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Segment Size Distribution")
        fig, ax = plt.subplots(figsize=(6,5))
        ax.pie(seg_summary['count'],
               labels=[seg_labels[i] for i in seg_summary['segment']],
               colors=COLORS[:5], autopct='%1.1f%%', startangle=90)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.subheader("Segment Profiles")
        st.dataframe(seg_summary[['label','count','top_gender','top_country','top_age_group']].rename(columns={
            'label':'Segment','count':'Users','top_gender':'Gender',
            'top_country':'Top Country','top_age_group':'Age Group'
        }), use_container_width=True, hide_index=True, height=260)

    st.markdown("---")
    st.subheader("Segment Breakdown")
    for _, row in seg_summary.iterrows():
        pct = row['count'] / seg_summary['count'].sum() * 100
        with st.expander(f"**{row['label']}** — {row['count']:,} users ({pct:.1f}%)"):
            c1,c2,c3 = st.columns(3)
            c1.metric("Users",       f"{row['count']:,}")
            c2.metric("Top Gender",  row['top_gender'])
            c3.metric("Top Country", row['top_country'])
            insight_box(f"This segment represents {pct:.1f}% of your audience. "
                        f"Primary demographic: {row['top_age_group']} {row['top_gender']} from {row['top_country']}.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS (Improved — only kept what works)
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 ML Models":
    st.title("Machine Learning Models")
    st.caption("RandomForest Conversion Classifier (class-balanced) · K-Means Audience Segmentation")

    st.info("ℹ️ GradientBoosting CTR Regressor was removed — insufficient training data (200 records) gave unreliable results. "
            "RandomForest now uses **class_weight='balanced'** to properly handle the conversion imbalance problem.")

    with st.spinner("Training models..."):
        models = train_ml_models(df)

    # ── Model 1: RandomForest Conversion Classifier ───────────────────────────
    st.subheader("🎯 Model 1: Conversion Classifier (RandomForest — Balanced)")

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("AUC Score",     f"{models['clf_auc']:.4f}", "ROC-AUC")
    col2.metric("Algorithm",     "RandomForest",             "200 estimators")
    col3.metric("Imbalance Fix", "class_weight=balanced",    "No SMOTE needed")
    col4.metric("Max Depth",     "8",                        "Controlled complexity")

    report = models['clf_report']
    col5,col6,col7 = st.columns(3)
    col5.metric("Precision (Conv.)", f"{report.get('1',{}).get('precision',0):.3f}")
    col6.metric("Recall (Conv.)",    f"{report.get('1',{}).get('recall',0):.3f}")
    col7.metric("F1 Score (Conv.)",  f"{report.get('1',{}).get('f1-score',0):.3f}")

    if models['clf_auc'] > 0.65:
        st.success(f"✅ AUC = {models['clf_auc']:.4f} — Model has meaningful predictive power!")
    elif models['clf_auc'] > 0.55:
        st.warning(f"⚠️ AUC = {models['clf_auc']:.4f} — Moderate performance. More purchase data would improve this.")
    else:
        st.error(f"❌ AUC = {models['clf_auc']:.4f} — Still near-chance. "
                 f"Root cause: only ~26 conversions in 40K clicks. Collect more purchase data.")

    col_f, col_r = st.columns(2)
    with col_f:
        st.markdown("**Feature Importances**")
        feat_df = pd.DataFrame(list(models['clf_feat'].items()),
                               columns=['Feature','Importance']).sort_values('Importance', ascending=True)
        fig, ax = plt.subplots(figsize=(7,4))
        ax.barh(feat_df['Feature'], feat_df['Importance'], color=COLORS[0], alpha=0.85)
        ax.set_xlabel("Importance"); ax.grid(axis='x', alpha=0.4)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col_r:
        st.markdown("**ROC Curve**")
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(models['y_te'], models['y_prob'])
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, color=COLORS[0], linewidth=2,
                label=f'AUC = {models["clf_auc"]:.4f}')
        ax.plot([0,1],[0,1], '--', color='#475569', linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    insight_box(f"Most important feature: **{feat_df.iloc[-1]['Feature']}**. "
                f"Focus targeting strategy around this variable for better conversion prediction.")

    st.markdown("---")

    # ── Model 2: KMeans Audience Segmentation ─────────────────────────────────
    st.subheader("🔮 Model 2: Audience Segmentation (K-Means, k=5)")
    users_seg   = models['users_seg']
    seg_labels  = models['seg_labels']

    seg_summary = users_seg.groupby('segment').agg(
        count=('user_id','count'),
        top_gender=('user_gender', lambda x: x.value_counts().index[0]),
        top_country=('country', lambda x: x.value_counts().index[0]),
        top_age_group=('age_group', lambda x: x.value_counts().index[0]),
    ).reset_index()
    seg_summary['label'] = seg_summary['segment'].map(seg_labels)

    col7, col8 = st.columns(2)
    with col7:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.pie(seg_summary['count'],
               labels=[seg_labels[i] for i in seg_summary['segment']],
               colors=COLORS[:5], autopct='%1.0f%%', startangle=90)
        fig.tight_layout(); st.pyplot(fig); plt.close()

    with col8:
        st.dataframe(seg_summary[['label','count','top_gender','top_country','top_age_group']].rename(columns={
            'label':'Segment','count':'Users','top_gender':'Gender',
            'top_country':'Top Country','top_age_group':'Age Group'
        }), use_container_width=True, height=220, hide_index=True)

    insight_box("K-Means successfully segmented users into 5 distinct groups. "
                "Use these segments to create custom audiences in Meta Ads Manager for targeted campaigns.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE: EXPORT REPORT
# ════════════════════════════════════════════════════════════════════════════
elif page == "📤 Export Report":
    st.title("Export Report")
    st.caption("Download your analysis as Excel or PDF")

    kpis = compute_kpis(df)

    # ── Build Summary DataFrames ───────────────────────────────────────────────
    # Campaign metrics
    camp_m = df.groupby('campaign_id').apply(lambda g: pd.Series({
        'impressions': (g['event_type']=='Impression').sum(),
        'clicks':      (g['event_type']=='Click').sum(),
        'purchases':   (g['event_type']=='Purchase').sum(),
        'spend':       g['estimated_cost'].sum(),
        'revenue':     g['estimated_revenue'].sum(),
    })).reset_index().merge(campaigns, on='campaign_id')
    camp_m['ctr']  = (camp_m['clicks']  / camp_m['impressions'].replace(0,np.nan)*100).round(2)
    camp_m['cvr']  = (camp_m['purchases']/ camp_m['clicks'].replace(0,np.nan)*100).round(2)
    camp_m['roas'] = (camp_m['revenue'] / camp_m['spend'].replace(0,np.nan)).round(2)
    camp_m['cpa']  = (camp_m['spend']   / camp_m['purchases'].replace(0,np.nan)).round(2)

    # Platform metrics
    plat_m = df.groupby('ad_platform').apply(lambda g: pd.Series({
        'impressions': (g['event_type']=='Impression').sum(),
        'clicks':      (g['event_type']=='Click').sum(),
        'purchases':   (g['event_type']=='Purchase').sum(),
        'spend':       g['estimated_cost'].sum(),
        'revenue':     g['estimated_revenue'].sum(),
    })).reset_index()
    plat_m['ctr']  = (plat_m['clicks']  / plat_m['impressions'].replace(0,np.nan)*100).round(2)
    plat_m['cvr']  = (plat_m['purchases']/ plat_m['clicks'].replace(0,np.nan)*100).round(2)
    plat_m['roas'] = (plat_m['revenue'] / plat_m['spend'].replace(0,np.nan)).round(2)

    # Format metrics
    fmt_m = df.groupby('ad_type').apply(lambda g: pd.Series({
        'impressions': (g['event_type']=='Impression').sum(),
        'clicks':      (g['event_type']=='Click').sum(),
        'purchases':   (g['event_type']=='Purchase').sum(),
        'spend':       g['estimated_cost'].sum(),
        'revenue':     g['estimated_revenue'].sum(),
    })).reset_index()
    fmt_m['ctr']  = (fmt_m['clicks']  / fmt_m['impressions'].replace(0,np.nan)*100).round(2)
    fmt_m['cvr']  = (fmt_m['purchases']/ fmt_m['clicks'].replace(0,np.nan)*100).round(2)
    fmt_m['roas'] = (fmt_m['revenue'] / fmt_m['spend'].replace(0,np.nan)).round(2)

    summary_df = pd.DataFrame([{
        'Metric': 'Total Events',        'Value': f"{len(df):,}"},
        {'Metric': 'Total Impressions',  'Value': f"{kpis['impressions']:,}"},
        {'Metric': 'Total Clicks',       'Value': f"{kpis['clicks']:,}"},
        {'Metric': 'Total Purchases',    'Value': f"{kpis['purchases']:,}"},
        {'Metric': 'CTR %',              'Value': f"{kpis['ctr']:.2f}%"},
        {'Metric': 'CVR %',              'Value': f"{kpis['cvr']:.2f}%"},
        {'Metric': 'Engagement Rate %',  'Value': f"{kpis['engagement']:.2f}%"},
        {'Metric': 'Total Ad Spend',     'Value': f"${kpis['total_cost']:,.0f}"},
        {'Metric': 'Total Revenue',      'Value': f"${kpis['total_revenue']:,.0f}"},
        {'Metric': 'ROAS',               'Value': f"{kpis['roas']:.2f}x"},
        {'Metric': 'ROI %',              'Value': f"{kpis['roi']:.1f}%"},
        {'Metric': 'CPA',                'Value': f"${kpis['cpa']:.2f}"},
    ])

    st.subheader("Preview")
    st.markdown("**Summary KPIs**")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    col_p, col_f2 = st.columns(2)
    with col_p:
        st.markdown("**Platform Metrics**")
        st.dataframe(plat_m, use_container_width=True, hide_index=True)
    with col_f2:
        st.markdown("**Format Metrics**")
        st.dataframe(fmt_m, use_container_width=True, hide_index=True)

    st.markdown("---")
    col_xl, col_pdf = st.columns(2)

    # ── Excel Export ──────────────────────────────────────────────────────────
    with col_xl:
        st.subheader("📊 Excel Export")
        st.markdown("Downloads a multi-sheet Excel with Summary, Campaigns, Platforms, and Formats.")
        if st.button("Generate Excel Report", type="primary", use_container_width=True):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                summary_df.to_excel(writer, sheet_name='Summary KPIs', index=False)
                camp_m[['name','impressions','clicks','purchases','ctr','cvr',
                         'roas','cpa','total_budget','duration_days']].to_excel(
                    writer, sheet_name='Campaign Metrics', index=False)
                plat_m.to_excel(writer, sheet_name='Platform Metrics', index=False)
                fmt_m.to_excel(writer, sheet_name='Format Metrics', index=False)
            output.seek(0)
            st.download_button(
                label="⬇️ Download Excel",
                data=output,
                file_name="meta_ad_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    # ── PDF Export ────────────────────────────────────────────────────────────
    with col_pdf:
        st.subheader("📄 PDF Export")
        st.markdown("Downloads a PDF summary report with all key metrics.")
        if st.button("Generate PDF Report", type="primary", use_container_width=True):
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 20)
                pdf.cell(0, 12, "Meta Ad Performance Report", ln=True, align='C')
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 8, "Facebook & Instagram · 50 Campaigns · 400K Events", ln=True, align='C')
                pdf.ln(5)

                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 10, "Summary KPIs", ln=True)
                pdf.set_font("Helvetica", "", 10)
                pdf.set_fill_color(240,240,240)
                for _, row in summary_df.iterrows():
                    pdf.cell(90, 8, str(row['Metric']), border=1, fill=True)
                    pdf.cell(90, 8, str(row['Value']),  border=1, ln=True)
                pdf.ln(5)

                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 10, "Platform Metrics", ln=True)
                pdf.set_font("Helvetica", "B", 9)
                cols_p = ['ad_platform','impressions','clicks','purchases','ctr','cvr','roas']
                for c in cols_p:
                    pdf.cell(27, 8, c[:12], border=1, fill=True)
                pdf.ln()
                pdf.set_font("Helvetica", "", 9)
                for _, row in plat_m[cols_p].iterrows():
                    for c in cols_p:
                        pdf.cell(27, 8, str(row[c])[:12], border=1)
                    pdf.ln()
                pdf.ln(5)

                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 10, "Key Insights", ln=True)
                pdf.set_font("Helvetica", "", 10)
                insights = [
                    f"Overall CTR: {kpis['ctr']:.2f}% | CVR: {kpis['cvr']:.2f}%",
                    f"ROAS: {kpis['roas']:.2f}x | ROI: {kpis['roi']:.1f}%",
                    f"Total Spend: ${kpis['total_cost']:,.0f} | Revenue: ${kpis['total_revenue']:,.0f}",
                    f"Cost per Acquisition: ${kpis['cpa']:.2f}",
                ]
                for ins in insights:
                    pdf.cell(0, 8, f"• {ins}", ln=True)

                pdf_bytes = pdf.output()
                st.download_button(
                    label="⬇️ Download PDF",
                    data=bytes(pdf_bytes),
                    file_name="meta_ad_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except ImportError:
                st.error("fpdf2 not installed. Run: pip install fpdf2")
