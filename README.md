# 📊 Meta Ad Intelligence Engine


> **An end-to-end Meta Ad analytics platform analyzing 400K+ ad events across Facebook & Instagram — with funnel analysis, ROAS/ROI tracking, A/B testing, ML-powered conversion prediction, and a live Streamlit dashboard.**

🌐 **Dataset:** [Social Media Advertisement Performance — Kaggle](https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance) &nbsp;|&nbsp; 👩‍💻 **Author:** [Divyanshi Singh — LinkedIn](https://www.linkedin.com/in/divyanshi018572) &nbsp;|&nbsp; 🚀 **Live Demo:** [Streamlit App](https://2eydel6rqdciyjzgjkcics.streamlit.app/)

---

## 💡 What This Does

SocialAds360 is a full-stack Business Intelligence dashboard built to simulate the analytics workflow of a real-world Meta Ads analyst. It ingests raw ad event data, processes it through multiple analytical lenses, and surfaces actionable insights — from top-of-funnel impressions down to revenue attribution, audience profiling, and ML-powered conversion prediction.

The project combines **data engineering**, **statistical analysis**, **machine learning**, and **interactive visualization** into a single deployable Streamlit app — covering every layer a marketing or BI analyst would care about.

---

## 🔄 Project Pipeline

**1. Data Ingestion & Merging**
Four raw CSVs (`ad_events`, `ads`, `campaigns`, `users`) are loaded via `@st.cache_data` and joined on shared keys to form a unified 400K+ row master dataframe.

**2. Feature Engineering**
Estimated cost and revenue columns are derived per event type. Day-of-week, time-of-day, and campaign budget features are prepared for downstream ML consumption.

**3. KPI Computation**
Core metrics — CTR, CVR, ROAS, ROI, CPA, Engagement Rate — are computed dynamically with full sidebar filter support (platform + ad format).

**4. Funnel & Platform Analysis**
Impression → Click → Purchase drop-off is visualized per platform, ad type, and time segment with annotated drop-off percentages.

**5. ROAS & ROI Tracking**
Revenue attribution and return metrics are broken down by campaign, platform, ad format, and target audience for budget optimization.

**6. A/B Testing Engine**
Chi-Square statistical testing compares conversion rates across ad variants with significance flags, p-values, and lift percentage calculations.

**7. Audience Segmentation (K-Means)**
Users are clustered into 5 behavioral segments — Young Digital Explorers, Mainstream Adults, Mature Professionals, Global Millennials, Emerging Market Youth — using K-Means on age, gender, and country features.

**8. ML Conversion Predictor (Random Forest)**
A class-balanced Random Forest classifier (`class_weight='balanced'`) predicts purchase likelihood from click-level features, with ROC-AUC scoring, confusion matrix, and feature importance breakdown.

**9. Export Engine**
Full analysis exported as a multi-sheet Excel workbook (Summary KPIs, Campaign Metrics, Platform Metrics, Format Metrics) or a formatted PDF summary report — all downloadable in-app.

---

## 🗂️ Dashboard Sections

| Page | What It Shows |
|------|--------------|
| 🏠 About & Overview | Project summary, pipeline, tech stack, dataset info |
| ⚡ Dashboard | Live KPIs — impressions, CTR, CVR, ROAS, ROI, CPA |
| 🔻 Funnel Analysis | Drop-off visualization across platforms and ad types |
| 💰 ROAS & ROI | Revenue attribution by campaign, format, and audience |
| 🧪 A/B Testing | Chi-Square significance testing across ad variants |
| 👥 Audience Segments | K-Means clustering — 5 behavioral user groups |
| 🤖 ML Models | Random Forest conversion predictor + feature importance |
| 📤 Export Report | Excel & PDF download with all metrics |

---

## 📐 Key Metrics Explained

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| CTR | Clicks / Impressions × 100 | Ad relevance and engagement |
| CPC | Spend / Clicks | Cost efficiency per click |
| CPM | Spend / Impressions × 1000 | Cost to reach 1,000 people |
| ROAS | Revenue / Ad Spend | Revenue generated per $1 spent |
| ROI | (Revenue − Spend) / Spend × 100 | Profit percentage on ad spend |
| CVR | Conversions / Clicks × 100 | Quality of traffic driven |
| CPA | Spend / Purchases | Cost to acquire one customer |
| Engagement Rate | (Likes + Comments + Shares) / Impressions × 100 | Audience interaction level |

---

## 🤖 ML Details

**Conversion Classifier — Random Forest**
- Target: whether a click results in a purchase
- Features: `ad_platform`, `ad_type`, `target_gender`, `target_age_group`, `day_of_week`, `time_of_day`, `total_budget`, `duration_days`
- Class imbalance handled natively via `class_weight='balanced'` (no SMOTE dependency)
- Outputs: ROC-AUC score, classification report, confusion matrix heatmap, feature importance bar chart

**Audience Segmentation — K-Means (k=5)**
- Clusters users by `age_group`, `user_gender`, `country` (label-encoded + StandardScaler)
- Produces 5 named segments ready for Meta custom audience targeting
- Visualized with pie chart distribution and a summary demographics table

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Dashboard | Streamlit |
| Data | Pandas, NumPy |
| ML / Stats | Scikit-learn (RandomForest, KMeans), SciPy (Chi-Square) |
| Visualization | Matplotlib, Seaborn |
| Export | openpyxl, fpdf2 |
| BI Tools | Power BI (DAX), Microsoft Excel + VBA |
| DevOps | Git, GitHub, Render |

---

## ⚙️ Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/Divyanshi018572/SocialAds-360-Dashboard.git
cd SocialAds-360-Dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place data files
# data/raw/ad_events.csv
# data/raw/ads.csv
# data/raw/campaigns.csv
# data/raw/users.csv

# 4. Run the app
streamlit run SocialAds.py
```

**Environment variables** — copy `.env.example` to `.env` and fill in any required config keys before running.

---

## 📦 Dataset

| Field | Detail |
|-------|--------|
| Source | [Kaggle — Social Media Advertisement Performance](https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance) |
| Platforms | Facebook, Instagram |
| Total Events | 400,000+ |
| Campaigns | 50 |
| Ads | 200 |
| Ad Formats | Image, Video, Carousel, Stories |
| Event Types | Impression, Click, Like, Comment, Share, Purchase |
| User Features | Age Group, Gender, Country |

---

## 📁 Folder Structure

```
SocialAds-360-Dashboard/
│   SocialAds.py              ← Main Streamlit app
│   requirements.txt
│   .env.example
│
├───pages/                    ← Multi-page Streamlit modules
│       Ad_Performance.py
│       Campaign_Analysis.py
│       Cluster_Results.py
│       Time_Patterns.py
│
├───data/
│   ├───raw/                  ← Source CSVs
│   │       ad_events.csv
│   │       ads.csv
│   │       campaigns.csv
│   │       users.csv
│   └───processed/            ← Cleaned & merged outputs
│           ad_metrics_clustered.csv
│           final_merged_data.csv
│           merged_ad_data.csv
│
├───ad_analysis_ml_notebooks/
│       ad_analysis.ipynb     ← EDA & ML experimentation
│
├───config/
│       config.yaml           ← App configuration
│
└───screenshots/              ← Dashboard & plot exports
        adtype_performance.png
        campaign_performance.png
        click_rate_time_patterns.png
        cluster_comparison.png
        correlation_heatmap.png
        day_of_week_events.png
        elbow_silhouette_kmeans.png
        event_type_distribution.png
        pca_2d_scatter.png
        platform_event_analysis.png
        time_based_patterns.png
```

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 📊 Dataset | [Kaggle](https://www.kaggle.com/datasets/alperenmyung/social-media-advertisement-performance) |
| 🚀 Live App | [Streamlit](https://meta-ad-intelligence-engine-noknavfbsz7x9bjo7eh3ey.streamlit.app/) |
| 💼 LinkedIn | [linkedin.com/in/divyanshi018572](https://www.linkedin.com/in/divyanshi018572) |
| 🐙 GitHub | [github.com/Divyanshi018572](https://github.com/Divyanshi018572) |

---

## 👩 Author

**Divyanshi Singh** · AI Engineer · Data Scientist · NLP Enthusiast  
B.Tech Civil Engineering + Minor in Data Science · MMMUT Gorakhpur (2022–2026)  
📍 Basti, UP &nbsp;|&nbsp; 📧 divyanshis499@gmail.com

---

> *"400K+ ad events. Every click tracked. Every rupee attributed. This is what real Meta ad analytics looks like."*
