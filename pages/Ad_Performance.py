import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Ad Performance", layout="wide")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df         = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'merged_ad_data.csv'))
    ad_metrics = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'ad_metrics_clustered.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df, ad_metrics

df, ad_metrics = load_data()

# Reconstruct platform column
df['ad_platform'] = 'Facebook'
df.loc[df['platform_Instagram'] == 1, 'ad_platform'] = 'Instagram'

# Reconstruct ad_type column
for t in ['Carousel', 'Image', 'Stories', 'Video']:
    col = f'type_{t}'
    if col in df.columns:
        df.loc[df[col] == 1, 'ad_type'] = t

st.title("📊 Ad Performance Overview")
st.markdown("---")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("Filters")

platform_filter = st.sidebar.multiselect(
    "Platform",
    options = df['ad_platform'].unique().tolist(),
    default = df['ad_platform'].unique().tolist()
)

adtype_filter = st.sidebar.multiselect(
    "Ad Type",
    options = df['ad_type'].unique().tolist(),
    default = df['ad_type'].unique().tolist()
)

# Apply filters
filtered = df[
    (df['ad_platform'].isin(platform_filter)) &
    (df['ad_type'].isin(adtype_filter))
]

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Events",    f"{len(filtered):,}")
col2.metric("Click Rate",      f"{filtered['is_click'].mean()*100:.2f}%")
col3.metric("Purchase Rate",   f"{filtered['is_purchase'].mean()*100:.2f}%")
col4.metric("Engagement Rate", f"{filtered['is_engagement'].mean()*100:.2f}%")

st.markdown("---")

# ── Row 1: Platform and Ad Type performance ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Click Rate by Platform")
    ctr_plat = filtered.groupby('ad_platform')['is_click'].mean().reset_index()
    ctr_plat.columns = ['platform', 'click_rate']
    ctr_plat['click_rate'] = ctr_plat['click_rate'] * 100

    fig = px.bar(
        ctr_plat,
        x     = 'platform',
        y     = 'click_rate',
        color = 'platform',
        text  = ctr_plat['click_rate'].apply(lambda x: f'{x:.2f}%'),
        color_discrete_sequence = ['#4267B2', '#C13584']
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False, yaxis_title='CTR (%)')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Click Rate by Ad Type")
    ctr_type = filtered.groupby('ad_type')['is_click'].mean().reset_index()
    ctr_type.columns = ['ad_type', 'click_rate']
    ctr_type['click_rate'] = ctr_type['click_rate'] * 100
    ctr_type = ctr_type.sort_values('click_rate', ascending=False)

    fig2 = px.bar(
        ctr_type,
        x     = 'ad_type',
        y     = 'click_rate',
        color = 'ad_type',
        text  = ctr_type['click_rate'].apply(lambda x: f'{x:.2f}%'),
        color_discrete_sequence = px.colors.qualitative.Set2
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(showlegend=False, yaxis_title='CTR (%)')
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Event breakdown and purchase rate ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Event Type Breakdown by Platform")
    ct = pd.crosstab(
        filtered['ad_platform'],
        filtered['event_type'],
        normalize='index'
    ) * 100
    ct = ct.reset_index().melt(
        id_vars='ad_platform',
        var_name='event_type',
        value_name='percentage'
    )
    fig3 = px.bar(
        ct,
        x        = 'ad_platform',
        y        = 'percentage',
        color    = 'event_type',
        barmode  = 'stack',
        color_discrete_sequence = px.colors.qualitative.Set3
    )
    fig3.update_layout(yaxis_title='Percentage (%)')
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Purchase Rate by Ad Type")
    pvr_type = filtered.groupby('ad_type')['is_purchase'].mean().reset_index()
    pvr_type.columns = ['ad_type', 'purchase_rate']
    pvr_type['purchase_rate'] = pvr_type['purchase_rate'] * 100
    pvr_type = pvr_type.sort_values('purchase_rate', ascending=False)

    fig4 = px.bar(
        pvr_type,
        x     = 'ad_type',
        y     = 'purchase_rate',
        color = 'ad_type',
        text  = pvr_type['purchase_rate'].apply(lambda x: f'{x:.2f}%'),
        color_discrete_sequence = px.colors.qualitative.Pastel
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(showlegend=False, yaxis_title='Purchase Rate (%)')
    st.plotly_chart(fig4, use_container_width=True)

# ── Row 3: Targeting impact ───────────────────────────────────────────────────
st.markdown("---")
st.subheader("Does Targeting Improve Performance? (Known Users Only)")

known = filtered[filtered['user_gender'] != 'Unknown']

if len(known) > 0:
    col1, col2, col3 = st.columns(3)

    metrics = [
        ('is_click',      'Click Rate',      col1),
        ('is_purchase',   'Purchase Rate',   col2),
        ('is_engagement', 'Engagement Rate', col3)
    ]

    for metric, title, col in metrics:
        rates = known.groupby('is_well_targeted')[metric].mean() * 100
        rates_df = pd.DataFrame({
            'targeting': ['Not Targeted', 'Well Targeted'],
            'rate'     : rates.values
        })
        fig = px.bar(
            rates_df,
            x     = 'targeting',
            y     = 'rate',
            color = 'targeting',
            text  = rates_df['rate'].apply(lambda x: f'{x:.2f}%'),
            color_discrete_sequence = ['#FF6B6B', '#51CF66'],
            title = title
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(showlegend=False, yaxis_title='Rate (%)')
        col.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No known users in current filter selection")