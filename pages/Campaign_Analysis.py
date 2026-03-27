import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Campaign Analysis", layout="wide")

@st.cache_data
def load_data():
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df   = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'merged_ad_data.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

st.title("📈 Campaign Analysis")
st.markdown("---")

# ── Aggregate campaign metrics ────────────────────────────────────────────────
camp_metrics = df.groupby('name').agg(
    total_events   = ('event_id',    'count'),
    total_clicks   = ('is_click',    'sum'),
    total_purchases = ('is_purchase', 'sum'),
    total_budget   = ('total_budget', 'first'),
    duration_days  = ('duration_days','first')
).reset_index()

camp_metrics['click_rate']     = camp_metrics['total_clicks']    / camp_metrics['total_events']
camp_metrics['purchase_rate']  = camp_metrics['total_purchases'] / camp_metrics['total_events']
camp_metrics['cost_per_event'] = camp_metrics['total_budget']    / camp_metrics['total_events']
camp_metrics['cost_per_click'] = camp_metrics['total_budget']    / camp_metrics['total_clicks'].replace(0, 1)

# ── Sidebar: sort and filter ──────────────────────────────────────────────────
st.sidebar.title("Options")
sort_by = st.sidebar.selectbox(
    "Sort campaigns by",
    ['total_events', 'click_rate', 'purchase_rate', 'cost_per_event']
)
top_n = st.sidebar.slider("Show top N campaigns", 5, 50, 10)

top_camps = camp_metrics.nlargest(top_n, sort_by)

# ── KPI row ───────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Campaigns",    f"{len(camp_metrics)}")
col2.metric("Total Budget",       f"${camp_metrics['total_budget'].sum():,.0f}")
col3.metric("Avg Click Rate",     f"{camp_metrics['click_rate'].mean()*100:.2f}%")
col4.metric("Avg Cost per Event", f"${camp_metrics['cost_per_event'].mean():.2f}")

st.markdown("---")

# ── Row 1: Top campaigns ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader(f"Top {top_n} Campaigns by {sort_by.replace('_',' ').title()}")
    fig = px.bar(
        top_camps.sort_values(sort_by),
        x                        = sort_by,
        y                        = 'name',
        orientation              = 'h',
        color                    = sort_by,
        color_continuous_scale   = 'Blues'
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Budget vs Click Rate")
    fig2 = px.scatter(
        camp_metrics,
        x        = 'total_budget',
        y        = 'click_rate',
        size     = 'total_events',
        color    = 'purchase_rate',
        hover_name = 'name',
        hover_data = ['total_events', 'total_budget'],
        color_continuous_scale = 'RdYlGn',
        labels = {
            'total_budget' : 'Total Budget ($)',
            'click_rate'   : 'Click Rate',
            'purchase_rate': 'Purchase Rate'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Duration and efficiency ───────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Campaign Duration vs Total Events")
    fig3 = px.scatter(
        camp_metrics,
        x          = 'duration_days',
        y          = 'total_events',
        size       = 'total_budget',
        color      = 'click_rate',
        hover_name = 'name',
        color_continuous_scale = 'Viridis',
        labels = {
            'duration_days': 'Duration (days)',
            'total_events' : 'Total Events',
            'click_rate'   : 'CTR'
        }
    )
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.subheader("Most Cost Efficient Campaigns")
    efficient = camp_metrics.nsmallest(10, 'cost_per_event')
    fig4 = px.bar(
        efficient.sort_values('cost_per_event', ascending=True),
        x                      = 'cost_per_event',
        y                      = 'name',
        orientation            = 'h',
        color                  = 'cost_per_event',
        color_continuous_scale = 'Greens_r',
        text                   = efficient['cost_per_event'].apply(lambda x: f'${x:.2f}')
    )
    fig4.update_traces(textposition='outside')
    fig4.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)

# ── Full campaign table ───────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Full Campaign Table")
st.dataframe(
    camp_metrics.sort_values('total_events', ascending=False).style.format({
        'total_budget'   : '${:,.0f}',
        'click_rate'     : '{:.2%}',
        'purchase_rate'  : '{:.2%}',
        'cost_per_event' : '${:.2f}',
        'cost_per_click' : '${:.2f}'
    }),
    use_container_width=True
)