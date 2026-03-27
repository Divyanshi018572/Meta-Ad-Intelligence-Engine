import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Time Patterns", layout="wide")

@st.cache_data
def load_data():
    import os
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df   = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'merged_ad_data.csv'))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

df = load_data()

st.title("⏰ Time Patterns")
st.markdown("When do ads perform best?")
st.markdown("---")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.title("Filters")
event_filter = st.sidebar.multiselect(
    "Event Type",
    options = df['event_type'].unique().tolist(),
    default = df['event_type'].unique().tolist()
)
filtered = df[df['event_type'].isin(event_filter)]

# ── Row 1: Time of day and day of week ────────────────────────────────────────
col1, col2 = st.columns(2)

tod_order = ['Morning', 'Afternoon', 'Evening', 'Night']
dow_order = ['Monday', 'Tuesday', 'Wednesday',
             'Thursday', 'Friday', 'Saturday', 'Sunday']

with col1:
    st.subheader("Events by Time of Day")
    tod = filtered['time_of_day'].value_counts().reindex(tod_order).reset_index()
    tod.columns = ['time_of_day', 'count']
    fig = px.bar(
        tod,
        x     = 'time_of_day',
        y     = 'count',
        color = 'time_of_day',
        color_discrete_sequence = ['#FFD93D', '#FF6B6B', '#6BCB77', '#4D96FF'],
        text  = tod['count'].apply(lambda x: f'{x:,}')
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Events by Day of Week")
    dow = filtered['day_of_week'].value_counts().reindex(dow_order).reset_index()
    dow.columns = ['day', 'count']
    dow['is_weekend'] = dow['day'].isin(['Saturday', 'Sunday'])
    fig2 = px.bar(
        dow,
        x     = 'day',
        y     = 'count',
        color = 'is_weekend',
        color_discrete_map = {False: '#4C72B0', True: '#DD8452'},
        text  = dow['count'].apply(lambda x: f'{x:,}'),
        labels = {'is_weekend': 'Weekend'}
    )
    fig2.update_traces(textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)

# ── Heatmap ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Event Heatmap — Day × Time of Day")

heat = pd.crosstab(
    filtered['day_of_week'],
    filtered['time_of_day']
).reindex(dow_order)[tod_order]

fig3 = px.imshow(
    heat,
    color_continuous_scale = 'YlOrRd',
    aspect = 'auto',
    labels = dict(color='Event Count'),
    text_auto = True
)
fig3.update_layout(height=400)
st.plotly_chart(fig3, use_container_width=True)

# ── CTR by time ───────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Click Rate by Time of Day")
    ctr_tod = filtered.groupby('time_of_day')['is_click'].mean().reindex(tod_order).reset_index()
    ctr_tod.columns = ['time_of_day', 'ctr']
    ctr_tod['ctr_pct'] = ctr_tod['ctr'] * 100
    fig4 = px.line(
        ctr_tod,
        x       = 'time_of_day',
        y       = 'ctr_pct',
        markers = True,
        labels  = {'ctr_pct': 'CTR (%)'}
    )
    fig4.update_traces(line_color='coral', line_width=3, marker_size=10)
    st.plotly_chart(fig4, use_container_width=True)

with col2:
    st.subheader("Click Rate by Day of Week")
    ctr_dow = filtered.groupby('day_of_week')['is_click'].mean().reindex(dow_order).reset_index()
    ctr_dow.columns = ['day', 'ctr']
    ctr_dow['ctr_pct']    = ctr_dow['ctr'] * 100
    ctr_dow['is_weekend'] = ctr_dow['day'].isin(['Saturday', 'Sunday'])
    fig5 = px.bar(
        ctr_dow,
        x     = 'day',
        y     = 'ctr_pct',
        color = 'is_weekend',
        color_discrete_map = {False: '#4C72B0', True: '#DD8452'},
        labels = {'ctr_pct': 'CTR (%)', 'is_weekend': 'Weekend'}
    )
    st.plotly_chart(fig5, use_container_width=True)

# ── Weekly trend ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Weekly Event Trend")

filtered['week'] = filtered['timestamp'].dt.to_period('W').astype(str)
weekly = filtered.groupby('week').agg(
    total_events = ('event_id', 'count'),
    click_rate   = ('is_click', 'mean')
).reset_index()
weekly['click_rate_pct'] = weekly['click_rate'] * 100

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x    = weekly['week'],
    y    = weekly['total_events'],
    name = 'Total Events',
    fill = 'tozeroy',
    line = dict(color='steelblue', width=2)
))
fig6.update_layout(
    xaxis_title = 'Week',
    yaxis_title = 'Event Count',
    hovermode   = 'x unified'
)
st.plotly_chart(fig6, use_container_width=True)