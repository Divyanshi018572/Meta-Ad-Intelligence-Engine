import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Cluster Results", layout="wide")

@st.cache_data
def load_data():
    import os
    ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ad_metrics = pd.read_csv(os.path.join(ROOT, 'data', 'processed', 'ad_metrics_clustered.csv'))
    return ad_metrics

ad_metrics = load_data()

# Cluster names
cluster_names = {
    0: 'High Budget Engagers',
    1: 'Expensive Converters',
    2: 'All Round Stars',
    3: 'Budget Converters',
    4: 'Ultra Efficient'
}
ad_metrics['cluster_name'] = ad_metrics['cluster'].map(cluster_names)

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

st.title("🎯 Cluster Results")
st.markdown("K-Means segmentation of 200 ads into 5 performance groups")
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

for i, col in enumerate([col1, col2, col3, col4, col5]):
    cluster_data = ad_metrics[ad_metrics['cluster'] == i]
    col.metric(
        label = cluster_names[i],
        value = f"{len(cluster_data)} ads",
        delta = f"CTR: {cluster_data['click_rate'].mean()*100:.1f}%"
    )

st.markdown("---")

# ── Cluster comparison charts ─────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Performance Metrics by Cluster")

    metric = st.selectbox(
        "Select metric to compare",
        ['click_rate', 'purchase_rate', 'engagement_rate', 'events_per_budget']
    )

    cluster_perf = ad_metrics.groupby('cluster_name')[metric].mean().reset_index()
    cluster_perf.columns = ['cluster', 'value']
    cluster_perf['value_pct'] = cluster_perf['value'] * 100

    fig = px.bar(
        cluster_perf.sort_values('value', ascending=False),
        x     = 'cluster',
        y     = 'value_pct' if 'rate' in metric else 'value',
        color = 'cluster',
        text  = cluster_perf.sort_values('value', ascending=False)['value_pct'].apply(
                    lambda x: f'{x:.3f}%') if 'rate' in metric else None,
        color_discrete_sequence = colors
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(showlegend=False,
                      xaxis_title='Cluster',
                      yaxis_title=metric.replace('_',' ').title())
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Budget vs Efficiency by Cluster")
    fig2 = px.scatter(
        ad_metrics,
        x          = 'total_budget',
        y          = 'events_per_budget',
        color      = 'cluster_name',
        size       = 'total_events',
        hover_data = ['ad_id', 'click_rate', 'purchase_rate'],
        color_discrete_sequence = colors,
        labels = {
            'total_budget'     : 'Total Budget ($)',
            'events_per_budget': 'Events per Dollar',
            'cluster_name'     : 'Cluster'
        }
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── PCA visualization ─────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Visualization in 2D (PCA)")

feature_cols = [
    'click_rate', 'purchase_rate', 'engagement_rate',
    'impression_rate', 'unique_users', 'events_per_budget', 'total_budget'
]

# Scale and apply PCA
X = ad_metrics[feature_cols].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    'PC1'          : X_pca[:, 0],
    'PC2'          : X_pca[:, 1],
    'cluster_name' : ad_metrics.loc[X.index, 'cluster_name'].values,
    'ad_id'        : ad_metrics.loc[X.index, 'ad_id'].values,
    'click_rate'   : ad_metrics.loc[X.index, 'click_rate'].values,
    'total_budget' : ad_metrics.loc[X.index, 'total_budget'].values
})

fig3 = px.scatter(
    pca_df,
    x          = 'PC1',
    y          = 'PC2',
    color      = 'cluster_name',
    hover_data = ['ad_id', 'click_rate', 'total_budget'],
    color_discrete_sequence = colors,
    labels = {'cluster_name': 'Cluster'},
    title  = f'PCA — PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% | '
             f'PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}%'
)
fig3.update_traces(marker=dict(size=12, line=dict(width=1, color='white')))
st.plotly_chart(fig3, use_container_width=True)

# ── Cluster detail table ──────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Cluster Detail Table")

selected_cluster = st.selectbox(
    "Select cluster to inspect",
    options = list(cluster_names.values())
)

cluster_data = ad_metrics[ad_metrics['cluster_name'] == selected_cluster]
st.write(f"**{len(cluster_data)} ads** in this cluster")
st.dataframe(
    cluster_data[[
        'ad_id', 'cluster_name', 'total_events',
        'click_rate', 'purchase_rate', 'engagement_rate',
        'total_budget', 'events_per_budget'
    ]].style.format({
        'click_rate'        : '{:.2%}',
        'purchase_rate'     : '{:.2%}',
        'engagement_rate'   : '{:.2%}',
        'total_budget'      : '${:,.0f}',
        'events_per_budget' : '{:.6f}'
    }),
    use_container_width=True
)