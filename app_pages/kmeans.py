import plotly.express as px
import streamlit as st
from src.dashboard import context, table

bundle, students, summary = context()
metrics = bundle['metrics']
evaluation = table('outputs/k_evaluation.csv')
st.title('K-Means Analysis')
st.success(f"Selected K = {metrics['selected_k']}: maximum silhouette over K=2–10.")
left, right = st.columns(2)
with left:
    st.plotly_chart(px.line(evaluation, x='K', y='Inertia', markers=True, title='Elbow method'))
with right:
    st.plotly_chart(px.line(evaluation, x='K', y='Silhouette', markers=True, title='Silhouette score'))
with st.container(horizontal=True):
    for label, key in [('Silhouette ↑', 'silhouette'), ('Davies–Bouldin ↓', 'davies_bouldin'), ('Calinski–Harabasz ↑', 'calinski_harabasz'), ('Inertia', 'inertia')]:
        st.metric(label, f'{metrics[key]:.4f}', border=True)
st.write('Arrows indicate the generally preferred direction. Inertia falls as K increases and cannot select K by itself. Scores measure geometric separation, not educational validity.')
st.write('The continuous synthetic habits can favour broad groups; four clusters are not forced when silhouette supports another K.')
st.dataframe(evaluation, hide_index=True)
st.plotly_chart(px.bar(summary, x='Learning_Behaviour', y='Student_Count', title='Cluster distribution'))
