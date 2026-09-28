import plotly.express as px
import streamlit as st
from src.dashboard import context, table
from src.visualization import pca_chart, radar_chart
from src.preprocessing import FEATURES

bundle, students, summary = context()
st.title('Cluster Visualization')
coordinates = table('outputs/pca_students.csv')
centers = bundle['pca'].transform(bundle['kmeans_model'].cluster_centers_)
st.plotly_chart(pca_chart(coordinates, centers))
variance = bundle['metrics']['pca_explained_variance']
st.caption(f'First two components retain {sum(variance[:2]):.1%} of standardized variance. Clustering uses all ten features, not these two plotted coordinates.')
if st.checkbox('Show interactive 3D PCA'):
    st.plotly_chart(pca_chart(coordinates, centers, True))
feature = st.selectbox('Compare cluster averages', FEATURES, format_func=lambda name: name.replace('_', ' '))
st.plotly_chart(px.bar(summary, x='Learning_Behaviour', y=feature, color='Learning_Behaviour', title='Average ' + feature.replace('_', ' ') + ' by cluster'))
st.plotly_chart(radar_chart(summary))
