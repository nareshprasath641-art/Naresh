"""Entry point: streamlit run app.py."""
import streamlit as st
from src.dashboard import initialize

st.set_page_config(page_title='Student learning analytics', page_icon=':material/school:', layout='wide')
st.sidebar.title('Learning analytics')
st.sidebar.caption('K-Means · synthetic student data')
initialize()
pages = [
    ('home', 'Home', 'home'), ('dataset', 'Dataset Analysis', 'table_chart'),
    ('eda', 'Exploratory Data Analysis', 'query_stats'), ('kmeans', 'K-Means Analysis', 'hub'),
    ('clusters', 'Student Clusters', 'groups'), ('visualization', 'Cluster Visualization', 'scatter_plot'),
    ('prediction', 'Student Behaviour Prediction', 'person_search'), ('upload', 'CSV Upload', 'upload_file'),
]
st.navigation([st.Page(f'app_pages/{file}.py', title=title, icon=f':material/{icon}:') for file, title, icon in pages]).run()
st.caption('Synthetic demonstration only. Groups describe similarity, not ability, academic success, or a diagnosis.')
