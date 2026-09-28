"""Shared cached data loading and CSV upload interface."""
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from src.preprocessing import ROOT
from src.clustering import load_bundle, assign_students, csv_bytes

@st.cache_resource(show_spinner='Preparing student analytics...', max_entries=2)
def cached_bundle(dataset_stamp, model_stamp):
    return load_bundle()

def initialize():
    dataset = ROOT / 'student_learning_behavior.csv'
    model = ROOT / 'models/kmeans_model.pkl'
    try:
        bundle = cached_bundle(dataset.stat().st_mtime_ns if dataset.exists() else 0, model.stat().st_mtime_ns if model.exists() else 0)
    except (ValueError, OSError, EOFError) as error:
        st.error(f'Unable to prepare the project: {error}. Run python src/clustering.py to rebuild.')
        st.stop()
    st.session_state['bundle'] = bundle

@st.cache_data(max_entries=12)
def read_csv(name, stamp):
    return pd.read_csv(ROOT / name)

def table(name):
    return read_csv(name, (ROOT / name).stat().st_mtime_ns)

def context():
    return st.session_state['bundle'], table('outputs/clustered_students.csv'), table('outputs/cluster_summary.csv')

def download(frame, name, label='Download CSV'):
    st.download_button(label, csv_bytes(frame), name, mime='text/csv')

def upload_panel(uploaded):
    if uploaded is None:
        return
    try:
        frame = pd.read_csv(uploaded)
        assigned = assign_students(frame, st.session_state['bundle'])
    except (ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        st.error(str(error))
        return
    st.success(f'Assigned {len(assigned):,} students to existing clusters.')
    st.caption(f'{int(frame.isna().sum().sum())} missing cells detected. Behaviour features use saved training medians; original empty cells remain in the export.')
    if frame.Student_ID.duplicated().any():
        st.warning('Repeated Student_ID values are retained as separate uploaded rows.')
    counts = assigned.Learning_Behaviour.value_counts().rename_axis('Learning behaviour').reset_index(name='Students')
    st.plotly_chart(px.bar(counts, x='Learning behaviour', y='Students', title='Uploaded student distribution'))
    st.dataframe(assigned, hide_index=True)
    download(assigned, 'uploaded_clustered_students.csv')
