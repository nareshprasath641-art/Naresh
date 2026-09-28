import json
import plotly.express as px
import streamlit as st
from src.dashboard import table, download
from src.preprocessing import ROOT, FEATURES

st.title('Dataset Analysis')
raw = table('student_learning_behavior.csv')
st.write(f'Dataset shape: {raw.shape[0]:,} rows × {raw.shape[1]} columns')
st.dataframe(raw.head(), hide_index=True)
st.subheader('Missing values and data types')
st.dataframe(raw.isna().sum().rename('Missing values').to_frame().join(raw.dtypes.astype(str).rename('Data type')))
st.subheader('Descriptive statistics')
st.dataframe(raw.describe())
report = json.loads((ROOT / 'outputs/diagnostics.json').read_text())
st.write(f"Exact duplicate rows: {report['duplicate_rows']} · Repeated student IDs: {report['duplicate_ids']}")
st.subheader('Basic outlier analysis')
st.json(report['iqr_outlier_counts'])
st.caption('Counts outside 1.5 × IQR fences. Plausible extreme habits are retained; these counts are diagnostics, not automatic deletion rules.')
st.plotly_chart(px.imshow(raw[FEATURES].corr(), zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='Feature correlation heatmap'))
download(raw, 'student_learning_behavior.csv')
