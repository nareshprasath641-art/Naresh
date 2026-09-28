import streamlit as st
from src.dashboard import upload_panel, table, download
from src.preprocessing import REQUIRED

st.title('CSV Upload')
st.write('Assign another dataset to the existing learned groups. Uploads stay in this session and do not retrain the model.')
st.caption('Required columns: ' + ', '.join(REQUIRED))
st.caption('Hours are per day; class interactions are per week; percentages and marks use 0–100. Empty numeric cells are accepted. Invalid text, infinity, and values outside supported ranges are rejected.')
download(table('student_learning_behavior.csv').head(10), 'student_upload_template.csv', 'Download example CSV')
uploaded = st.file_uploader('Student CSV', type=['csv'])
upload_panel(uploaded)
