import pandas as pd
import streamlit as st
from src.dashboard import context
from src.preprocessing import BOUNDS
from src.clustering import assign_students

bundle, students, summary = context()
st.title('Student Behaviour Prediction')
st.info('This is cluster assignment based on similarity to learned groups, not a supervised prediction of academic success. Age is collected for reference and does not affect assignment.')
st.caption('Hours are per day. Class interactions are per week. Marks and test scores are on a 0–100 scale.')
with st.form('student_form'):
    inputs = {'Student_ID': 'NEW_STUDENT'}
    columns = st.columns(2)
    for index, (name, (low, high)) in enumerate(BOUNDS.items()):
        with columns[index % 2]:
            default = float(students[name].median())
            if name in ['Age', 'Class_Interactions']:
                inputs[name] = st.number_input(name.replace('_', ' '), low, high, int(default), step=1)
            else:
                inputs[name] = st.number_input(name.replace('_', ' '), float(low), float(high), default, step=.1)
    submitted = st.form_submit_button('Analyze Learning Behaviour', type='primary')
if submitted:
    result = assign_students(pd.DataFrame([inputs]), bundle).iloc[0]
    st.subheader('Student Learning Behaviour')
    st.success(f'Cluster: {result.Cluster} · Category: {result.Learning_Behaviour}')
    profile = summary.loc[summary.Cluster == result.Cluster].iloc[0]
    st.write(f'The nearest center belongs to a group averaging {profile.Study_Hours_Per_Day:.1f} study hours/day, {profile.Attendance_Percentage:.1f}% attendance, and {profile.Assignment_Completion_Percentage:.1f}% assignment completion. This is a relative description of the synthetic training population.')
