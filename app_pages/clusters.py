import streamlit as st
from src.dashboard import context, download

bundle, students, summary = context()
st.title('Student Clusters')
st.dataframe(summary, hide_index=True)
st.caption('Profiles use median-imputed training measurements. Labels are relative descriptions ranked by study, attendance, assignments, interactions, and preparation; they are not ground-truth student types.')
cluster = st.selectbox('Select a cluster', summary.Cluster.tolist(), format_func=lambda value: f"Cluster {value}: {summary.loc[summary.Cluster == value, 'Learning_Behaviour'].iloc[0]}")
profile = summary.loc[summary.Cluster == cluster].iloc[0]
st.write(f"This group averages {profile.Study_Hours_Per_Day:.1f} study hours/day, {profile.Attendance_Percentage:.1f}% attendance, {profile.Assignment_Completion_Percentage:.1f}% assignment completion, and {profile.Average_Test_Score:.1f} test marks.")
selected = students[students.Cluster == cluster]
st.write(f'{len(selected):,} students in this cluster')
st.dataframe(selected, hide_index=True)
download(selected, f'cluster_{cluster}_students.csv', 'Download selected students')
download(students, 'clustered_students.csv', 'Download all clustered students')
download(summary, 'cluster_summary.csv', 'Download cluster summary')
