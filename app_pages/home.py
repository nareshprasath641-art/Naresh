import streamlit as st
from src.dashboard import context

bundle, students, summary = context()
metrics = bundle['metrics']
st.caption('STUDENT ANALYTICS / OVERVIEW')
st.title('Student Learning Behaviour Clustering using K-Means')
st.write('Explore learning habits, discover similar students, and understand the characteristics of each learned group.')
with st.container(horizontal=True):
    st.metric('Students', f'{len(students):,}', border=True)
    st.metric('Learning features', len(metrics['features']), border=True)
    st.metric('Selected clusters', metrics['selected_k'], border=True)
    st.metric('Silhouette score', f"{metrics['silhouette']:.3f}", border=True)
with st.container(border=True):
    st.subheader('How it works')
    st.write('K-Means groups standardized student measurements around nearby centers. Median imputation fills missing learning measurements before scaling. We compare K=2 through 10 and select the highest silhouette score, supported by the elbow plot.')
    st.write('Age and Student_ID are retained for reference and excluded from clustering. Behaviour labels are assigned after training by comparing engagement-related cluster averages.')
st.info('All records are generated. These results demonstrate a workflow and do not establish real-world student categories or educational effectiveness.')
st.dataframe(summary[['Cluster', 'Learning_Behaviour', 'Student_Count']], hide_index=True)
