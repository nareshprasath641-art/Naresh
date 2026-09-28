import plotly.express as px
import streamlit as st
from src.dashboard import table
from src.preprocessing import FEATURES

st.title('Exploratory Data Analysis')
raw = table('student_learning_behavior.csv')
feature = st.selectbox('Learning measurement', FEATURES, format_func=lambda name: name.replace('_', ' '))
left, right = st.columns(2)
with left:
    st.plotly_chart(px.histogram(raw, x=feature, nbins=30, title=feature.replace('_', ' ') + ' distribution'))
with right:
    st.plotly_chart(px.box(raw, y=feature, title='Spread and unusual observations'))
st.plotly_chart(px.scatter(raw, x=feature, y='Average_Test_Score', opacity=.4, title=feature.replace('_', ' ') + ' vs average test score'))
st.caption('Correlations reflect the synthetic generator and do not demonstrate causation.')
