import streamlit as st
import datetime 
from datetime import  timedelta
import time
def RASA_TIME() :
    col1, col2 = st.beta_columns(2)
    with col1:
        di = st.date_input(
 "When's start point?",
(datetime.date.today()-timedelta(days=190)), datetime.date(2017, 12, 30),(datetime.date.today()-timedelta(days=185)))
        Di=str(datetime.datetime.strptime(str(di), '%Y-%m-%d').strftime('%d-%b-%Y'))

        df = st.date_input(
 "When's stop point?",
di+timedelta(days=180),di+timedelta(days=3*31),(datetime.date.today()-timedelta(days=10)))
        Df=str(datetime.datetime.strptime(str(df), '%Y-%m-%d').strftime('%d-%b-%Y'))
    with col2:
        location= st.text_input("what's your city?", 'california')
        V_F = st.selectbox('Level of ventilation:',
( 'None','Low',
'Normal', 'High'),index=1)
        
    return [Di, Df, location, V_F]