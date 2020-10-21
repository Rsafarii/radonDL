import time
import sys
import pandas as pd
from numpy import asarray
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TKAgg',warn=False, force=True)

st.beta_set_page_config(
 page_title="DEEP RADON ANALYSER!",
  layout="centered"
)
@st.cache(suppress_st_warning=True,persist=False,allow_output_mutation=True) 
def PASS():
	passw = st.sidebar.text_input('Enter password')
	if passw == 'r.faghihi':
		pass
	else :
		st.stop()
PASS()
#@st.cache(suppress_st_warning=True,persist=False,allow_output_mutation=True) 
def run() :
	st.sidebar.title('Radon Deep  Analyser')
	col1, col2 = st.sidebar.beta_columns(2)
	with col1:
		st.header(" ")
		st.image("1.jpg", use_column_width=False)

	with col2:
		st.header(" ")
		st.image("2.jpeg", use_column_width=False) 


	st.sidebar.markdown('** Step 1 :**Available information ....')
	time.sleep(1)
	V = st.sidebar.selectbox('Level of ventilation:',
				( 'None','Low',
                    		'Normal', 'High'),index=2)
	UD_0 = st.sidebar.slider('Outdoor Tempereture(C):', -10, 35, 1)            		
	ID_0 = st.sidebar.slider('Indoor Tempereture(C):', UD_0, 35, 25)
	R_0 = st.sidebar.slider('Radon Concentration(Bq/m3):', 5, 2000, 20)
	st.write(' ')
	st.write(' ')
	st.sidebar.markdown('** Step 2 :**Model Training... ')
	from RASA_FIT import RASA_FIT
	my0 = RASA_FIT(V, ID_0, UD_0, R_0)
	m = my0[0]
	y0 = my0[1]
	with st.spinner('Wait for it...'):
			time.sleep(1)
	#st.success('Training is done!')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	x =np.arange(0, 30)
	y = m*x+y0
	plt.subplots(figsize=(8, 4))
	plt.plot(x,y, 'go')
	plt.title('Fitting Model ')
	plt.ylabel('Radon concentration(Bq/m3)')
	plt.xlabel('Indoor-Outdoor Temperuter difference(C)')
	st.pyplot() 
	return my0
my0=run()
m = my0[0]
y0 = my0[1]
st.sidebar.markdown('** Step 3 :**Model Testing... ')

UD_1 = st.sidebar.slider('Please specify the Outdoor Tempereture(C):', -10, 35, 5)
ID_1 = st.sidebar.slider('Please specify the Indoor Tempereture(C):', UD_1, 35, 25)	

with st.spinner('Wait for it...'):
	r=m*abs(ID_1-UD_1)+y0
st.success('Prediction is done!')
st.sidebar.write('Prediction  is', round(r, 3), 'Bq/m3')
st.write('Prediction  is', round(r, 3), 'Bq/m3')
st.empty()
#st.sidebar.button('Click here to RUN again!')
#st.button('Click here to RUN again!')
