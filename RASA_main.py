import datetime 
from datetime import  timedelta
from wwo_hist import retrieve_hist_data
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
	passw = st.text_input('Enter password')
	if passw == 'r.faghihi':
		pass
	else :
		st.stop()
PASS()

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
	R_0 = st.sidebar.slider('Radon Concentration(Bq/m3):', 5, 2000, 20)
	st.write(' ')
	st.write(' ')
	
	from RASA_FIT import RASA_FIT
	my0 = RASA_FIT(V,UD_0, R_0)
	m = my0[0]
	y0 = my0[1]
	#with st.spinner('Wait for it...'):
			#time.sleep(1)
	#st.success('Training is done!')
	st.set_option('deprecation.showPyplotGlobalUse', False)
	x =np.arange(0, 30)
	y = m*x+y0
	return my0
my0=run()
m = my0[0]
y0 = my0[1]

di = st.date_input(
 "When's start point?",
(datetime.date.today()-timedelta(days=365)), datetime.date(2017, 12, 30),(datetime.date.today()-timedelta(days=365)))
Di=str(datetime.datetime.strptime(str(di), '%Y-%m-%d').strftime('%d-%b-%Y'))

df = st.date_input(
 "When's stop point?",
di+timedelta(days=363),di+timedelta(days=6*31),(datetime.date.today()-timedelta(days=1)))
Df=str(datetime.datetime.strptime(str(df), '%Y-%m-%d').strftime('%d-%b-%Y'))

api_key= '8764781779e0415daeb81317200511'

location= st.text_input("what's your city?", 'california')
location_list=[location]
if st.button('Click here to Continue...!'):
	pass
else:
	st.stop()
time.sleep(4)
st.sidebar.markdown('** Step 2 :**Model Training... ')
chek = 1
with st.spinner('Wait for it...'):
	while chek:
		try:
			hist_weather_data = retrieve_hist_data(api_key,
                                	location_list,
                                	Di,
                                	Df,
                                	24,
                                	location_label = False,
                                	export_csv = False,
                                	store_df = True)
			st.dataframe(hist_weather_data[0].head())
			chek =0

		except :
			pass

wDate = hist_weather_data[0]
wDate=wDate.dropna(how='any')
dates = pd.date_range(start='20171231', end='20210501',freq='M')
mean_data = []
Mean_data = []
Mean_value=[]
Date_value =[]
for i in range(0,dates.size-1):
        if np.any((dates[i+1]>= wDate['date_time']) & (dates[i]< wDate['date_time'])):
            	t2= wDate[(dates[i+1]>= wDate['date_time']) & (dates[i]< wDate['date_time'])]
            	mean_data.append(t2)
            	Mean_data.append(t2.loc[:,['date_time', 'maxtempC', 'mintempC']])
            	mean_v=int (((t2.loc[:,[ 'maxtempC', 'mintempC']]).astype('int').mean()).mean())
            	Date_value.append(str(t2.iloc[0,0].date().year) + '-' +str(t2.iloc[0,0].date().month))
            	Mean_value.append(mean_v) 
MEAN = np.array(Mean_value)
pr=m*(MEAN)+y0
st.sidebar.markdown('** Step 3 :**Model Testing... ')
with st.spinner('Wait for It...'):
	time.sleep(3)
st.success('Prediction is done!')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(Date_value,pr, 'go')
fig.autofmt_xdate()
ax.set_title('Fitting Model ')
ax.set_ylabel('Radon concentration(Bq/m3)')
ax.set_xlabel('Date')
st.pyplot() 
st.empty()
#st.sidebar.button('Click here to RUN again!')
#st.button('Click here to RUN again!')
st.sidebar.success('Prediction is done!')
