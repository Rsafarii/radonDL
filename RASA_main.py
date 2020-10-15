
import time
import sys
import pandas as pd
from sklearn import preprocessing
from numpy import asarray
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('TKAgg',warn=False, force=True)
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error

st.beta_set_page_config(
 page_title="DEEP RADON ANALYSER!",
  layout="centered",
 initial_sidebar_state="collapsed")
 
st.title('RADON DEEP ANALYSER ')
col1, col2 = st.beta_columns(2)
with col1:
	st.header(" ")
	st.image("1.jpg", use_column_width=False)

with col2:
	st.header(" ")
	st.image("2.jpeg", use_column_width=False) 
st.markdown('** Step 1 :** Testing Parameters of DEEP LEARNING Model...')

st.info("Please input Parameters for model testing...")
time.sleep(1)
st.info("Suggestion: dont change default values to evaluate model with real TEST data!")
chek_df_ts = st.radio(
    "whats your choice for Testing parameters:",
    ('Default', 'Custom'))
if chek_df_ts == 'Default' :
	TB ='Family_house'
	F= 'Basement'
	TR= 'Basement'
	G= 'Holocene_deposits'
	V= 'Manual'
	O= 'Occasional'
	T= 15.8
	P= 98.63
	DM= 96
	H= 61.00
	st.write('You selected the default values for the Testing parameters')
	with st.beta_expander("See the default values of  Testing parameters..."):
		st.write('Type of Building:',TB)
		st.write('Floor:', F)
		st.write('Type of Room:', TR)
		st.write('Ventilation:',V)
		st.write('Occupation:', O)
		st.write('Simplified Geology:', G)
		st.write('Duration of Measurement(hr):',DM ,'hr')
		st.write('Pressure:',P , 'kPa')
		st.write('Humidity:',H , '%')
		st.write('Temperature:',T , 'C')
else :	
	TB = st.selectbox('Type of Building:',
			( 'Family_house','Faculty_building',
                    	'Wine_cellar_house', 'Residential_building',
                    	'Weekend_cottage', 'Office_building'),index=0)
	F = st.selectbox('Floor:',
			('Basement', '1st', '3rd',
                    	'Ground floor'))
	TR = st.selectbox('Type of Room:',
			('Basement','Storage', 'Laboratory',
                    	'office', 
                    	'Bedroom', "BasementStorage",
                    	'Hall', 'Kitchen storage', 'Dining hall', 
                    	'BasementGarage', 'Living room', 'Garage')) 
	V = st.selectbox('Ventilation:',
			('Manual','None', 'Mixed'))
	O= st.selectbox('Occupation:',
			('Occasional','Low', 'High'))
	G= st.selectbox('Simplified Geology:',
			('Holocene_deposits', 'Neogene_clastics',
                   	 'Triassic_carbonates'))

	DM = st.number_input('Duration of Measurement(hr):', min_value=24, max_value=300, value=96, step=1)
	P = st.number_input('Pressure (kPa):', min_value=float(80), max_value=float(120), value=98.63, step=0.01)
	H = st.number_input('Humidity (%):', min_value=float(20), max_value=float(95), value=float(61), step=0.1)
	T = st.number_input('Temperature (C):', min_value=float(10), max_value=float(40), value=15.8, step=0.1)
condation3 =  st.button('Click here after selections')
if condation3 !=True :
	st.stop()
else :
	pass
st.markdown('** Step 2:** Preprocessing of Training data...')
with st.spinner('Wait for it...'):
	from RASA_dataProc import RASA_dataProc
	DATA = RASA_dataProc()
	X_train = DATA[0]
	Y_train = DATA[1]
	X_test =DATA[2]
	Y_test =DATA[3]
	column_trf =DATA[4]
	min_max_scaler= DATA[5]
	Dataset1 =DATA[6]
	my_bar = st.progress(0)
	for percent_complete in range(100):
		time.sleep(0.05)
		my_bar.progress(percent_complete + 1)
st.success('Preprocessing is done!')
#print(X_train.shape,  X_test.shape, Y_train.shape,  Y_test.shape)

st.write('')
st.write('')
time.sleep(1)
st.markdown('** Step 3:** Finding best Model...')
with st.spinner('Wait for it...'):
	from RASA_modelCom import RASA_modelCom
	MODEL_ANALYS = RASA_modelCom(X_train, Y_train)
	model_results = MODEL_ANALYS[0]
	model_names =MODEL_ANALYS[1]
	Min_model_real_name =MODEL_ANALYS[2]
	best_model = Min_model_real_name
	my_bar = st.progress(0)
	for percent_complete in range(100):
		time.sleep(0.05)
		my_bar.progress(percent_complete + 1)
st.success('Best model has been Found!')
from RASA_plotModelcom import RASA_plotModelcom
RASA_plotModelcom(model_results,model_names)
st.write('Best model is: ', best_model )
st.write('')
st.write('')
time.sleep(1)
st.markdown('** Step 4:**Fit Training data to the best Model ...')
with st.spinner('Wait for it...'):
	from RASA_train import RASA_train
	RASA_train(best_model,X_train,Y_train)
	my_bar = st.progress(0)
	for percent_complete in range(100):
		time.sleep(0.01)
		my_bar.progress(percent_complete + 1)
st.success('Fitting is done!')
st.write('')
st.write('')
time.sleep(1)

st.markdown('** Final step:** Model Testing... ...')
Test_Dataset = pd.DataFrame([[TB, F, TR, V, O, DM, G, P, H, T, 47, 30.2]],
index=[22],
columns=['Type of Building', 'Floor', 'Type of Room', 'Ventilation',
       'Occupation', 'Measurement(Duration (hours))', 'Simplified Geology',
       'Pressure (kPa)', 'Humidity (%)', 'Temperature (C)', 'Radon_AM (Bq/m3)',
       'Radon_SD(Bq/m3)'])
       
condationTEST =1
if condationTEST :
	if (asarray(Test_Dataset) == asarray(Dataset1.loc[22:23, :])).all() :
    		TEST =0
    		yhat = best_model.predict(X_test[TEST].reshape(1,36))
    		print('Prediction(Bq/m3)', yhat)
    		print('Real value (Bq/m3)', Y_test[TEST])
    

	else :
    		TEST =1
    		test = column_trf.transform(Test_Dataset)
    		test_min_max= min_max_scaler.transform(test[0:1, 0:36])
    		yhat = best_model.predict(test_min_max[0:1, 0:36].reshape(1,36))
    		print('Prediction(Bq/m3)', yhat)
	
	time.sleep(2)
	with st.beta_expander("See  values of  Testing parameters:"):
		st.write('Your selection for Type of Building:', TB)
		st.write('Your selection for Type of Room:', TR)
		st.write('Your selection for Floor:', F)
		st.write('Your selection for Ventilation:', V)
		st.write('Your selection for Occupation:', O)
		st.write('Your selection for Simplified Geology:', G)
		st.write('Your selection for Duration of Measurement:', DM, 'hr')
		st.write('Your selection for Pressure:', P, 'kPa')
		st.write('Your selection for Humidity:', H, '%')
		st.write('Your selection for Temperature :', T, 'C')
with st.spinner('Wait for it...'):
	time.sleep(20)
st.success('Prediction is done!')
st.write(' ')
st.write(' ')
st.write(' ')
st.write(' ')
if TEST :
	st.write('Prediction about radon Concentration is', yhat[0], 'Bq/m3')
else :
	st.write('Prediction about radon Concentration is', yhat[0], 'Bq/m3')
	st.write('Real value of  radon Concentration is ', Y_test[TEST], 'Bq/m3')
#st.balloons()
st.empty()
st.write(' ')
st.write(' ')
st.button('Clic here to RUN again!')
	




	

	

		
	
	
	
	
	

    	
    



	
 
 







	
