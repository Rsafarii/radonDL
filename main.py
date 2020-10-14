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
#import tensorflow as tf 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


#mt= tf.keras.metrics.SparseTopKCategoricalAccuracy(
    k=4, name='sparse_top_k_categorical_accuracy', dtype=None)

 
st.beta_set_page_config(
 page_title="DEEP RADON ANALYSER!",
  page_icon="images1.ico",layout="centered",
 initial_sidebar_state="collapsed")
 
st.title('..........RADON DEEP ANALYSER.......... ')
col1, col2 = st.beta_columns(2)
with col1:
	st.header(" ")
	st.image("1.jpg ", use_column_width=False)

with col2:
	st.header(" ")
	st.image("2.jpeg", use_column_width=False)

#st.header('Train DEEP LEARNING Model')
st.markdown('** Step 1 :** Training Parameters of DEEP LEARNING Model...')
st.info("Please input Parameters for model Training...")
time.sleep(1)
st.info("Suggestion:  Defaults are optimal values! ")
time.sleep(1)
chek_df_tr = st.radio(
    "whats your choice for Training parameters:",
    ('Default', 'Custom'))
if chek_df_tr == 'Default' :
	epochs =200
	Bach=22
	st.write('You selected the default values for the training parameters')
	with st.beta_expander("See the default values of  train parameters..."):
		st.write('Epochs:200')
		st.write('Bach size:22')
	
else :

	#epochs = st.slider('please specify the number of Epochs:', 100, 500, 200)
	#st.write("Epochs : ", epochs)
	epochs = st.number_input('please specify the number of Epochs:', min_value=10, max_value=1000, value=200, step=10)
	Bach = st.slider('please specify the Bach size', 12, 22, 22)
st.write('The current Epochs is: ', epochs)
st.write('The current Bach size is: ', Bach)

st.markdown('** Step 2 :** Testing Parameters of DEEP LEARNING Model...')

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

#df1 = pd.read_excel('DATA_Anita2020.xlsx')
#dataset1=df1
#print(df1.shape, type(df1))
#dataset1 = df1.values
#dataset1_asarray =asarray(dataset1)
Dataset1 = pd.DataFrame([['Faculty_building', 'Basement', 'Storage', 'None', 'Low', 99,
        'Holocene_deposits', 99.02, 69.3, 20.6, 286.1, 99.0],
       ['Faculty_building', 'Basement', 'Laboratory', 'Manual',
        'Occasional', 144, 'Holocene_deposits', 99.77, 31.9, 24.3, 62.6,
        24.4],
       ['Faculty_building', '1st', 'office', 'Manual', 'High', 386,
        'Holocene_deposits', 98.73, 53.8, 27.3, 52.2, 39.8],
       ['Faculty_building', '3rd', 'office', 'Manual', 'High', 146,
        'Holocene_deposits', 99.29, 57.7, 26.0, 57.1, 47.8],
       ['Family_house', 'Basement', 'Basement', 'Manual', 'Occasional',
        68, 'Holocene_deposits', 99.5, 69.1, 26.6, 40.2, 22.2],
       ['Family_house', '1st', 'Bedroom', 'Manual', 'High', 143,
        'Holocene_deposits', 99.51, 62.9, 28.9, 20.8, 17.0],
       ['Wine_cellar_house', 'Basement', 'BasementStorage', 'Manual',
        'Low', 218, 'Neogene_clastics', 100.05, 54.0, 25.6, 32.0, 19.8],
       ['Family_house', 'Basement', 'Storage', 'Manual', 'Occasional',
        354, 'Holocene_deposits', 99.86, 51.0, 14.2, 32.0, 18.9],
       ['Family_house', 'Ground floor', 'Hall', 'Manual', 'High', 168,
        'Holocene_deposits', 99.74, 49.0, 15.6, 27.0, 14.3],
       ['Wine_cellar_house', 'Basement', 'BasementStorage', 'Manual',
        'Occasional', 288, 'Neogene_clastics', 98.53, 89.5, 10.3, 90.4,
        77.0],
       ['Wine_cellar_house', 'Ground floor', 'Kitchen storage', 'Manual',
        'High', 288, 'Neogene_clastics', 97.31, 93.9, 10.9, 31.7, 29.3],
       ['Residential_building', '1st', 'Bedroom', 'Manual', 'High', 164,
        'Holocene_deposits', 100.23, 43.0, 23.3, 50.0, 32.3],
       ['Family_house', '1st', 'Dining hall', 'Manual', 'High', 96,
        'Holocene_deposits', 98.77, 62.0, 18.0, 31.0, 20.8],
       ['Family_house', 'Basement', 'BasementGarage', 'Manual', 'Low',
        96, 'Neogene_clastics', 99.21, 44.3, 22.5, 46.2, 27.8],
       ['Family_house', 'Ground floor', 'Living room', 'Manual', 'High',
        96, 'Neogene_clastics', 99.08, 59.4, 22.8, 21.7, 15.2],
       ['Wine_cellar_house', 'Basement', 'Storage', 'Manual',
        'Occasional', 75, 'Neogene_clastics', 98.52, 75.0, 14.0, 18.0,
        10.5],
       ['Residential_building', 'Ground floor', 'Living room', 'Manual',
        'High', 212, 'Holocene_deposits', 98.84, 55.6, 22.6, 53.5, 41.8],
       ['Weekend_cottage', 'Ground floor', 'Storage', 'Manual', 'Low',
        285, 'Triassic_carbonates', 99.53, 53.0, 23.6, 56.0, 47.8],
       ['Residential_building', 'Ground floor', 'Living room', 'Manual',
        'High', 96, 'Holocene_deposits', 99.67, 63.0, 27.4, 57.0, 45.3],
       ['Office_building', 'Ground floor', 'Office', 'Mixed', 'High', 96,
        'Holocene_deposits', 99.02, 49.6, 25.2, 32.2, 24.4],
       ['Family_house', 'Basement', 'Garage', 'Manual', 'Low', 126,
        'Holocene_deposits', 99.38, 51.0, 25.6, 47.0, 32.5],
       ['Family_house', 'Basement', 'BasementStorage', 'Manual', 'Low',
        286, 'Neogene_clastics', 99.69, 39.0, 23.6, 70.0, 50.1],
       ['Family_house', 'Basement', 'Basement', 'Manual', 'Occasional',
        96, 'Holocene_deposits', 98.63, 61.0, 15.8, 47.0, 30.2]],
     index=range(0,23,1),
     columns=['Type of Building', 'Floor', 'Type of Room', 'Ventilation',
       'Occupation', 'Measurement(Duration (hours))', 'Simplified Geology',
       'Pressure (kPa)', 'Humidity (%)', 'Temperature (C)', 'Radon_AM (Bq/m3)',
       'Radon_SD(Bq/m3)'])

column_trf = make_column_transformer((OneHotEncoder(),['Type of Building',
        'Floor', 'Type of Room', 'Ventilation',
       'Occupation', 'Simplified Geology',
        ]),remainder='passthrough')
onehot_dataset = column_trf.fit_transform(Dataset1)
data1 = asarray(onehot_dataset)
DATA = data1
print(DATA.shape)
NORM_DATA = DATA

X1 = NORM_DATA[:,0:36]
X1 = X1.reshape(23,36)
Y1 = NORM_DATA[:,36:37]
Y1 = Y1.reshape(23,1)

min_max_scaler = preprocessing.MinMaxScaler()
min_max_NORM_DATA_X = min_max_scaler.fit_transform(X1)

X = min_max_NORM_DATA_X
Y = Y1
print(X.shape)
print(Y.shape)

X_train = X[0:21,:]
Y_train = Y[0:21,:]
X_val = X[21:22,:]
Y_val = Y[21:22,:]
X_test = X[22:23,:]
Y_test = Y[22:23,:]
print(X_train.shape, X_val.shape, X_test.shape, Y_train.shape, Y_val.shape, Y_test.shape)


#condation1 =  st.button('Clic here to start Training')
condation1 = True


if condation1 != True :
	st.stop()
else :
	st.markdown('** Step 3 :** Model Training...')
	with st.spinner('Wait for it...'):
		cond2 = True
		while cond2 :
			try :
	        		del model
			except :
	        		pass
			model = Sequential([ 
			    	Dense(200, activation='relu', input_shape=(36,)),
			    	#Dropout(.1),
			    	Dense(200, activation='relu'),
			   	# Dropout(.1),
			    	Dense(200, activation='relu'),
			    	Dense(200, activation='relu'),
			    	Dense(200, activation='relu'),
			    	Dense(200, activation='relu'),
			    	Dense(200, activation='relu'),
			    	Dense(200, activation='relu'),
			    	#Dense(800, activation='tanh'),
			   	#Dropout(.1),
			    	#Dense(12, activation='sigmoid'),
	                    	Dense(1, activation='relu'),])
			model.compile(optimizer='adam',
		               	loss='mse',
		           	metrics=['mse','mae'])
			FIT_MODEL = model.fit(X_train, Y_train,
		           	batch_size=20, epochs=200,
                           	validation_data=(X_val, Y_val), verbose=2)
			x = FIT_MODEL.history['loss'][-1]
			if x < 1 :
                		cond2=False
			#st.write(FIT_MODEL.history['loss'][-1])
			#cond2 = False
		else:
	
			my_bar = st.progress(0)
			for percent_complete in range(100):
				time.sleep(0.05)
				my_bar.progress(percent_complete + 1)
			st.success('Train is done!')
			st.set_option('deprecation.showPyplotGlobalUse', False)
			time.sleep(2)
			
			plt.subplots(figsize=(7, 3))
			plt.plot(FIT_MODEL.history['loss'])
			plt.title('Model loss')
			plt.ylabel('Loss')
			plt.xlabel('Epoch')
			st.pyplot() 
			
			#st.line_chart(FIT_MODEL.history['loss'])

time.sleep(1)
st.markdown('** Step 4 :** Model Testing...')
Test_Dataset = pd.DataFrame([[TB, F, TR, V, O, DM, G, P, H, T, 47, 30.2]],
index=[22],
columns=['Type of Building', 'Floor', 'Type of Room', 'Ventilation',
       'Occupation', 'Measurement(Duration (hours))', 'Simplified Geology',
       'Pressure (kPa)', 'Humidity (%)', 'Temperature (C)', 'Radon_AM (Bq/m3)',
       'Radon_SD(Bq/m3)'])
#st.dataframe(Test_Dataset,width=4000, height=4000)
#st.table(Test_Dataset)

#condationTEST =  st.button('Clic here to evaluate **prediction** of medol about **test data**')
condationTEST =1
if condationTEST :
	if (asarray(Test_Dataset) == asarray(Dataset1.loc[22:23, :])).all() :
    		TEST =0
    		yhat = model.predict(X_test[TEST].reshape(1,36))
    		print('Prediction(Bq/m3)', yhat[0][0])
    		print('Real value (Bq/m3)', Y_test[TEST][0])
    

	else :
    		TEST =1
    		test = column_trf.transform(Test_Dataset)
    		test_min_max= min_max_scaler.transform(test[0:1, 0:36])
    		yhat = model.predict(test_min_max[0:1, 0:36].reshape(1,36))
    		print('Prediction(Bq/m3)', yhat[0][0])
	st.write('testing data: ')
	time.sleep(2)
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
		time.sleep(10)
	st.success('Prediction is done!')
	st.write(' ')
	st.write(' ')
	st.write(' ')
	st.write(' ')
	if TEST :
		st.write('             Prediction about radon Concentration is', yhat[0][0], 'Bq/m3')
	else :
		st.write('             Prediction about radon Concentration is', yhat[0][0], 'Bq/m3')
		st.write('             Real value of  radon Concentration is ', Y_test[TEST][0], 'Bq/m3')
#st.balloons()
st.empty()
st.write(' ')
st.write(' ')
st.button('Clic here to RUN again!')
	

	

		
	
	
	
	
	

    	
    



	
 
 







	
