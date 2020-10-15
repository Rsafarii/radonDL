def RASA_dataProc():
	
	import pandas as pd
	from sklearn import preprocessing
	from numpy import asarray
	import numpy as np
	import streamlit as st
	from sklearn.compose import make_column_transformer
	from sklearn.preprocessing import OneHotEncoder
	
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
	Y1 = Y1.reshape(23,)

	min_max_scaler = preprocessing.MinMaxScaler()
	min_max_NORM_DATA_X = min_max_scaler.fit_transform(X1)

	X = min_max_NORM_DATA_X
	Y = Y1
	print(X.shape)
	print(Y.shape)

	X_train = X[0:22,:]
	Y_train = Y[0:22]
	X_test = X[22:23,:]
	Y_test = Y[22:23]
	return [X_train,Y_train, X_test,Y_test,column_trf, min_max_scaler, Dataset1 ]