
def RASA_modelCom(X_train, Y_train) :
	import pandas as pd
	from sklearn import preprocessing
	from numpy import asarray
	import numpy as np
	import streamlit as st
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
	folds   = 10
	seed =9
	metric  = "neg_mean_squared_error"

	models = {}
	models["Linear"]        = LinearRegression()
	models["Lasso"]         = Lasso()
	models["ElasticNet"]    = ElasticNet()
	models["KNN"]           = KNeighborsRegressor()
	models["DecisionTree"]  = DecisionTreeRegressor()
	models["SVR"]           = SVR()
	models["AdaBoost"]      = AdaBoostRegressor()
	models["GradientBoost"] = GradientBoostingRegressor()
	models["RandomForest"]  = RandomForestRegressor()
	models["ExtraTrees"]    = ExtraTreesRegressor()

	# 10-fold cross validation for each model
	itr=0
	model_results = []
	model_mean_results = []
	model_names   = []
	for model_name in models:

		model   = models[model_name]
		k_fold  = KFold(n_splits=folds, random_state=seed, shuffle=True)
		results = cross_val_score(model, X_train, Y_train, cv=k_fold, scoring=metric)
		model_mean_results.append(round(results.mean(), 3))
		model_results.append(results)
		model_names.append(model_name)
		print("{}: {}, {}".format(model_name, round(results.mean(), 3), round(results.std(), 3)))
		if itr ==0 :
			Min = round(results.mean(), 3)
		else :
			Min = max(Min ,round(results.mean(), 3))
		itr = itr+1
	print(model_mean_results, Min)
	Min_index = model_mean_results.index(Min)
	Min_model_name = model_names[Min_index]
	Min_model_real_name = models[Min_model_name]
	print(Min_model_real_name)
	return [model_results,model_names,Min_model_real_name ]
