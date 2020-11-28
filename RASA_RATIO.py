def RASA_RATIO(V_I,  V_F, R_0_I):
	from numpy import asarray
	import numpy as np
	import streamlit as st
	r=[50, 52, 30, 20]
	V = ['None', 'Low', 'Normal', 'High']
	Ratio = r[V.index(V_F)]/r[V.index(V_I)]
	R_0_F = R_0_I * Ratio
	return R_0_F
		
	