def RASA_FIT(V, ID_0, UD_0, R_0):
	from numpy import asarray
	import numpy as np
	import streamlit as st
	if V == 'None' :
		m= 1.78
	elif V == 'Low' :
		m = 0.39
	elif V == 'Normal' :
		m = -1.16
	else :
		m = -1.42
	y0=R_0-m*abs(ID_0 - UD_0)
	return [m, y0]
		
	