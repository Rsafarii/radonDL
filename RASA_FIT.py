def RASA_FIT(V,  UD_0, R_0):
	from numpy import asarray
	import numpy as np
	import streamlit as st
	if V == 'None' :
		m= -1.35
	elif V == 'Low' :
		m = -0.28
	elif V == 'Normal' :
		m = 0.74
	else :
		m = 1.04
	y0=R_0-m*(UD_0)
	return [m, y0]
		
	