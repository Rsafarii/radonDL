def RASA_FIT(V_I,  UD_I, R_I):
	from numpy import asarray
	import numpy as np
	import streamlit as st
	if V_I == 'None' :
		m_I= -1.35
	elif V_I == 'Low' :
		m_I = -0.28
	elif V_I == 'Normal' :
		m_I = 0.74
	else :
		m_I = 1.04
	R_0_I=R_I-m_I*(UD_I)
	return  R_0_I
		
	