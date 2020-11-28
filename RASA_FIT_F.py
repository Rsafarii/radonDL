# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 13:01:46 2020

@author: RASA-
"""
def RASA_FIT_F(V_F):
	from numpy import asarray
	import numpy as np
	import streamlit as st
	if V_F == 'None' :
		m_F= -1.35
	elif V_F == 'Low' :
		m_F = -0.28
	elif V_F == 'Normal' :
		m_F = 0.74
	else :
		m_F = 1.04

	return  m_F