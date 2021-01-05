import base64
import datetime 
from datetime import  timedelta
from wwo_hist import retrieve_hist_data
import time
import sys
import pandas as pd
from numpy import asarray
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

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


#@st.cache(suppress_st_warning=True,persist=True,allow_output_mutation=True) 
def TITLE():
    components.html(
'''              
<style>  
p{
text-align: center;
color:#ff944d;
font-family:TimesNewRoman, Times, serif;
font-size:20px;

}              
   body {
	margin: 0;
	height: 100px;
	display: flex;
	align-items: center;
	justify-content: center;
	background-color:#fff;
}

</style>

<div> 
  <p>  Radon Deep Analyser </p>
</div>

''',height=100,)
#@st.cache(suppress_st_warning=True,persist=True,allow_output_mutation=True) 
def TITLE_SIDEBAR():
    with st.sidebar.beta_container():
        components.html(
'''
                
<style>  
         
   body {
	margin: 0;
	height: 100px;
	display: flex;
	align-items: center;
	justify-content: center;
	background-color:#f0f2f6;
}

.loader {
    width: 120px;
    height: 120px;
    font-size: 10px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}

.loader .face {
    position: absolute;
    border-radius: 50%;
    border-style: solid;
    animation: animate 3s linear infinite;
}

.loader .face:nth-child(1) {
    width: 80%;
    height: 80%;
    color: gold;
    border-color: currentColor transparent transparent currentColor;
    border-width: 0.2em 0.2em 0em 0em;
    --deg: -45deg;
    animation-direction: normal;
}

.loader .face:nth-child(2) {
    width: 70%;
    height: 70%;
    color: lime;
    border-color: currentColor currentColor transparent transparent;
    border-width: 0.2em 0em 0em 0.2em;
    --deg: -135deg;
    animation-direction: reverse;
}

.loader .face .circle {
    position: absolute;
    width: 50%;
    height: 0.1em;
    top: 50%;
    left: 50%;
    background-color: transparent;
    transform: rotate(var(--deg));
    transform-origin: left;
}

.loader .face .circle::before {
    position: absolute;
    top: -0.5em;
    right: -0.5em;
    content: '';
    width: 1em;
    height: 1em;
    background-color: currentColor;
    border-radius: 50%;
    box-shadow: 0 0 2em,
                0 0 4em,
                0 0 6em,
                0 0 8em,
                0 0 10em,
                0 0 0 0.5em rgba(255, 255, 0, 0.1);
}

@keyframes animate {
    to {
        transform: rotate(1turn);
    }
}

</style>

	<div class="loader">
  <div class="face">
    <div class="circle"></div>
  </div>
  <div class="face">
    <div class="circle"></div>
  </div>
  <div>
  <p align='center' style='font-size:15px; color:red;'>  RADON </p>
</div>

</div>

''',height=100,)

TITLE()
PASS()
TITLE_SIDEBAR()


st.sidebar.markdown('** Step 1 :**Getting available information')
time.sleep(1)
V_I = st.sidebar.selectbox('Level of ventilation:',
				( 'None','Low',
                    		'Normal', 'High'),index=2)
UD_I = st.sidebar.slider('Outdoor Tempereture(C):', -10, 35, 1)            		
R_I = st.sidebar.slider('Radon Concentration(Bq/m3):', 5, 2000, 20)
st.write(' ')
st.write(' ')

from RASA_TIME import RASA_TIME
out_RASA_TIME = RASA_TIME()
location = out_RASA_TIME[2]
location_list=[location]
Di=out_RASA_TIME[0]
Df=out_RASA_TIME[1]
V_F= out_RASA_TIME[3]

if st.button('Click here to Continue...!'):
	pass
else:
	st.stop()
time.sleep(4)
st.sidebar.markdown('** Step 2 :**Model Training ')
chek1 = 1
placeholder = st.empty()
with placeholder.beta_container():
    components.html('''
 <style>
* {
  border: 0;
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

:root {
  font-size: calc(16px + (24 - 16)*(100vw - 320px)/ (1280 - 320));
}

body, .preloader {
  display: flex;
}

body {
  background:  #ffffff;
  color: #3df1f1;
  font: 1em Dosis, sans-serif;
  height: 100vh;
  line-height: 1.5;
  perspective: 40em;
}

.preloader {
  animation: tiltSpin 8s linear infinite;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  margin: auto;
  width: 17em;
  height: 17em;
}
.preloader, .preloader__ring {
  transform-style: preserve-3d;
}
.preloader__ring {
  animation-name: spin;
  animation-duration: 4s;
  animation-timing-function: inherit;
  animation-iteration-count: inherit;
  font-size: 2em;
  position: relative;
  height: 3rem;
  width: 1.5rem;
}
.preloader__ring:nth-child(even) {
  animation-direction: reverse;
}
.preloader__sector {
  font-weight: 600;
  position: absolute;
  top: 0;
  left: 0;
  text-align: center;
  text-transform: uppercase;
  transform: translateZ(7rem);
}
.preloader__sector, .preloader__sector:empty:before {
  display: inline-block;
  width: 100%;
  height: 100%;
}
.preloader__sector:empty:before {
  background: linear-gradient(transparent 45%, currentColor 45% 55%, transparent 55%);
  content: "";
}
.preloader__sector:nth-child(2) {
  transform: rotateY(12deg) translateZ(7rem);
}
.preloader__sector:nth-child(3) {
  transform: rotateY(24deg) translateZ(7rem);
}
.preloader__sector:nth-child(4) {
  transform: rotateY(36deg) translateZ(7rem);
}
.preloader__sector:nth-child(5) {
  transform: rotateY(48deg) translateZ(7rem);
}
.preloader__sector:nth-child(6) {
  transform: rotateY(60deg) translateZ(7rem);
}
.preloader__sector:nth-child(7) {
  transform: rotateY(72deg) translateZ(7rem);
}
.preloader__sector:nth-child(8) {
  transform: rotateY(84deg) translateZ(7rem);
}
.preloader__sector:nth-child(9) {
  transform: rotateY(96deg) translateZ(7rem);
}
.preloader__sector:nth-child(10) {
  transform: rotateY(108deg) translateZ(7rem);
}
.preloader__sector:nth-child(11) {
  transform: rotateY(120deg) translateZ(7rem);
}
.preloader__sector:nth-child(12) {
  transform: rotateY(132deg) translateZ(7rem);
}
.preloader__sector:nth-child(13) {
  transform: rotateY(144deg) translateZ(7rem);
}
.preloader__sector:nth-child(14) {
  transform: rotateY(156deg) translateZ(7rem);
}
.preloader__sector:nth-child(15) {
  transform: rotateY(168deg) translateZ(7rem);
}
.preloader__sector:nth-child(16) {
  transform: rotateY(180deg) translateZ(7rem);
}
.preloader__sector:nth-child(17) {
  transform: rotateY(192deg) translateZ(7rem);
}
.preloader__sector:nth-child(18) {
  transform: rotateY(204deg) translateZ(7rem);
}
.preloader__sector:nth-child(19) {
  transform: rotateY(216deg) translateZ(7rem);
}
.preloader__sector:nth-child(20) {
  transform: rotateY(228deg) translateZ(7rem);
}
.preloader__sector:nth-child(21) {
  transform: rotateY(240deg) translateZ(7rem);
}
.preloader__sector:nth-child(22) {
  transform: rotateY(252deg) translateZ(7rem);
}
.preloader__sector:nth-child(23) {
  transform: rotateY(264deg) translateZ(7rem);
}
.preloader__sector:nth-child(24) {
  transform: rotateY(276deg) translateZ(7rem);
}
.preloader__sector:nth-child(25) {
  transform: rotateY(288deg) translateZ(7rem);
}
.preloader__sector:nth-child(26) {
  transform: rotateY(300deg) translateZ(7rem);
}
.preloader__sector:nth-child(27) {
  transform: rotateY(312deg) translateZ(7rem);
}
.preloader__sector:nth-child(28) {
  transform: rotateY(324deg) translateZ(7rem);
}
.preloader__sector:nth-child(29) {
  transform: rotateY(336deg) translateZ(7rem);
}
.preloader__sector:nth-child(30) {
  transform: rotateY(348deg) translateZ(7rem);
}

/* Animations */
@keyframes tiltSpin {
  from {
    transform: rotateY(0) rotateX(30deg);
  }
  to {
    transform: rotateY(1turn) rotateX(30deg);
  }
}
@keyframes spin {
  from {
    transform: rotateY(0);
  }
  to {
    transform: rotateY(1turn);
  }
}
</style>
<div class="preloader">
  <div class="preloader__ring">
    <div class="preloader__sector">R</div>
    <div class="preloader__sector">a</div>
    <div class="preloader__sector">d</div>
    <div class="preloader__sector">o</div>
    <div class="preloader__sector">n</div>
    <div class="preloader__sector">.</div>
    <div class="preloader__sector">.</div>
    <div class="preloader__sector">.</div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
  </div>
  <div class="preloader__ring">
    <div class="preloader__sector">R</div>
    <div class="preloader__sector">a</div>
    <div class="preloader__sector">d</div>
    <div class="preloader__sector">o</div>
    <div class="preloader__sector">n</div>
    <div class="preloader__sector">.</div>
    <div class="preloader__sector">.</div>
    <div class="preloader__sector">.</div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
    <div class="preloader__sector"></div>
  </div>
</div>                           
''',height=260,)

#api_key= '8764781779e0415daeb81317200511'
api_key='136e59990d93422f930103939202811'
while chek1<20:
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
                break
            except :
                chek1 +=1
else :
        placeholder.empty()
        st.info('Can not connect to the server! Please try again ....')
        st.stop()
st.write('')
placeholder.empty()
EMP1 = st.empty()
EMP2 = st.empty()

EMP1.success('Weather data were collected!')

wDate = hist_weather_data[0]
wDate=wDate.dropna(how='any')
from RASA_W_ANALYZE import RASA_W_ANALYZE
out_RASA_W_ANALYZE = RASA_W_ANALYZE(wDate)
MEAN = out_RASA_W_ANALYZE[0]
Date_value =out_RASA_W_ANALYZE[1]

from RASA_FIT import RASA_FIT
R_0_I = RASA_FIT(V_I,  UD_I, R_I)
from RASA_RATIO import RASA_RATIO
R_0_F = RASA_RATIO(V_I,  V_F, R_0_I)

from RASA_FIT_F import RASA_FIT_F
m_F = RASA_FIT_F(V_F)	


st.sidebar.markdown('** Step 3 :**Model Testing ')
with st.spinner('Wait for It...'):
    pr=m_F*(MEAN)+R_0_F
time.sleep(10)
EMP2.success('Predictions were made!')
st.set_option('deprecation.showPyplotGlobalUse', False)
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
st.sidebar.success('Predictions were made!')
time.sleep(2)

EMP1.empty()
EMP2.empty()