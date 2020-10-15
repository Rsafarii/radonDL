def RASA_plotModelcom(model_results,model_names):
	import streamlit as st
	import matplotlib.pyplot as plt
	import matplotlib
	#matplotlib.use('TKAgg',warn=False, force=True)
	
	figure = plt.figure()
	figure.suptitle(' Models comparison')
	axis = figure.add_subplot(111)
	plt.boxplot(model_results)
	axis.set_xticklabels(model_names, rotation = 45, ha="right")
	axis.set_ylabel("Mean Squared Error (MSE)")
	plt.margins(0.05, 0.1)
	st.set_option('deprecation.showPyplotGlobalUse', False)
	st.pyplot()
	#plt.savefig("model_mse_scores.png")
	#plt.clf()
	#plt.close()