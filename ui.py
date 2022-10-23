from ast import Str
import io
import requests
from PIL import Image
import json
#from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np	
#import mpld3
#import streamlit.components.v1 as components

#----------------contants and variables definition----------------------------------
# interact with FastAPI endpoint


@st.cache  # No need for TTL this time. It's static data :)
def get_data_by_state():
	# name spaces 
	fields = list(pd.read_csv("NTA_NAME_SORT.csv").iloc[:,1])
	# dataset
	df = pd.read_csv('df_compose_BINES_TO_NUM.csv')
	df_OH = pd.read_csv('df_compose_BINES_ONEHOT1.csv')

	return fields, df, df_OH


fields, df, df_OH = get_data_by_state()

#backend = "http://fastapi:8000/segmentation"
backend = "https://fastapi-mvp-app-dzvcjejuga-uc.a.run.app/"
BIN = 0
LICENSE = 0
NTA = 0

#-------------------backend functions------------------------------------------------
def process_query(backend: str, BIN: int, LICENSE: str,  NTA: str) -> str:
	server_url = backend+"rework_index"
	params = {"LICENSE": LICENSE, "NTA":NTA, "BIN": BIN}
	r = requests.get(server_url, params = params, timeout=8000)
	return r

def plot_3d_scatter(x1, x2, y, ax=None, fig = None):
    if (fig is None) and (ax is None):
        fig = plt.figure(figsize = (15,10))
        ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x1, x2, y, alpha = 0.5, c= y)
    ax.set_xlabel(x1.name)
    ax.set_ylabel(x2.name)
    ax.set_zlabel(y.name)

#---------------------UI----------------------------------------------------------------
# construct UI layout
st.title("Rework prediction in construction projects: NYC case (MVP)")

st.write(
    """Obtain the predictions of how many times it will be necessary to renew construction permits.
        This might occur by internal and external reasons """
)  # description and instructions


def main_page():
	# Título y subtítulo.
	#st.title('Introducción a Streamlit')
	st.markdown('''Model overview''') 
	st.subheader("City of NYC")
	st.markdown('''***''')
	#st.sidebar.markdown('''
#	Drop down menu:
#
#		* I. Intro 
#
#		* II. Chart views 
#
#		* III. API test''')
#
	st.markdown(
        '''
        ### What is the hypotesis?
        An attempt is made to predict the amount of rework in civil works.

        ### What are the characteristics of the model?
        The results depend strongly on 3 features: 
        + Building Unique Number (BIN)
        + Neighborhood Unique Name (NTA)
        + Contractor's license number (Builder)
        ''')

	st.markdown('''
    	### types of permits NYC DOB NOW:''')

	img = Image.open('Permits.PNG')
	st.image(img)

	st.markdown('''
    	### Decision tree Screenshot (just up to the 2nd level)''')

	img = Image.open('arbol-2level.JPG')
	st.image(img)

	st.markdown('''### Feature importance:''')

	img = Image.open('FEATURE_IMPORTANCE.png')
	st.image(img)

	st.markdown('''### Model parameters:''')

	img = Image.open('tree-param.jpg')
	st.image(img)

	st.markdown('''***''')
    

def pageII():
	#Título.
	st.title('Chart views')
	st.markdown('***')
	#Subtítulo.
	st.subheader('Initial exploratory data analysis')

	#Uso de Checkboxs: Muestra información cuando se selecciona la caja.
	if st.checkbox('Show DF'): #Nombre al lado de la caja.
		st.dataframe(df) #Qué hace cuando se selecciona la caja.

	if st.checkbox('Data view'):#, disabled=True
		#Botones: Muestra información cuando se selecciona el botón.
		if st.button("Show head"): #Nombre del botón.
			st.write(df.head()) #Qué hace cuando se selecciona el botón.
		if st.button("Show tail"):
			st.write(df.tail())

	st.subheader("Dimensions:")
	#Radios: 
	df_dim = st.radio('Dimensions to show:', ('rows', 'Cols', 'both'), horizontal=True)

	if df_dim == 'rows':
		st.write('rows count:', df.shape[0])

	elif df_dim == 'cols':
		st.write('cols count:', df.shape[1])
		st.write('Cols: ',df.columns)
	else:
		st.write('rows count:', df.shape[0])
		st.write('cols count:', df.shape[1]) 
		st.markdown('***')
	if(st.checkbox('Scatter plot between features:', True) == True):

		ancho = st.sidebar.slider('Chart width', 1,10,10)
		alto = st.sidebar.slider('Chart heigth', 1,10,10)
		fig = plt.figure(figsize=(ancho,alto))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(df.license_prof, df.Bin, df.rework_index_compose, alpha = 0.5, c= df.rework_index_compose)
		ax.set_xlabel(df.license_prof.name)
		ax.set_ylabel(df.Bin.name)
		ax.set_zlabel(df.rework_index_compose.name)
		#fig_html = mpld3.fig_to_html(fig)
		#components.html(fig_html, height=alto)
		st.pyplot(fig)

		fig2 = plt.figure(figsize=(ancho,alto))
		ax2 = fig2.add_subplot(111, projection='3d')
		ax2.scatter(df.license_prof[50000:60000], df.Bin, df.rework_index_compose, alpha = 0.5, c= df.rework_index_compose)
		ax2.set_xlabel(df.license_prof.name)
		ax2.set_ylabel(df.Bin.name)
		ax2.set_zlabel(df.rework_index_compose.name)
		#fig_html = mpld3.fig_to_html(fig)
		#components.html(fig_html, height=alto)
		st.pyplot(fig2)

def pageIII():
	#Título.
	#st.title('Predicciones')
	#st.markdown('***')
	#st.subheader('Ingreso los datos para obtener la prediccion del modelo')
	BIN = st.selectbox('Enter the BIN number',df.Bin.unique())#, default = 3335229)
	LICENSE = st.selectbox('Enter the license number',df.license_prof.unique())#, default = 16217)
	NTA = st.selectbox('Enter the NTA name',df.GIS_NTA_NAME.unique())#, default = "Fort Greene")
	#prediccion = {"rework_predicted":"Presionar Predecir Rework"}
	show_result = st.container()

	if st.button("Predict rework"):
		#NTA_VAL = df_OH.columns.get_loc[NTA]
		prediccion = process_query(backend, BIN, LICENSE, NTA)
		prediccion = json.loads(prediccion.text)
		try:
			if int(prediccion["rework_predicted"].strip("[").strip("]")) == 3:
				prediccion["rework_predicted"] = "3 o more times"
		except: pass

		show_result.write("the rework value will be: " + str(prediccion["rework_predicted"]) + " with 61% of accuracy")

	st.markdown('***')

#----------------------left navigate bar--------------------------------------

page_names_to_funcs = {
    'I. Intro': main_page,
    'II. Chart views': pageII,
    'III. API test': pageIII }

selected_page = st.sidebar.selectbox("Select the page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


##-----------------Footer--------------------------------------------------------------
st.write('## References')
st.markdown('''	* [API model](https://fastapi-mvp-app-dzvcjejuga-uc.a.run.app/docs/)
* [NYC DOB official](https://data.cityofnewyork.us/Housing-Development/DOB-NOW-Build-Approved-Permits/rbx6-tga4)
* [Streamlit blog](https://blog.streamlit.io/)''')