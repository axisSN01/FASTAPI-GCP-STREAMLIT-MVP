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
st.title("Rework prediction in construction projects NYC (MVP)")

st.write(
    """Obtain the predictions of how many times It will necessary to renew permits of construction.
        This occurs by internal and external reasons """
)  # description and instructions


def main_page():
	# Título y subtítulo.
	#st.title('Introducción a Streamlit')
	st.markdown('''# Test model''') 
	st.subheader("City of NYC")
	st.markdown('''***''')
	st.sidebar.markdown('''
	* I. Introducción 
	* II. Visualizaciones
	* III. Pruebas API''')

	st.markdown(
        '''
        ### ¿Cual es la hipotesis base?
        Se intenta predecir la cantidad de retrabajo en las obras civiles.

        ### ¿Que caracteristicas tiene el modelo?
        Los resultados dependen fuertemente de 2 caracteristicas: 
        + Numero unico de edificio (BIN)
		+ Nombre unico de vecindario (NTA)
        + Numero de licencia del contratista (Constructor)
        ''')

	st.markdown('''
    	### Captura del arbol del modelo (hasta el 2do nivel)''')

	img = Image.open('arbol-2level.JPG')
	st.image(img)

	st.markdown('''### Feature importance:''')

	img = Image.open('FEATURE_IMPORTANCE.png')
	st.image(img)

	st.markdown('''### Parametros del modelo:''')

	img = Image.open('tree-param.jpg')
	st.image(img)

	st.markdown('''***''')
    

def pageII():
	#Título.
	st.title('Visualizaciones')
	st.markdown('***')
	#Subtítulo.
	st.subheader('Exploración inicial del dataset')

	#Uso de Checkboxs: Muestra información cuando se selecciona la caja.
	if st.checkbox('Mostrar DF'): #Nombre al lado de la caja.
		st.dataframe(df) #Qué hace cuando se selecciona la caja.

	if st.checkbox('Vista de datos'):#, disabled=True
		#Botones: Muestra información cuando se selecciona el botón.
		if st.button("Mostrar head"): #Nombre del botón.
			st.write(df.head()) #Qué hace cuando se selecciona el botón.
		if st.button("Mostrar tail"):
			st.write(df.tail())

	st.subheader("Información de las dimensiones")
	#Radios: 
	df_dim = st.radio('Dimensión a mostrar:', ('Filas', 'Columnas', 'Ambas'), horizontal=True)

	if df_dim == 'Filas':
		st.write('Cantidad de filas:', df.shape[0])

	elif df_dim == 'Columnas':
		st.write('Cantidad de columnas:', df.shape[1])
		st.write('Columnas: ',df.columns)
	else:
		st.write('Cantidad de filas:', df.shape[0])
		st.write('Cantidad de columnas:', df.shape[1]) 
		st.markdown('***')
	if(st.checkbox('Gráfico de dispersión entre features', True) == True):
		ancho = st.sidebar.slider('Ancho del gráfico', 1,10,6)
		alto = st.sidebar.slider('Alto del gráfico', 1,10,4)
		fig = plt.figure(figsize=(ancho,alto))
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(df.license_prof, df.Bin, df.rework_index_compose, alpha = 0.5, c= df.rework_index_compose)
		ax.set_xlabel(df.license_prof.name)
		ax.set_ylabel(df.Bin.name)
		ax.set_zlabel(df.rework_index_compose.name)
		st.pyplot(fig)

def pageIII():
	#Título.
	#st.title('Predicciones')
	#st.markdown('***')
	#st.subheader('Ingreso los datos para obtener la prediccion del modelo')
	BIN = st.selectbox('Ingrese numero de BIN',df.Bin.unique())#, default = 3335229)
	LICENSE = st.selectbox('Ingrese numero de Licencia',df.license_prof.unique())#, default = 16217)
	NTA = st.selectbox('Ingrese numero de NTA',df.GIS_NTA_NAME.unique())#, default = "Fort Greene")
	prediccion = 0
	if st.button("Predecir rework"):
		#NTA_VAL = df_OH.columns.get_loc[NTA]
		prediccion = process_query(backend, BIN, LICENSE, NTA)
		prediccion = json.loads(prediccion.text)

	show_result = st.container()
	show_result.write("El valor de retrabajo es: " + str(prediccion["rework_predicted"]) + " con un 60% de precision")
	st.markdown('***')

#----------------------left navigate bar--------------------------------------

page_names_to_funcs = {
    'I. Introducción': main_page,
    'II. Visualizaciones': pageII,
    'III. Pruebas API': pageIII }

selected_page = st.sidebar.selectbox("Seleccione página", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()


##-----------------Footer--------------------------------------------------------------
st.write('## Material complementario')
st.markdown('''	* [API del modelo](https://fastapi.io)
* [Documentación oficial NYC DOB](https://data.cityofnewyork.us/)
* [Blog oficial Streamlit](https://blog.streamlit.io/)''')