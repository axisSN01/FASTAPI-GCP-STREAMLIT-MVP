# Proyecto escuela calculo de indice de retrabajo, en obras edilicias de New York City

## Hipotesis: 
Se plantea la hipotesis de determinar un indice de retrabajo, basado en la relacion existente entre: 
+ Permisos de inicio de obra VS. Permisos de renovacion de obra

se asume que el ratio:  `Renovacion/ Inicio` Podria dar informacion valiosa al momento de elegir la empresa contratista, en funcion de la zona geografica o en funcion del tipo de edificio.

--- 
## El caso base: 
El conocimiento a priori de este ratio es del `#25%` de probabilidad de acierto, debido a que las opciones son 4 eventos aleatorios: 

+ 0: No debo renovar el permiso
+ 1: Debo renover una vez el permiso
+ 2: Debo renovar dos veces el permiso
+ 3: Debo renovar 3 o mas veces el permiso

---

## Precision del modelo de machine learning: 
El modelo desarrollado tiene una precision del `#%61` y esta basado en el algoritmo XGBOOST de clasificacion, las caracteristicas se pueden ver en la web del FrontEnd.

--- 

# Front-end del modelo:

[FrontEnd - streamlit](https://axissn01-fastapi-gcp-streamlit-mvp-ui-2swe87.streamlitapp.com/)


# Back-end del modelo:

[BackEnd - Fast API](https://fastapi-mvp-app-dzvcjejuga-uc.a.run.app/docs)


--- 

# Arquitectura de los datos

<img src="DF-architecture.PNG" height="300">

# Digrama de capas de la implementacion de FAST API

<img src="Layer-diagram.PNG" height="300">

--- 






