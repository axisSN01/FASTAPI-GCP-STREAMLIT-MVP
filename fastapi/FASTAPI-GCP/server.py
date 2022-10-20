import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
import xgboost as xgb
import numpy as np
import pandas as pd

#xgb_tree = xgb.Booster({'nthread': 4})  # init model
#xgb_tree.load_model('xgb_model_REG_012.json')  # load data
#xgb_tree = xgb.load_model("xgb_model_REG_012.json")


#-------------------------------read lookup table de nombres------------------------------------

NTA_NUM = pd.read_csv("NTA_NAME_SORT.csv")

#-----------------------------Cargamos modelo---------------------------------------------
# levantamos el modelo guardado y repetimos prueba - valido para XGBOOST V. >= 1.6
#xgb_tree = xgb.Booster()
#xgb_tree = xgb.XGBClassifier()
xgb_tree = xgb.XGBClassifier(objective='multi:softprob')
xgb_tree.load_model("model_CLS_61.json")


# Start FastAPI app -------------------------------------------------------------------------------------
app = FastAPI(
    title="API evaluación de retrabajos en obras civiles",
    description="""Ingresando el nombre del Neighborhood Tabulation Areas y la licencia del contratista, se predice el indice
                    de retrabajo, basado en un modelo de ML XGBOOST, actualmente con una precision de 66%""",
    version="0.0.0",
)


@app.get("/")
async def home():
    return {"message": "API reworks index online. Go to '/docs' for reference."}


@app.get("/rework_index")
async def renew_permits(NTA: str, BIN: int, LICENSE: int) -> str:
    """
    Predict reworks index (how much times You would need to renew work permits). <br>
    :param NTA: STR <br>
    :param BIN: INT <br>
    :param LICENSE: INT <br>
    :return: str - ["times to renew work permits"]
    """
    data = list(np.zeros(196).astype("int"))
    item_uno = int(NTA_NUM[NTA_NUM.GIS_NTA_NAME == NTA].index.values)

    data[0] = BIN
    data[1] = LICENSE
    data[item_uno + 4] =1
    names = pd.Series(["Bin","license_prof"])
    names = names.append(NTA_NUM.GIS_NTA_NAME)
    to_predict = pd.DataFrame(data = np.array(data).reshape(1,196), columns= list(names))
    
    request = xgb_tree.predict(to_predict)

    #request = "request OK"
    
    return {'rework_predicted': str(request)}
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # run in terminal 