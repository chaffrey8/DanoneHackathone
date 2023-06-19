from Python.ReadFiles import *
from numpy import round
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import json

df = read_data()
X,y = build_Xy(df)

df_test = read_data(False)
dfran = brands_matrix(df_test)
dfran.columns = [f'B_{col}' for col in dfran.columns]
dflan = language_matrix(df_test)
dflan.columns = [f'L_{col}' for col in dflan.columns]
dfhie = hierarchy_matrix(df_test)
dfhie.columns = [f'H_{col}' for col in dfhie.columns]
dfsec = selling_matrix(df_test)
dfsec.columns = [f'S_{col}' for col in dfsec.columns]
dfopr = origins_matrix(df_test)
dfopr.columns = [f'O_{col}' for col in dfopr.columns]
dfpma = materials_matrix(df_test)
dfpma.columns = [f'M_{col}' for col in dfpma.columns]
dfnut = nutrition_matrix(df_test)
dfnut.columns = [f'N_{col}' for col in dfnut.columns]
dfcova = covariables_matrix(df_test)
dfcova.columns = [f'C_{col}' for col in dfcova.columns]
X_test = concat([dfran,dflan,dfhie,dfsec,dfopr,dfpma,dfnut,dfcova],axis=1)

y_test = dtc.predict(X_test)

target = {str(i):int(v) for i,v in enumerate(y_test)}
jdi = {'target':target}
with open(BASE_DIR / 'data' / 'predictions.json', "w") as outfile:
    json.dump(jdi, outfile,indent=0)