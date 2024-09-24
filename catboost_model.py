import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import ast


MODEL_NAME = "catboost_model"
PROPORTION = 1/1000 # Cant datos
TEST = False

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


print("Cargando los datos...")
ctr_15 = pd.read_csv ("./datos/ctr_15.csv")
ctr_16 = pd.read_csv ("./datos/ctr_16.csv")
ctr_17 = pd.read_csv ("./datos/ctr_17.csv")
ctr_18 = pd.read_csv ("./datos/ctr_18.csv")
ctr_19 = pd.read_csv ("./datos/ctr_19.csv")
ctr_20 = pd.read_csv ("./datos/ctr_20.csv")
ctr_21 = pd.read_csv ("./datos/ctr_21.csv")
ctr_test = pd.read_csv ("./datos/ctr_test.csv")
eval_data = ctr_test

# convertimos auction_time a datetime y despues lo separo en columnas (año,mess,dia,hora,minuto)
for df in [ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21,eval_data]:
    df["auction_time"] = pd.to_datetime(df["auction_time"], unit='s')
    df["day"] = df["auction_time"].dt.day
    df["hour"] = df["auction_time"].dt.hour
    df["minute"] = df["auction_time"].dt.minute
    df["weekday"] = df["auction_time"].dt.weekday

# datos de etnrenamiento (agregar el 21)
train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20])
if not TEST:
    train_data = pd.concat([train_data,ctr_21])

train_data = train_data.sample(frac=PROPORTION, random_state=1234)

## nos quedamos cortos de ram
del ctr_15, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21
gc.collect()


y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
eval_data = eval_data.drop(columns=["Label"])
print("Datos cargados")


# consigo los indices de las variables categoricas (ya que el catboost las maneja nativamente) pero tienen que ser en string o int 
categorical_cols = X_train.select_dtypes(exclude='number').columns
if not TEST:
    categorical_cols_eval = eval_data.select_dtypes(exclude='number').columns

# convierto a str los objetos porque el catboost es exquisito (no objetos)
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
if not TEST:
    eval_data[categorical_cols_eval] = eval_data[categorical_cols_eval].astype(str)

# tamaño de los datos
print("cantidad columnas: " + str(X_train.shape[1]))
print("cantidad filas: " + str(X_train.shape[0]))

del train_data
gc.collect()


# CatBoost
print("Training the CatBoost model...")
catboost_model = CatBoostClassifier(iterations=10, learning_rate=0.1, depth=10, loss_function='Logloss', eval_metric='AUC', random_seed=2345, verbose=True, early_stopping_rounds=100, class_weights=[1, 100])
catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list())

if not TEST:
    # nos creamos el archivo de submission
    assert X_train.columns.equals(eval_data.columns)
    y_preds = catboost_model.predict_proba(eval_data)[:, catboost_model.classes_ == 1].squeeze()

    submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
    submission_df["id"] = submission_df["id"].astype(int)
    submission_df.to_csv(MODEL_NAME + ".csv", sep=",", index=False)

print("Se entreno el " + MODEL_NAME + " con un: " + str(PROPORTION) + " de los datos")
print("sin one hot encoding de listas")


# me mate la ram
#del X_train
#del y_train
#del eval_data
#gc.collect()
# Testeamos a ver como nos va a ir
#y_test = ctr_21["Label"]
#X_test = ctr_21.drop(columns=["Label"])
#X_test,_ = one_hot_encode_lists(X_test, list_columns, mlb_dict)
#X_test, _ = X_test.align(X_train, join='outer', axis=1, fill_value=0)
#categorical_cols_test = X_test.select_dtypes(exclude='number').columns
#X_test[categorical_cols_test] = X_test[categorical_cols_test].astype(str)
#y_preds_test = catboost_model.predict_proba(X_test)[:, catboost_model.classes_ == 1].squeeze()
#roc_auc = roc_auc_score(y_test, y_preds_test)
#print("ROC AUC: " + str(roc_auc) + " en el set de testeo (21) usando el " + str(PROPORTION) + " de los datos")
