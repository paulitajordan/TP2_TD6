import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
import numpy as np


# mejorar lo de las listas
MODEL_NAME = "catboost_model"
PROPORTION = 1/10 # Cant datos
TEST = False

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


ctr_15 = pd.read_csv ("./datos/ctr_15.csv")
ctr_16 = pd.read_csv ("./datos/ctr_16.csv")
ctr_17 = pd.read_csv ("./datos/ctr_17.csv")
ctr_18 = pd.read_csv ("./datos/ctr_18.csv")
ctr_19 = pd.read_csv ("./datos/ctr_19.csv")
ctr_20 = pd.read_csv ("./datos/ctr_20.csv")
ctr_21 = pd.read_csv ("./datos/ctr_21.csv")
ctr_test = pd.read_csv ("./datos/ctr_test.csv")


ctr_15["Dia"] = 1
ctr_16["Dia"] = 2
ctr_17["Dia"] = 3
ctr_18["Dia"] = 4
ctr_19["Dia"] = 5
ctr_20["Dia"] = 6
ctr_21["Dia"] = 7 # de validación
ctr_test["Dia"] = 8 # para optimizar
eval_data = ctr_test

# datos de etnrenamiento (agregar el 21)
train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20])
if not TEST:
    train_data = pd.concat([train_data,ctr_21])

train_data = train_data.sample(frac=PROPORTION, random_state=1234)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

# consigo los indices de las variables categoricas (ya que el catboost las maneja nativamente) pero tienen que ser en string o int 
categorical_cols = X_train.select_dtypes(exclude='number').columns
categorical_cols_eval = eval_data.select_dtypes(exclude='number').columns
catgorical_indices = [X_train.columns.get_loc(col) for col in categorical_cols]


# convierto a str los objetos porque el catboost es exquisito (no objetos)
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
eval_data[categorical_cols_eval] = eval_data[categorical_cols_eval].astype(str)

del train_data
gc.collect()

# CatBoost
catboost_model = CatBoostClassifier()
print("Training the CatBoost model...")
catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=10, loss_function='Logloss', eval_metric='AUC', random_seed=2345, verbose=100)
catboost_model.fit(X_train, y_train, cat_features=catgorical_indices)

# nos creamos el archivo de submission
y_preds = catboost_model.predict_proba(eval_data)[:, catboost_model.classes_ == 1].squeeze()
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv(MODEL_NAME + ".csv", sep=",", index=False)

print("Se entreno usando el " + MODEL_NAME + " con un: " + str(PROPORTION) + " de los datos")


# Testeamos a ver como nos va a ir
#y_test = ctr_21["Label"]
#X_test = ctr_21.drop(columns=["Label"])
#categorical_cols_test = ctr_test.select_dtypes(exclude='number').columns
#X_test[categorical_cols_test] = X_test[categorical_cols_test].astype(str)
#y_preds_test = catboost_model.predict_proba(X_test)[:, catboost_model.classes_ == 1].squeeze()
#test_data = ctr_21
#print("ROC AUC: " + str(roc_auc) + " en el set de testeo (21) usando el " + str(PROPORTION) + " de los datos")
#roc_auc = roc_auc_score(y_test, y_preds_test)