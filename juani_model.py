import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import re
import numpy as np


MODEL_NAME = "catboost_model"
PROPORTION = 0.25 # Cant datos

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

print("Datos cargados")

# Función para limpiar y convertir a enteros
def clean_and_convert(lst_str):
    if isinstance(lst_str, str) and len(lst_str) > 0:
        # Limpiamos los caracteres no numéricos
        clean_list = [re.sub(r'\D', '', i) for i in lst_str.split(",")]
        # Filtramos los elementos que no sean vacíos tras la limpieza
        clean_list = [int(i) for i in clean_list if i.isdigit()]
        return clean_list
    return []

# convertimos auction_time a datetime y despues lo separo en columnas (año,mess,dia,hora,minuto)
for df in [ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21,eval_data]:
    # convertimos a datetime
    df["auction_time"] = pd.to_datetime(df["auction_time"], unit='s')
    df["day"] = df["auction_time"].dt.day
    df["hour"] = df["auction_time"].dt.hour
    df["minute"] = df["auction_time"].dt.minute
    df["weekday"] = df["auction_time"].dt.weekday
    # Trabajamos las listas
    # Tamaños
    df["auction_list_0_size"] = df["auction_list_0"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["action_list_1_size"] = df["action_list_1"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)
    df["action_list_2_size"] = df["action_list_2"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)
    # Promedios
    df["action_list_1_mean"] = df["action_list_1"].apply(lambda x: sum(clean_and_convert(x)) / len(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)
    df["action_list_2_mean"] = df["action_list_2"].apply(lambda x: sum(clean_and_convert(x)) / len(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)
    
    # Sumatorias
    df["action_list_1_sum"] = df["action_list_1"].apply(lambda x: sum(clean_and_convert(x)))
    df["action_list_2_sum"] = df["action_list_2"].apply(lambda x: sum(clean_and_convert(x)))

    # Medianas
    df["action_list_1_median"] = df["action_list_1"].apply(lambda x: np.median(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)
    df["action_list_2_median"] = df["action_list_2"].apply(lambda x: np.median(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)

    # Varianzas
    df["action_list_1_variance"] = df["action_list_1"].apply(lambda x: np.var(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)
    df["action_list_2_variance"] = df["action_list_2"].apply(lambda x: np.var(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)

    # Max y Min
    df["action_list_1_max"] = df["action_list_1"].apply(lambda x: max(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)
    df["action_list_1_min"] = df["action_list_1"].apply(lambda x: min(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)
    df["action_list_2_max"] = df["action_list_2"].apply(lambda x: max(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)
    df["action_list_2_min"] = df["action_list_2"].apply(lambda x: min(clean_and_convert(x)) if len(clean_and_convert(x)) > 0 else 0)

    # Rango max-min
    df["action_list_1_range"] = df["action_list_1_max"] - df["action_list_1_min"]
    df["action_list_2_range"] = df["action_list_2_max"] - df["action_list_2_min"]

    # Cantidad de valores únicos
    df["action_list_1_unique"] = df["action_list_1"].apply(lambda x: len(set(clean_and_convert(x))) if len(clean_and_convert(x)) > 0 else 0)
    df["action_list_2_unique"] = df["action_list_2"].apply(lambda x: len(set(clean_and_convert(x))) if len(clean_and_convert(x)) > 0 else 0)

    print("Dataframe transformado")

# datos de etnrenamiento (agregar el 21)
train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21])

# saco la mitad de los datos con label 0
data_with_label_1 = train_data[train_data["Label"] == 1]
data_with_label_0 = train_data[train_data["Label"] == 0].sample(frac=PROPORTION, random_state=1234)

train_data = pd.concat([data_with_label_1, data_with_label_0])


## nos quedamos cortos de ram
del ctr_15, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21
gc.collect()

# shuffle
train_data = train_data.sample(frac=1, random_state=1234)

print("columas de tiempo y listas transformadas")



y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])



# consigo los indices de las variables categoricas (ya que el catboost las maneja nativamente) pero tienen que ser en string o int 
categorical_cols = X_train.select_dtypes(exclude='number').columns
# convierto a str los objetos porque el catboost es exquisito (no objetos)
X_train[categorical_cols] = X_train[categorical_cols].astype(str)
eval_data[categorical_cols] = eval_data[categorical_cols].astype(str)


# tamaño de los datos
print("cantidad columnas: " + str(X_train.shape[1]))
print("cantidad filas: " + str(X_train.shape[0]))

del train_data
gc.collect()


# CatBoost
print("Training the CatBoost model...")
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=8, loss_function='Logloss', eval_metric='AUC', random_seed=11, verbose=True, early_stopping_rounds=100)
catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list())

#usamos predict_proba en lugar de predict porque mejora el auc
y_preds = catboost_model.predict_proba(eval_data)[:, 1]
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv(MODEL_NAME + ".csv", sep=",", index=False)

print("Se entreno el " + MODEL_NAME + " con un: " + str(PROPORTION) + " de los datos")