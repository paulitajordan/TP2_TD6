import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import re
import numpy as np
from sklearn.model_selection import train_test_split
import json

# IDEAS
#  probar pasando los ids a categoricos

MODEL_NAME = "catboost_model"
PROPORTION = 1/100 # Cant datos

print("Cargando los datos...")
ctr_15_full = pd.read_csv ("./datos/ctr_15.csv")
ctr_16 = pd.read_csv ("./datos/ctr_16.csv")
ctr_17 = pd.read_csv ("./datos/ctr_17.csv")
ctr_18 = pd.read_csv ("./datos/ctr_18.csv")
ctr_19 = pd.read_csv ("./datos/ctr_19.csv")
ctr_20 = pd.read_csv ("./datos/ctr_20.csv")
ctr_21 = pd.read_csv ("./datos/ctr_21.csv")
print("Datos cargados")

# concat
ctr_15_full_unclean = pd.concat([ctr_15_full, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21], ignore_index=True)

ctr_15_full = ctr_15_full_unclean.drop(columns=["action_categorical_5", "action_categorical_6", "auction_categorical_5", "creative_categorical_11", "auction_categorical_10", "creative_categorical_1", "auction_categorical_2", "timezone_offset", "auction_boolean_1", "auction_categorical_3", "auction_boolean_2", "has_video", "creative_categorical_7", "auction_boolean_0", "creative_categorical_10"])
print("Columnas eliminadas")

# reducir la cantidad de datos
ctr_15 = ctr_15_full.sample(frac=PROPORTION, random_state=1234)
del ctr_15_full, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21
gc.collect()

# Función para limpiar y convertir a enteros
def clean_and_convert(lst_str):
    if isinstance(lst_str, str) and len(lst_str) > 0:
        # Limpiamos los caracteres no numéricos
        clean_list = [re.sub(r'\D', '', i) for i in lst_str.split(",")]
        # Filtramos los elementos que no sean vacíos tras la limpieza
        clean_list = [int(i) for i in clean_list if i.isdigit()]
        return clean_list
    return []


# one hot encoding de la columna auction_list_0 (que es una lista de strings), example: "[""social_networking"",""IAB24"",""lifestyle"",""IAB14""]"



# convertimos auction_time a datetime y despues lo separo en columnas (año,mess,dia,hora,minuto)
for df in [ctr_15]:
    # convertimos a datetime
    #df = df.drop(columns=["action_categorical_5", "action_categorical_6", "auction_categorical_5", "creative_categorical_11", "auction_categorical_10", "creative_categorical_1", "auction_categorical_2", "timezone_offset", "auction_boolean_1", "auction_categorical_3", "auction_boolean_2", "creative_width", "has_video", "creative_categorical_7", "auction_boolean_0", "creative_categorical_10"])

    # convertimos a datetime
    df["auction_time"] = pd.to_datetime(df["auction_time"], unit='s')
    df["day"] = df["auction_time"].dt.day
    df["hour"] = df["auction_time"].dt.hour
    df["minute"] = df["auction_time"].dt.minute
    df["fullMinute"] = df["auction_time"].dt.hour * 60 + df["auction_time"].dt.minute
    df["weekday"] = df["auction_time"].dt.weekday
    ## seno y coseno de la hora
    #df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    #df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    ## seno y coseno del minuto
    #df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    #df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    #df["fullMinute_sin"] = np.sin(2 * np.pi * df["fullMinute"] / 1440)
    #df["fullMinute_cos"] = np.cos(2 * np.pi * df["fullMinute"] / 1440)
    ## drop auction_time
    df = df.drop(columns=["auction_time", "auction_time"])


    # relaciones no lineales de bidfloor
    #df["bidfloor_squared"] = df["auction_bidfloor"] ** 2
    #df["bidfloor_sqrt"] = np.sqrt(df["auction_bidfloor"])
    #df["bidfloor_log"] = np.log(df["auction_bidfloor"] + 1)
    #df["bidfloor_exp"] = np.exp(df["auction_bidfloor"])
    #df["bidfloor_inv"] = 1 / (df["auction_bidfloor"]+ 0.0001)

    df["creative_area"] = (df["creative_height"] * df["creative_width"] - (df["creative_height"] * df["creative_width"]).mean()) / (df["creative_height"] * df["creative_width"]).std()


    # Trabajamos las listas
    # Tamaños
    #df["action_list_1_size"] = df["action_list_1"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)
    df["action_list_2_size"] = df["action_list_2"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)

    # info de auction list 0
    df["auction_list_0_size"] = df["auction_list_0"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)
    df["auction_list_0_unique"] = df["auction_list_0"].apply(lambda x: len(set(x.split(","))) if isinstance(x, str) else 0)

    # separo las listas en columnas solo con auction_list_0
    auction_list_0 = df["auction_list_0"].str.split(",", expand=True)
    # limpio los [ y ] de las columnas nuevas
    auction_list_0 = auction_list_0.apply(lambda x: x.str.replace("[", ""))
    auction_list_0 = auction_list_0.apply(lambda x: x.str.replace("]", ""))
    # me quedo solo con las primeras 5 columnas nuevas
    auction_list_0 = auction_list_0.iloc[:, :5]
    # renombro las columnas
    auction_list_0.columns = ["auction_list_0_" + str(i) for i in range(5)]

    # concateno las columnas nuevas
    df = pd.concat([df, auction_list_0], axis=1)
    # a ver como quedo el df 
    print(df.head())
    print(df.shape)

    # ahora distintas combinaciones de las columnas separadas
    df["auction_list_0_0_1"] = df["auction_list_0_0"] + df["auction_list_0_1"]
    df["auction_list_0_0_2"] = df["auction_list_0_0"] + df["auction_list_0_2"]
    df["auction_list_0_1_2"] = df["auction_list_0_1"] + df["auction_list_0_2"]
    df["auction_list_0_1_3"] = df["auction_list_0_1"] + df["auction_list_0_3"]
    df["auction_list_0_2_3"] = df["auction_list_0_2"] + df["auction_list_0_3"]
    df["auction_list_0_2_4"] = df["auction_list_0_2"] + df["auction_list_0_4"]
    df["auction_list_0_3_4"] = df["auction_list_0_3"] + df["auction_list_0_4"]
    df["auction_list_0_0_1_2"] = df["auction_list_0_0"] + df["auction_list_0_1"] + df["auction_list_0_2"]
    df["auction_list_0_1_2_3"] = df["auction_list_0_1"] + df["auction_list_0_2"] + df["auction_list_0_3"]



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
    #df["action_list_1_unique"] = df["action_list_1"].apply(lambda x: len(set(clean_and_convert(x))) if len(clean_and_convert(x)) > 0 else 0)
    #df["action_list_2_unique"] = df["action_list_2"].apply(lambda x: len(set(clean_and_convert(x))) if len(clean_and_convert(x)) > 0 else 0)

    print("Dataframe transformado, cantidad de columnas: ")
    # imprimo las columnas
    print(df.shape[1])
    ctr_15 = df

print("cantidad columnas: " + str(ctr_15.shape[1]))

# consigo los indices de las variables categoricas (ya que el catboost las maneja nativamente) pero tienen que ser en string o int 
categorical_cols = ctr_15.select_dtypes(exclude='number').columns
# convierto a str los objetos porque el catboost es exquisito (no objetos)
ctr_15[categorical_cols] = ctr_15[categorical_cols].astype(str)

print("cantidad columnas: " + str(ctr_15.shape[1]))

# datos de etnrenamiento (agregar el 21)
train_data, val_data = train_test_split(ctr_15, test_size=0.15, random_state=1234)
train_data, test_data = train_test_split(train_data, test_size=0.15, random_state=1234)



print("Datos separados")
print("Cantidad de datos de entrenamiento: " + str(train_data.shape[0]))
print("Cantidad de datos de validacion: " + str(val_data.shape[0]))
print("Cantidad de datos de test: " + str(test_data.shape[0]))
print("Cantidad de columnas: " + str(train_data.shape[1]))


## nos quedamos cortos de ram
del ctr_15
gc.collect()

y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
y_val = val_data["Label"]
X_val = val_data.drop(columns=["Label"])
y_test = test_data["Label"]
X_test = test_data.drop(columns=["Label"])



gc.collect()


# optimizar los hiperparametros


# probar menos learning rate y menos depth
# CatBoost
print("Training the CatBoost model...")
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=8, loss_function='Logloss', eval_metric='AUC', random_seed=11, verbose=True, early_stopping_rounds=100)
catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list())

# Predict on the evaluation set
y_preds = catboost_model.predict_proba(X_val)[:, catboost_model.classes_ == 1].squeeze()
print("AUC on the validation set: " + str(roc_auc_score(y_val, y_preds)))

# Predict on the test set
y_preds = catboost_model.predict_proba(X_test)[:, catboost_model.classes_ == 1].squeeze()
print("AUC on the test set: " + str(roc_auc_score(y_test, y_preds)))

# subset selection
print("Feature importances:")
feature_importances = catboost_model.get_feature_importance(prettified=True)
print(feature_importances)

# me guardo las features importantes
feature_importances.to_csv(MODEL_NAME + "_feature_importances.csv", index=False)


#Best AUC: 0.8626837758530653
#Best depth: 8
#Best learning rate: 0.01