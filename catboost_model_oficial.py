import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import re
import numpy as np
from sklearn.linear_model import LogisticRegression

# IDEAS
#  probar pasando los ids a categoricos

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
print("cantidad de columnas: " + str(ctr_15.shape[1]))


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
df_count = 1
for df in [ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21,eval_data]:
    # borramos columnas
    df = df.drop(columns=["action_categorical_5", "action_categorical_6", "auction_categorical_5", "creative_categorical_11", "auction_categorical_10", "creative_categorical_1", "auction_categorical_2", "timezone_offset", "auction_boolean_1", "auction_categorical_3", "auction_boolean_2", "creative_width", "has_video", "creative_categorical_7", "auction_boolean_0", "creative_categorical_10"])


    # convertimos a datetime
    df["auction_time"] = pd.to_datetime(df["auction_time"], unit='s')
    df["day"] = df["auction_time"].dt.day
    df["hour"] = df["auction_time"].dt.hour
    df["minute"] = df["auction_time"].dt.minute
    df["fullMinute"] = df["auction_time"].dt.hour * 60 + df["auction_time"].dt.minute
    df["weekday"] = df["auction_time"].dt.weekday
    # seno y coseno de la hora
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    # seno y coseno del minuto
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["fullMinute_sin"] = np.sin(2 * np.pi * df["fullMinute"] / 1440)
    df["fullMinute_cos"] = np.cos(2 * np.pi * df["fullMinute"] / 1440)
    # drop auction_time
    df = df.drop(columns=["auction_time"])

    # Trabajamos las listas

    # estandarizamos (normal (0,1)) las variables numericas que vamos a usar abajo
    #df["auction_bidfloor_std"] = (df["auction_bidfloor"] - df["auction_bidfloor"].mean()) / df["auction_bidfloor"].std()
    #df["auction_age"] = (df["auction_age"] - df["auction_age"].mean()) / df["auction_age"].std()
    #df["creative_height"] = (df["creative_height"] - df["creative_height"].mean()) / df["creative_height"].std()
    #df["creative_width"] = (df["creative_width"] - df["creative_width"].mean()) / df["creative_width"].std()



    # Tamaños
    df["auction_list_0_size"] = df["auction_list_0"].apply(lambda x: len(x) if isinstance(x, list) else 0)
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
    # SACAARRRRR ESTOSSS############################################
    #df["action_list_1_unique"] = df["action_list_1"].apply(lambda x: len(set(clean_and_convert(x))) if len(clean_and_convert(x)) > 0 else 0)
    #df["action_list_2_unique"] = df["action_list_2"].apply(lambda x: len(set(clean_and_convert(x))) if len(clean_and_convert(x)) > 0 else 0)
    ############################################################
    print("Dataframe transformado " + str(df_count) + " de 8")
    df_count += 1
    print(df.head())
    print(df.shape)


print("cantidad de columnas: " + str(ctr_15.shape[1]))

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

del train_data, data_with_label_0, data_with_label_1
gc.collect()


# probar menos learning rate y menos depth
# CatBoost
print("Training the CatBoost model...")
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=8, loss_function='Logloss', eval_metric='AUC', random_seed=11, verbose=True, early_stopping_rounds=100)
catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list())

#usamos predict_proba en lugar de predict porque mejora el auc
y_preds = catboost_model.predict_proba(eval_data)[:, 1]
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv(MODEL_NAME + ".csv", sep=",", index=False)

feature_importances = catboost_model.get_feature_importance(prettified=True)
print("Feature importances")
print(feature_importances)


print("Se entreno el " + MODEL_NAME + " con un: " + str(PROPORTION) + " de los datos")

print(" training another catboost")
# CatBoost

catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.01, depth=8, loss_function='Logloss', eval_metric='AUC', random_seed=11, verbose=True, early_stopping_rounds=100)
catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list())

y_preds = catboost_model.predict_proba(eval_data)[:, 1]
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv(MODEL_NAME + "_2.csv", sep=",", index=False)

print("Se entreno el " + MODEL_NAME + "_2" + " con un: " + str(PROPORTION) + " de los datos")

print(" training another catboost")
# CatBoost
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=8, loss_function='Logloss', eval_metric='AUC', random_seed=11, verbose=True, early_stopping_rounds=100, class_weights=[1, 4])
catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list())

y_preds = catboost_model.predict_proba(eval_data)[:, 1]
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv(MODEL_NAME + "_3.csv", sep=",", index=False)

print("Se entreno el " + MODEL_NAME + "_3" + " con un: " + str(PROPORTION) + " de los datos")

