import re
import gc
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


#Cargamos los datos
MODEL_NAME = "catboost_model"
PROPORTION_TRAIN = 0.25
PROPORTION_OPT = 1/5

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

#Seteamos las etapas a correr
analisis = False
opt = False
entrenamiento = False

#Funciones auxiliares y transformaciones a utilizar

# Función para limpiar y convertir a enteros
def clean_and_convert(lst_str):
    if isinstance(lst_str, str) and len(lst_str) > 0:
        # Limpiamos los caracteres no numéricos
        clean_list = [re.sub(r'\D', '', i) for i in lst_str.split(",")]
        # Filtramos los elementos que no sean vacíos tras la limpieza
        clean_list = [int(i) for i in clean_list if i.isdigit()]
        return clean_list
    return []

#Transformaciones
df_count = 1
for df in [ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21, ctr_test, eval_data]:
    #Borramos columnas
    df = df.drop(columns=["action_categorical_5", "action_categorical_6", "auction_categorical_5", "creative_categorical_11", "auction_categorical_10", "creative_categorical_1", "auction_categorical_2", "timezone_offset", "auction_boolean_1", "auction_categorical_3", "auction_boolean_2", "creative_width", "has_video", "creative_categorical_7", "auction_boolean_0", "creative_categorical_10"])
    #Convertimos a datetime
    df["auction_time"] = pd.to_datetime(df["auction_time"], unit='s')
    df["day"] = df["auction_time"].dt.day
    df["hour"] = df["auction_time"].dt.hour
    df["minute"] = df["auction_time"].dt.minute
    df["fullMinute"] = df["auction_time"].dt.hour * 60 + df["auction_time"].dt.minute
    df["weekday"] = df["auction_time"].dt.weekday
    #Seno y coseno de la hora
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    #Seno y coseno del minuto
    df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
    df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)
    df["fullMinute_sin"] = np.sin(2 * np.pi * df["fullMinute"] / 1440)
    df["fullMinute_cos"] = np.cos(2 * np.pi * df["fullMinute"] / 1440)
    #Drop auction_time
    df = df.drop(columns=["auction_time"])

    #Tamaños
    df["auction_list_0_size"] = df["auction_list_0"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df["action_list_2_size"] = df["action_list_2"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)

    #Info de auction list 0
    df["auction_list_0_size"] = df["auction_list_0"].apply(lambda x: len(x.split(",")) if isinstance(x, str) else 0)
    df["auction_list_0_unique"] = df["auction_list_0"].apply(lambda x: len(set(x.split(","))) if isinstance(x, str) else 0)

    #Aeparo las listas en columnas solo con auction_list_0
    auction_list_0 = df["auction_list_0"].str.split(",", expand=True)
    #Limpio los [ y ] de las columnas nuevas
    auction_list_0 = auction_list_0.apply(lambda x: x.str.replace("[", ""))
    auction_list_0 = auction_list_0.apply(lambda x: x.str.replace("]", ""))
    #Me quedo solo con las primeras 5 columnas nuevas
    auction_list_0 = auction_list_0.iloc[:, :5]
    #Renombro las columnas
    auction_list_0.columns = ["auction_list_0_" + str(i) for i in range(5)]

    #Concateno las columnas nuevas
    df = pd.concat([df, auction_list_0], axis=1)

    #Ahora distintas combinaciones de las columnas separadas
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

    print("Dataframe transformado " + str(df_count) + " de 8")
    df_count += 1
    print(df.head())
    print(df.shape)


print("cantidad de columnas: " + str(ctr_15.shape[1]))

#Concateno datos de entrenamiento
train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21])


#Para evitar problemas con la RAM
del ctr_15, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21
gc.collect()

#Tomo un porcentaje de los datos para el analisis
train_data_a = train_data.sample(frac=0.1, random_state=1234)

print("columas de tiempo y listas transformadas")

#Etapa de analisis:
if analisis:
    data_with_label_1 = train_data_a[train_data_a["Label"] == 1]
    data_with_label_0 = train_data_a[train_data_a["Label"] == 0].sample(frac=0.1, random_state=1234)

    ## graficamos un histograma de label = 1 contra weekday y hour
    # superponemos los histogramas y los guardamos
    sns.histplot(data_with_label_1["weekday"], bins=7, color='blue', alpha=0.5, label='label 1')
    sns.histplot(data_with_label_0["weekday"], bins=7, color='red', alpha=0.5, label='label 0')
    plt.savefig("weekday.png")
    plt.clf()

    sns.histplot(data_with_label_1["hour"], bins=24, color='blue', alpha=0.5, label='label 1')
    sns.histplot(data_with_label_0["hour"], bins=24, color='red', alpha=0.5, label='label 0')
    plt.savefig("hour.png")
    plt.clf()

    print("graficos guardados")


#Division de datos para optimizacion y entrenamiento

#Al tener una leve desproporcion de los datos, optamos por reducir la cantidad de label 0
data_with_label_1 = train_data[train_data["Label"] == 1]
data_with_label_0_train = train_data[train_data["Label"] == 0].sample(frac=PROPORTION_TRAIN, random_state=1234)
data_with_label_0_opt = train_data[train_data["Label"] == 0].sample(frac=PROPORTION_OPT, random_state=1234)

train_data = pd.concat([data_with_label_1, data_with_label_0])
opt_data = pd.concat([data_with_label_1, data_with_label_0])


y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

y_opt = opt_data["Label"]
X_opt = opt_data.drop(columns=["Label"])


# consigo los indices de las variables categoricas (ya que el catboost las maneja nativamente) pero tienen que ser en string o int 
categorical_cols_train = X_train.select_dtypes(exclude='number').columns
# convierto a str los objetos porque el catboost es exquisito (no objetos)
X_train[categorical_cols_train] = X_train[categorical_cols_train].astype(str)
eval_data[categorical_cols_train] = eval_data[categorical_cols_train].astype(str)

# consigo los indices de las variables categoricas (ya que el catboost las maneja nativamente) pero tienen que ser en string o int 
categorical_cols_opt = X_opt.select_dtypes(exclude='number').columns
# convierto a str los objetos porque el catboost es exquisito (no objetos)
X_opt[categorical_cols_opt] = X_opt[categorical_cols_opt].astype(str)
ctr_test[categorical_cols_opt] = ctr_test[categorical_cols_opt].astype(str)

# tamaño de los datos
print("cantidad columnas: " + str(X_train.shape[1]))
print("cantidad filas: " + str(X_train.shape[0]))

del train_data, data_with_label_0_train, data_with_label_0_opt, data_with_label_1
gc.collect()

#Etapa de optimizacion:
if opt:

    # Función principal para el grid search
    def perform_manual_gridsearch(X_train, y_train, X_val, y_val, categorical_cols):
        # Definir los hiperparámetros que se evaluarán
        iteraciones = [500, 750, 1000]
        depths = [4, 6, 8, 10]
        learning_rates = [0.01, 0.05, 0.1]
        
        aucs = []  # Lista para almacenar los resultados de AUC-ROC y parámetros
        
        # Grid search manual
        for iter in iteraciones:
            for depth in depths:
                for learning in learning_rates:
                    print(f"Entrenando con iteraciones={iter}, profundidad={depth}, learning_rate={learning}")
                    
                    # Crear el modelo con los hiperparámetros actuales
                    catboost_model = CatBoostClassifier(
                        iterations=iter,
                        learning_rate=learning,
                        depth=depth,
                        loss_function='Logloss',
                        eval_metric='AUC',
                        random_seed=11,
                        verbose=False,
                        early_stopping_rounds=100
                    )
                    
                    # Entrenar el modelo
                    catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list(), eval_set=(X_val, y_val), verbose=False)
                    
                    # Obtener las predicciones para el conjunto de validación
                    y_val_pred = catboost_model.predict_proba(X_val)[:, 1]  # Obtener las probabilidades de clase 1
                    
                    # Calcular el AUC-ROC
                    auc = roc_auc_score(y_val, y_val_pred)
                    print(f"AUC-ROC obtenido: {auc:.4f}")
                    
                    # Guardar los resultados en la lista
                    aucs.append((iter, depth, learning, auc))
        
        return aucs

    # Función para seleccionar la mejor combinación de hiperparámetros
    def seleccionar_mejores_parametros(aucs):
        # Buscar el valor máximo de AUC-ROC y sus parámetros asociados
        mejor_combinacion = max(aucs, key=lambda x: x[3])  # Ordenar por el AUC-ROC (4to elemento de cada tupla)
        
        print("\nMejores hiperparámetros encontrados:")
        print(f"Iteraciones: {mejor_combinacion[0]}")
        print(f"Profundidad: {mejor_combinacion[1]}")
        print(f"Learning Rate: {mejor_combinacion[2]}")
        print(f"Mejor AUC-ROC: {mejor_combinacion[3]:.4f}")
        
        return mejor_combinacion
    
    # Dividir en conjunto de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X_opt, y_opt, test_size=0.2, random_state=42)
    
    # Ejecutar el grid search manual
    aucs = perform_manual_gridsearch(X_train, y_train, X_val, y_val, categorical_cols_opt)
    
    # Seleccionar la mejor combinación de hiperparámetros
    mejor_combinacion = seleccionar_mejores_parametros(aucs)
    best_depth = mejor_combinacion[1]
    best_learning = mejor_combinacion[2]
    best_iter =  mejor_combinacion[0]
    
    # Guardar los resultados en un archivo CSV
    aucs_df = pd.DataFrame(aucs, columns=['iteraciones', 'profundidad', 'learning_rate', 'auc_roc'])
    aucs_df.to_csv("resultados_auc_roc.csv", index=False)
    
    print("\nLos resultados del GridSearch manual se han guardado en 'resultados_auc_roc.csv'.")


else:
    best_depth = 8
    best_learning = 0.01
    best_iter = 500
    #Best AUC: 0.8626837758530653
    #Best depth: 8
    #Best learning rate: 0.01

#Etapa de entrenamiento de modelo
if entrenamiento:
    # CatBoost
    print("Training the CatBoost model...")
    catboost_model = CatBoostClassifier(iterations=best_iter, learning_rate=best_learning, depth=best_depth, loss_function='Logloss', eval_metric='AUC', random_seed=11, verbose=True, early_stopping_rounds=100)
    catboost_model.fit(X_train, y_train, cat_features=categorical_cols_train.to_list())

    #Usamos predict_proba en lugar de predict porque mejora el auc
    y_preds = catboost_model.predict_proba(eval_data)[:, 1]
    submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
    submission_df["id"] = submission_df["id"].astype(int)
    submission_df.to_csv(MODEL_NAME + ".csv", sep=",", index=False)

    feature_importances = catboost_model.get_feature_importance(prettified=True)
    print("Feature importances")
    feature_importances.to_csv(MODEL_NAME + "_feature_importances.csv", index=False)
    print(feature_importances)


    print("Se entreno el " + MODEL_NAME + " con un: " + str(PROPORTION_TRAIN) + " de los datos")