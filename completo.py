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
PROPORTION_OPT = 1/100

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


#Etapa de analisis:

if analisis:
    # Establecer el estilo de los gráficos
    sns.set(style="whitegrid")

    # Función para preprocesar los datos
    def preprocess_data(df):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        le = LabelEncoder()
        for col in categorical_columns:
            df[col] = le.fit_transform(df[col].astype(str))
        return df

    # Función para generar la lista de las 25 correlaciones más fuertes
    def top_n_correlations(df, threshold=0.4):
        corr_matrix = df.corr()
        # Filtrar correlaciones según el umbral absoluto
        corr_pairs = corr_matrix.unstack().reset_index()
        corr_pairs.columns = ['Variable_1', 'Variable_2', 'Correlation']
        corr_pairs = corr_pairs[(corr_pairs['Variable_1'] != corr_pairs['Variable_2']) & 
                                (np.abs(corr_pairs['Correlation']) > threshold)]
        
        # Ordenar por correlación absoluta y tomar los 25 más altos
        top_correlations = corr_pairs.iloc[np.abs(corr_pairs['Correlation']).argsort()[::-1]].head(25)
        return top_correlations

    # Función para capturar relaciones complejas entre las variables con RandomForest
    def feature_importance(df, target_column):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
        plt.title('Importancia de las Variables', fontsize=16)
        plt.xlabel('Importancia', fontsize=14)
        plt.ylabel('Características', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        for index, value in enumerate(feature_importances['Importance']):
            plt.text(value, index, f'{value:.2f}', color='black', ha='left', va='center')
        plt.tight_layout()
        plt.show()

        return feature_importances

    # Función para generar histogramas de frecuencia
    def plot_frequency_histograms(df):
        # Extraer el mes, día, hora y año
        df["auction_time"] = pd.to_datetime(df["auction_time"], unit='s')
        df["month"] = df["auction_time"].dt.month
        df["day"] = df["auction_time"].dt.day
        df["hour"] = df["auction_time"].dt.hour
        df["year"] = df["auction_time"].dt.year
        
        # Crear histogramas
        plt.figure(figsize=(14, 10))

        plt.subplot(2, 2, 1)
        sns.histplot(df["year"], bins=df["year"].nunique(), kde=False, color='skyblue')
        plt.title('Frecuencia por Año')
        plt.xlabel('Año')
        plt.ylabel('Frecuencia')

        plt.subplot(2, 2, 2)
        sns.histplot(df["month"], bins=12, kde=False, color='lightgreen')
        plt.title('Frecuencia por Mes')
        plt.xlabel('Mes')
        plt.ylabel('Frecuencia')

        plt.subplot(2, 2, 3)
        sns.histplot(df["day"], bins=31, kde=False, color='salmon')
        plt.title('Frecuencia por Día')
        plt.xlabel('Día')
        plt.ylabel('Frecuencia')

        plt.subplot(2, 2, 4)
        sns.histplot(df["hour"], bins=24, kde=False, color='orange')
        plt.title('Frecuencia por Hora')
        plt.xlabel('Hora')
        plt.ylabel('Frecuencia')

        plt.tight_layout()
        plt.show()

    # Función principal para ejecutar todo el proceso
    def main(target_column):

        train = pd.concat([ctr_15, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21])
        train = train.sample(frac=0.1, random_state=42)
        train = preprocess_data(train)

        # Generar las correlaciones filtradas
        top_correlations = top_n_correlations(train, threshold=0.4)
        print("Top correlaciones con abs(Correlación) > 0.4:")
        print(top_correlations)

        # Visualización de la matriz de correlación filtrada
        correlation_matrix = train.corr()
        filtered_vars = pd.unique(top_correlations[['Variable_1', 'Variable_2']].values.ravel('K'))
        filtered_corr_matrix = correlation_matrix.loc[filtered_vars, filtered_vars]

        plt.figure(figsize=(12, 10))
        sns.heatmap(filtered_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True,
                    cbar_kws={"shrink": .8}, linewidths=0.5, linecolor='black')
        plt.title('Matriz de Correlación (Filtrada)', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

        feature_importances = feature_importance(train, target_column)

        # Generar histogramas de frecuencia
        plot_frequency_histograms(train)

        return top_correlations, feature_importances

    # Columna objetivo
    target_column = 'Label'

    # Ejecutar el análisis
    top_correlations, feature_importances = main(target_column)

#Funciones auxiliares y transformaciones a utilizar en optimizacion y train

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
for df in [ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21,eval_data]:
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

#Etapa de optimizacion:
if opt:
    pass
else:
    best_depth: 8
    best_learning: 0.01
    #Best AUC: 0.8626837758530653
    #Best depth: 8
    #Best learning rate: 0.01

#Etapa de entrenamiento de modelo
if entrenamiento:

    #Concateno datos de entrenamiento
    train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21])

    # saco la mitad de los datos con label 0
    data_with_label_1 = train_data[train_data["Label"] == 1]
    data_with_label_0 = train_data[train_data["Label"] == 0].sample(frac=PROPORTION_TRAIN, random_state=1234)

    train_data = pd.concat([data_with_label_1, data_with_label_0])


    #Para evitar problemas con la RAM
    del ctr_15, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21
    gc.collect()

    #Hago un shuffle
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


    # CatBoost
    print("Training the CatBoost model...")
    catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=8, loss_function='Logloss', eval_metric='AUC', random_seed=11, verbose=True, early_stopping_rounds=100)
    catboost_model.fit(X_train, y_train, cat_features=categorical_cols.to_list())

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