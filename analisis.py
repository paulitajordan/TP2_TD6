import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

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
    ctr_15 = pd.read_csv("datos/ctr_15.csv")
    ctr_16 = pd.read_csv("datos/ctr_16.csv")
    ctr_17 = pd.read_csv("datos/ctr_17.csv")
    ctr_18 = pd.read_csv("datos/ctr_18.csv")
    ctr_19 = pd.read_csv("datos/ctr_19.csv")
    ctr_20 = pd.read_csv("datos/ctr_20.csv")
    ctr_21 = pd.read_csv("datos/ctr_21.csv")

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
