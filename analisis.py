#En este archivo vamos a analizar el dataset para poder tomar decisiones de ingenieria de datos eficientemente para mejorar el modelo
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Función para preprocesar los datos
def preprocess_data(df):
    # Identificar las columnas categóricas
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    # Codificar las variables categóricas con LabelEncoder
    le = LabelEncoder()
    for col in categorical_columns:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# Función para generar la lista de las 25 correlaciones más fuertes
def top_n_correlations(df, n=25):
    corr_matrix = df.corr()
    # Obtener la matriz de correlaciones sin la diagonal
    corr_pairs = corr_matrix.unstack().reset_index()
    corr_pairs.columns = ['Variable_1', 'Variable_2', 'Correlation']
    
    # Filtrar correlaciones duplicadas y diagonales (1.0)
    corr_pairs = corr_pairs[corr_pairs['Variable_1'] != corr_pairs['Variable_2']]
    
    # Ordenar por correlación absoluta
    top_correlations = corr_pairs.iloc[np.abs(corr_pairs['Correlation']).argsort()[::-1]].head(n)
    
    return top_correlations

# Función para generar y filtrar la matriz de correlación
# def generate_filtered_correlation_matrix(df, n=25):
#     # Seleccionar solo las columnas numéricas para la matriz de correlación
#     numeric_df = df.select_dtypes(include=[np.number])
    
#     # Generar la matriz de correlación
#     correlation_matrix = numeric_df.corr()

#     # Obtener el top N de las correlaciones más fuertes
#     top_correlations = top_n_correlations(correlation_matrix, n)
    
#     # Mostrar los resultados de las principales correlaciones
#     print("\nTop {} correlaciones más fuertes (positivas o negativas):".format(n))
#     print(top_correlations)
    
#     # Filtrar el dataframe para incluir solo las variables con las correlaciones más fuertes
#     top_vars = pd.unique(top_correlations[['Variable_1', 'Variable_2']].values.ravel('K'))
#     filtered_corr_matrix = correlation_matrix.loc[top_vars, top_vars]
    
#     # Graficar la matriz de correlación filtrada
#     plt.figure(figsize=(16, 12))
#     sns.heatmap(filtered_corr_matrix, annot=True, cmap='coolwarm')
#     plt.title('Matriz de Correlación Filtrada (Top {} Correlaciones)'.format(n))
#     plt.show()

#     return filtered_corr_matrix, top_correlations

# Función para capturar relaciones complejas entre las variables con RandomForest
def feature_importance(df, target_column):
    # Separar las características (X) y la variable objetivo (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Entrenar un RandomForestClassifier para estimar la importancia de las variables
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Extraer la importancia de las características
    feature_importances = pd.DataFrame({'Feature': X.columns, 
                                        'Importance': model.feature_importances_})
    feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

    # Mostrar las importancias de las características
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importances)
    plt.title('Importancia de las Variables')
    plt.show()

    return feature_importances

# Función principal para ejecutar todo el proceso
def main(target_column):
    ctr_15 = pd.read_csv("datos/ctr_15.csv")
    ctr_16 = pd.read_csv("datos/ctr_16.csv")
    ctr_17 = pd.read_csv("datos/ctr_17.csv")
    ctr_18 = pd.read_csv("datos/ctr_18.csv")
    ctr_19 = pd.read_csv("datos/ctr_19.csv")
    ctr_20 = pd.read_csv("datos/ctr_20.csv")
    ctr_21 = pd.read_csv("datos/ctr_21.csv")

    # ctr_15["Dia"] = 1
    # ctr_16["Dia"] = 2
    # ctr_17["Dia"] = 3
    # ctr_18["Dia"] = 4
    # ctr_19["Dia"] = 5
    # ctr_20["Dia"] = 6
    # ctr_21["Dia"] = 7

    # Combinar los datasets
    train = pd.concat([ctr_15, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21])
    
    # Preprocesar los datos para codificar variables categóricas
    train = preprocess_data(train)
    
    # Generar la matriz de correlación filtrada y el top 20 de correlaciones
    top_correlations = top_n_correlations(train, n=25)
    print(top_correlations)
    
    # Capturar relaciones complejas entre las variables
    feature_importances = feature_importance(train, target_column)
    
    return top_correlations, feature_importances

# Columna objetivo
target_column = 'Label'

# Ejecutar el análisis
top_correlations, feature_importances = main(target_column)