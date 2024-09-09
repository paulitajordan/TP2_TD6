import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

# Cargar los datos
ctr_15 = pd.read_csv("datos/ctr_15.csv")
ctr_16 = pd.read_csv("datos/ctr_16.csv")
ctr_17 = pd.read_csv("datos/ctr_17.csv")
ctr_18 = pd.read_csv("datos/ctr_18.csv")
ctr_19 = pd.read_csv("datos/ctr_19.csv")
ctr_20 = pd.read_csv("datos/ctr_20.csv")
ctr_21 = pd.read_csv("datos/ctr_21.csv")
ctr_test = pd.read_csv("datos/ctr_test.csv")

ctr_15["Dia"] = 1
ctr_16["Dia"] = 2
ctr_17["Dia"] = 3
ctr_18["Dia"] = 4
ctr_19["Dia"] = 5
ctr_20["Dia"] = 6
ctr_21["Dia"] = 7
ctr_test["Dia"] = 8

# Combinar los datos de entrenamiento
train_data = pd.concat([ctr_15, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21])
eval_data = ctr_test

# Preprocesar los datos
train_data = train_data.sample(frac=1/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
X_train = X_train.select_dtypes(include='number')
del train_data
gc.collect()

# Configurar el pipeline
pipeline = make_pipeline(
    SimpleImputer(),
    RandomForestClassifier(random_state=2345)
)

# Definir los parámetros para GridSearch
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100],
    'randomforestclassifier__max_depth': [10, 20, 30],
    'randomforestclassifier__min_samples_split': [ 10, 20]
}

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=2, scoring='roc_auc', n_jobs=-1, verbose=2)

# Ajustar el modelo con los datos de entrenamiento
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Preparar los datos de prueba
eval_data = eval_data.select_dtypes(include='number')

# Hacer predicciones en el conjunto de evaluación
y_preds = best_model.predict_proba(eval_data.drop(columns=["id"]))[:, 1]

# Crear el archivo de envío
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)
