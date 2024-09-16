import pandas as pd
import gc
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import ast


MODEL_NAME = "catboost_model"
PROPORTION = 1/10 # Cant datos
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
    df["auction_time"] = pd.to_datetime(df["auction_time"])
    df["year"] = df["auction_time"].dt.year
    df["month"] = df["auction_time"].dt.month
    df["day"] = df["auction_time"].dt.day
    df["hour"] = df["auction_time"].dt.hour
    df["minute"] = df["auction_time"].dt.minute
    df["second"] = df["auction_time"].dt.second
    df["weekday"] = df["auction_time"].dt.weekday

# datos de etnrenamiento (agregar el 21)
train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20])
if not TEST:
    train_data = pd.concat([train_data,ctr_21])

train_data = train_data.sample(frac=PROPORTION, random_state=1234)

y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])

print("Datos cargados")

## Vi que hay 2 listas con valores que parecen ser ids, entonces hacemos one hot encoding
def one_hot_encode_lists(df, list_columns, mlb_dict=None):
    if mlb_dict is None:
        mlb_dict = {}
    
    for col in list_columns:
        # Convertimos los strings que representan listas en listas reales solo si está en formato string
        df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x)
        
        # Aseguramos que todas las entradas sean listas, incluyendo aquellas que ya son listas
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        
        if col not in mlb_dict:
            mlb_dict[col] = MultiLabelBinarizer()
            df_expanded = pd.DataFrame(mlb_dict[col].fit_transform(df[col]),
                                       columns=[f"{col}_{cls}" for cls in mlb_dict[col].classes_],
                                       index=df.index)
        else:
            df_expanded = pd.DataFrame(mlb_dict[col].transform(df[col]),
                                       columns=[f"{col}_{cls}" for cls in mlb_dict[col].classes_],
                                       index=df.index)
        
        # Concatenamos las nuevas columnas al DataFrame original y eliminamos la columna original
        df = pd.concat([df, df_expanded], axis=1).drop(columns=[col])
    
    return df, mlb_dict

# expandir las columnas que contienen listas
list_columns = ['action_list_1', 'action_list_2']
X_train, mlb_dict = one_hot_encode_lists(X_train, list_columns)
if not TEST:
    eval_data,_ = one_hot_encode_lists(eval_data, list_columns, mlb_dict)
    X_train, eval_data = X_train.align(eval_data, join='outer', axis=1, fill_value=0)


# consigo los indices de las variables categoricas (ya que el catboost las maneja nativamente) pero tienen que ser en string o int 
categorical_cols = X_train.select_dtypes(exclude='number').columns
catgorical_indices = [X_train.columns.get_loc(col) for col in categorical_cols]
if not TEST:
    categorical_cols_eval = eval_data.select_dtypes(exclude='number').columns
    categorical_indices_eval = [eval_data.columns.get_loc(col) for col in categorical_cols_eval]



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
catboost_model = CatBoostClassifier()
print("Training the CatBoost model...")
catboost_model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=10, loss_function='Logloss', eval_metric='AUC', random_seed=2345, verbose=100)
catboost_model.fit(X_train, y_train, cat_features=catgorical_indices)

if not TEST:
    # nos creamos el archivo de submission
    assert X_train.columns.equals(eval_data.columns)
    y_preds = catboost_model.predict_proba(eval_data)[:, catboost_model.classes_ == 1].squeeze()

    submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
    submission_df["id"] = submission_df["id"].astype(int)
    submission_df.to_csv(MODEL_NAME + ".csv", sep=",", index=False)

print("Se entreno el " + MODEL_NAME + " con un: " + str(PROPORTION) + " de los datos")


# me mate la ram
#del X_train
del y_train
del eval_data
gc.collect()


# Testeamos a ver como nos va a ir
y_test = ctr_21["Label"]
X_test = ctr_21.drop(columns=["Label"])
X_test,_ = one_hot_encode_lists(X_test, list_columns, mlb_dict)
X_test, _ = X_test.align(X_train, join='outer', axis=1, fill_value=0)
categorical_cols_test = X_test.select_dtypes(exclude='number').columns
X_test[categorical_cols_test] = X_test[categorical_cols_test].astype(str)
y_preds_test = catboost_model.predict_proba(X_test)[:, catboost_model.classes_ == 1].squeeze()
roc_auc = roc_auc_score(y_test, y_preds_test)
print("ROC AUC: " + str(roc_auc) + " en el set de testeo (21) usando el " + str(PROPORTION) + " de los datos")
