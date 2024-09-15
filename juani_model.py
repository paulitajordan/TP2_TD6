import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy import sparse
import numpy as np

# Proportion of the data to use for training
PROPORTION = 1/10

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

# Load the train data
train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20])

# Split the train data into train and test
test_data = ctr_21
y_test = ctr_21["Label"]
X_test = ctr_21.drop(columns=["Label"])

# Load the test data
eval_data = ctr_test

# Train a tree on the train data
train_data = train_data.sample(frac=PROPORTION)  # Use only 1/n of the data for training
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
#numeric_cols = X_train.select_dtypes(include='number').columns
categoric_cols = X_train.select_dtypes(exclude='number').columns

# Crea el transformador para manejar variables numéricas y categóricas
preprocessor = ColumnTransformer(
    transformers=[
        #('num', SimpleImputer(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categoric_cols)
        # los 2 param. de OneHotEncoder son para que no falle si hay valores desconocidos y para que use matrices ralas
    ])

del train_data
gc.collect()


print("Training the model...")
X_train_transformed = preprocessor.fit_transform(X_train)
x_test_transformed = preprocessor.transform(X_test)

# grid search
#depths = [30]
#min_samples = [100]
#for i in depths:
#    for j in min_samples:
#        cls = DecisionTreeClassifier(max_depth=i, min_samples_split= j, random_state=2345)
#        cls.fit(X_train_transformed, y_train)
#        y_preds_test = cls.predict_proba(x_test_transformed)[:, cls.classes_ == 1].squeeze()
#        auc = roc_auc_score(y_test, y_preds_test)
#        print("AUC: ", auc, " depth: ", i, " min_samples: ", j)
    



cls = DecisionTreeClassifier(max_depth=30, min_samples_split= 100, random_state=2345)
cls.fit(X_train_transformed, y_train)
print("Model trained!")


# Predict on the evaluation set

X_eval = eval_data.drop(columns=["id"])
X_eval_transformed = preprocessor.transform(X_eval) ## aca transforma las variables categóricas igual que con los datos de entrenamiento


y_preds = cls.predict_proba(X_eval_transformed)[:, cls.classes_ == 1].squeeze()



# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)

print("Done!, using: " + str(PROPORTION) + " proportion of the data")

##VALIDACOIÓN
y_preds_test = cls.predict_proba(x_test_transformed)[:, cls.classes_ == 1].squeeze()
auc = roc_auc_score(y_test, y_preds_test)
print("AUC: ", auc, " day 21")
print("categorical data")



# agregue one hot encoder para las variables categóricas
# saque el pipeline porque se rompía con lo de arriba
# aumente el max_depth a 20
# saque el min_impurity_decrease
# me guardo el ultimo dia para testear // volver a comentar para entrenar con todos los datos