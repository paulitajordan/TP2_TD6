import pandas as pd
import gc
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline

# Print all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


ctr_15 = pd.read_csv ("ctr_15.csv")
ctr_16 = pd.read_csv ("ctr_16.csv")
ctr_17 = pd.read_csv ("ctr_17.csv")
ctr_18 = pd.read_csv ("ctr_18.csv")
ctr_19 = pd.read_csv ("ctr_19.csv")
ctr_20 = pd.read_csv ("ctr_20.csv")
ctr_21 = pd.read_csv ("ctr_21.csv")
ctr_test = pd.read_csv ("ctr_test.csv")

ctr_15["Dia"] = 1
ctr_16["Dia"] = 2
ctr_17["Dia"] = 3
ctr_18["Dia"] = 4
ctr_19["Dia"] = 5
ctr_20["Dia"] = 6
ctr_21["Dia"] = 7
ctr_test["Dia"] = 8

# Load the train data
train_data = pd.concat([ctr_15,ctr_16,ctr_17,ctr_18,ctr_19,ctr_20,ctr_21])

# Load the test data
eval_data = ctr_test

# Train a tree on the train data
train_data = train_data.sample(frac=1/10)
y_train = train_data["Label"]
X_train = train_data.drop(columns=["Label"])
X_train = X_train.select_dtypes(include='number')
del train_data
gc.collect()

cls = make_pipeline(SimpleImputer(), DecisionTreeClassifier(max_depth=10, min_samples_split= 100, min_impurity_decrease= 0.001, random_state=2345)) #Le asignamos un valor a min_impurity_decrease
cls.fit(X_train, y_train)

# Predict on the evaluation set
eval_data = eval_data.select_dtypes(include='number')
y_preds = cls.predict_proba(eval_data.drop(columns=["id"]))[:, cls.classes_ == 1].squeeze()

# Make the submission file
submission_df = pd.DataFrame({"id": eval_data["id"], "Label": y_preds})
submission_df["id"] = submission_df["id"].astype(int)
submission_df.to_csv("basic_model.csv", sep=",", index=False)
