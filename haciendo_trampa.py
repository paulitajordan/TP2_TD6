import pandas as pd

# IDEAS
#  probar pasando los ids a categoricos


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
data = pd.concat([ctr_15_full, ctr_16, ctr_17, ctr_18, ctr_19, ctr_20, ctr_21], ignore_index=True)

# me quedo con las filas con label = 1
data = data[data["Label"] == 1]

# me guardo en un csv solo las columnas Label y auction_list_0
data = data[["Label", "auction_list_0"]]
data.to_csv("auctionlists0.csv", index=False)
print("Datos guardados")

# me fijo cuales son las listas que mas se repiten
auction_lists = data["auction_list_0"].value_counts().reset_index()
auction_lists.columns = ["auction_list_0", "count"]
auction_lists.to_csv("auctionlists0_counts.csv", index=False)
print("Datos guardados")

# convierto las listas que estaban en formato str a formato lista para despues poder separarlas

# separo las listas en columnas
data = data["auction_list_0"].str.split(",", expand=True)
# limpio los [ y ] de las columnas
data = data.apply(lambda x: x.str.replace("[", ""))
data = data.apply(lambda x: x.str.replace("]", ""))

# me quedo solo con las primeras 5 columnas nuevas
data = data.iloc[:, :5]



print(" asi quedo la data")
print(data.head())
print(data.shape)

# guardo los datos
data.to_csv("auctionlists0_separadas.csv", index=False)
print("Datos guardados")

