import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df["flower"] = iris.target
df.drop(["sepal length (cm)", "sepal width (cm)"], axis="columns", inplace=True)

scaler = MinMaxScaler()
scaler.fit(df[["petal length (cm)"]])
df["petal length (cm)"] = scaler.transform(df[["petal length (cm)"]])
scaler.fit(df[["petal width (cm)"]])
df["petal width (cm)"] = scaler.transform(df[["petal width (cm)"]])

km = KMeans(n_clusters=3)
y_predicter = km.fit_predict(df[["petal length (cm)", "petal width (cm)"]])
df["cluster"] = y_predicter

df0 = df[df.cluster == 0]
df1 = df[df.cluster == 1]
df2 = df[df.cluster == 2]

plt.scatter(df0["petal length (cm)"], df0["petal width (cm)"], color="red")
plt.scatter(df1["petal length (cm)"], df1["petal width (cm)"], color="blue")
plt.scatter(df2["petal length (cm)"], df2["petal width (cm)"], color="green")
plt.xlabel("petal length (cm)")
plt.ylabel("petal width (cm)")
plt.show()