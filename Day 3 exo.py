import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

#(a) get data
housing = fetch_california_housing()
X = housing["data"]

#(b) get polynomials, feature names
poly = PolynomialFeatures(degree=2, include_bias=False)
fn = poly.get_feature_names(housing.feature_names)

#(c) save to csv
df = pd.DataFrame(poly.fit_transform(X), columns = fn)
df["y"] = housing["target"]
df.to_csv("./output/polynomials.csv")


""""
Exercice 2
"""

#(a) get data
olympics = pd.read_csv("./data/olympics.csv", index_col=0)

#(b)print descriptives
olympics.dtypes
olympics.describe()


# The matrix's index is sorted by score, so abandonning score would not lose information on score ranking.
# Further, score is a linear combination of the other covariates. It has hence appears as the outcome of interest.
# This would make score the target vector to be predicted by X, the design matrix of performance on individual tasks.
# Therefore, it would make sense to assign it its own vector drop score from the dataframe.
olympics = olympics.drop(columns = ["score"])

# (c)
scaler = StandardScaler()
scaler.fit(olympics)
n_olympics = pd.DataFrame(scaler.transform(olympics), columns=olympics.columns)

#check it has unit variance
n_olympics.var(axis=0)

#(c)
pca = PCA(random_state=42)
pca.fit(n_olympics)
n_olympics_components = pd.DataFrame(pca.components_, columns = olympics.columns)
# The most influential covariates in the 1st component are the 100m sprint, 110m hurdles, and running long.
# The most influential covariates in the 2nd component are the discus throw, pole vault, 1,500m run.
# The most influential covariates in the 3rd component are the high jump, 100m, and the 1,500 run.
# These are the variables in which there is the most variance and most informative.

#(d)
df = pd.DataFrame(pca.explained_variance_ratio_, columns=["Explained Variance ratio"])
df["Cumulative"] = df["Explained Variance ratio"].cumsum()
# We need at least 7 covariates to explain at least 90% of the variance.

"""
Exercice 3
"""

#(a) get data
iris = load_iris()
x = iris["data"]

#(b) scale
scaler = StandardScaler()
scaler.fit(x)
x_unscaled = pd.DataFrame(iris["data"], columns=iris["feature_names"])
x = pd.DataFrame(scaler.transform(x), columns=iris["feature_names"])


# (c) kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)

# (d)
#Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(x)
#DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2)
dbscan.fit(x)

#join
df = pd.DataFrame(kmeans.labels_, columns = ["Kmeans"])
df["Agg"] = agg.labels_
df["DBSCAN"] = dbscan.labels_#-1 means noise - nothing to do with other clusters

print(f"K-means silouhette score: {silhouette_score(x, kmeans.labels_)} \n "
      f"Agglomerative clustering silouhette score:{silhouette_score(x, agg.labels_)} \n"
      f"DBSCAN clustering silouhette score{silhouette_score(x, dbscan.labels_)} \n")
#DBSCAN presents the highest silouhette score. It is the only to leave
#data points as noise, rather than clustering all observations. Agglomerative and kmeans clustering
#excludes no data points to noise and assign all to mutually exclusive cluseters.
#Hence DBSCAN must be treated differently.
#DBSCAN has the highest silouhette score.

#(e)
df["sepal width"] = x_unscaled["sepal width (cm)"]
df["petal length"] = x_unscaled["petal length (cm)"]

#(f)
df["DBSCAN"] = df["DBSCAN"].replace(-1, "Noise")

#(g)
df["id"] = df.index
value_vars = ["Kmeans", "Agg",  "DBSCAN"]
id_vars = ["sepal width", "petal length"]
melted = df.melt(id_vars=id_vars, value_vars=value_vars)
fig = sns.relplot(data = melted, x="sepal width", y="petal length", hue="value", col="variable")
fig.figure.savefig("./output/cluster_petal.pdf.")
#there are points that appear as clear outliers omitted by dbscam: this is intuitively convenient for a classification problem.
