import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, Normalizer
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

housing = fetch_california_housing()
X = housing["data"]
poly = PolynomialFeatures(degree=2, include_bias=False)
df = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names(housing.feature_names))
df["y"] = housing["target"]
df.to_csv("./output/polynomials.csv")


""""
Exercice 2
"""
olympics = pd.read_csv("./data/olympics.csv", index_col=0)
olympics.head()
olympics.dtypes
olympics.describe
# The matrix's index is sorted by score, so abandonning score would not lose information on score ranking.
# Further, score is a linear combination of the other covariates. It has hence appears as the outcome of interest.
# This would make score the Y vector to be predicted by X the design matrix of performance on indiviudal tasks. Therfore,
# it would make sense to assign it its own vector drop score from the dataframe.
olympics = olympics.drop(columns = ["score"])

# (b)
scaler = StandardScaler()
scaler.fit(olympics)
nolympics = pd.DataFrame(scaler.transform(olympics), columns=olympics.columns)
nolympics.var(axis=0)

# (c)
pca = PCA(random_state=42)
pca.fit(nolympics)
comp = pca.components_  # how does it represent covariates v componenets
df = pd.DataFrame(pca.explained_variance_ratio_, columns=["Explained Variance ratio"])
df["Cumulative"] = df["Explained Variance ratio"].cumsum()
# Taking components as the rows, the most influential covariates in 0 are 5, and 1 in decreasing order of influence.
# This corresponds to doing well on the 100m, the 400m, and the long jump, which might indicate that these skills
# are the most valuable to the decathlon. T
# These seem to correspond to a more agile profile.
# (d)
# We need at least 6 covariates to explain at least 90% of the variance.

"""
Exercice 3
"""
iris = load_iris()
x = iris["data"]
scaler = StandardScaler()
scaler.fit(x)
x = pd.DataFrame(scaler.transform(x), columns=iris["feature_names"])

# kmeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x)
kmeansdf = pd.DataFrame(kmeans.fit_transform(x), columns=kmeans.get_feature_names_out())

# Agglomerative Clustering
agg = AgglomerativeClustering(n_clusters=3)
agg.fit(x)
aggdf = pd.DataFrame(agg.get_params(), columns=agg.feature_names_in_)

# DBSCAN
dbscan = DBSCAN(eps=1, min_samples=2)
dbscan.fit(x)
dbscandf = pd.DataFrame(dbscan.components_, columns=dbscan.feature_names_in_)

# (d)
df = pd.DataFrame(x, columns=iris["feature_names"])
df["kmeans"] = kmeans.labels_
df["agg"] = agg.labels_
df["dbscan"] = dbscan.labels_ #-1 means noise - nothing to do with other clusters
df.drop(columns = ["sepal length (cm)", "petal width (cm)"])
df.head()

print(silhouette_score(x, kmeans.labels_))
print(silhouette_score(x, agg.labels_))
print(silhouette_score(x, dbscan.labels_))
#DBSCAN presents the highest silouhette score. It is the only to leave
#data points as noise, rather than clustering all observations, hence
#must be treated differently.

#(f)
df["dbscan"] = df["dbscan"].replace(-1, "Noise")

#(g)
df["id"] = df.index
value_vars = ["kmeans", "agg",  "dbscan"]
id_vars = ["sepal width (cm)", "petal length (cm)"]
melted = df.melt(id_vars=id_vars, value_vars=value_vars)
fig = sns.catplot(data = melted, x="sepal width (cm)", y="petal length (cm)", hue="value", col="variable")
fig.figure.savefig("./output/cluster_petal.pdf.")
#there are points that appear as clear outliers omitted by dbscam: this is intuitively convenient for a classification problem.
