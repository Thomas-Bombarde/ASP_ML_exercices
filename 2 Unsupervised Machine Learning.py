import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_iris
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# Exercice 1
# 1.a.
housing = fetch_california_housing()
X = housing["data"]

# 1.b.
polyfunc = PolynomialFeatures(degree=2, include_bias=False)
polynoms = polyfunc.fit(X)
fn = polyfunc.get_feature_names(housing.feature_names)
print(f"There are {len(fn)} features")

# 1.c.
df = pd.DataFrame(polynoms, columns = fn)
df["y"] = housing["target"]
df.to_csv("./output/polynomials.csv")

# Exercice 2
# 2.a.
olympics = pd.read_csv("./data/olympics.csv", index_col=0)
olympics.dtypes
olympics.describe()

# I drop the score column for three reasons
# First, The matrix's index is sorted by score, so dropping score would not lose information on score ranking.
# Second, score is a linear combination of the other covariates. This makes the rank of the matrix smaller than its
# dimension, and would prevent any OLS regressions we would want to make using a new target variable.
# Third, the score seems to be the outcome of interest, so in fact the target variable. It should thus not be included
# in the design matrix, but in its own seperate column.
# Therefore, I drop the score column and store it in its own vector.
olympics_score = olympics["score"]
olympics = olympics.drop(columns = ["score"])

# 2.b.
scaler = StandardScaler()
scaler.fit(olympics)
n_olympics = pd.DataFrame(scaler.transform(olympics), columns=olympics.columns)
n_olympics.var(axis=0)
# Subject to computational error, the previous vector shows that each
# variable has unit variance.

# 2.c.
pca = PCA(random_state=42)
pca.fit(n_olympics)
n_olympics_components = pd.DataFrame(pca.components_, columns = olympics.columns)
# The most influential covariates in the 1st component are the 100m sprint, 110m hurdles, and running long.
# The most influential covariates in the 2nd component are the discus throw, pole vault, 1,500m run.
# The most influential covariates in the 3rd component are the high jump, 100m, and the 1,500 run.
# These are the variables in which there is the most variance and which are hence the most informative.

# 2.d.
df = pd.DataFrame(pca.explained_variance_ratio_, columns=["Explained Variance ratio"])
df["Cumulative"] = df["Explained Variance ratio"].cumsum()
# We need at least 7 covariates to explain at least 90% of the variance.

# Exercice 3.

# 3.a.
iris = load_iris()
x = iris["data"]

# 3.b.
scaler = StandardScaler()
scaler.fit(x)
x_unscaled = pd.DataFrame(iris["data"], columns=iris["feature_names"]) #useful later
x = pd.DataFrame(scaler.transform(x), columns=iris["feature_names"])

# 3.c.
kmeans = KMeans(n_clusters=3, random_state=42)
agg = AgglomerativeClustering(n_clusters=3)
dbscan = DBSCAN(eps=1, min_samples=2)
kmeans.fit(x)
agg.fit(x)
dbscan.fit(x)

df = pd.DataFrame({"Kmeans": kmeans.labels_, "Agg": agg.labels_, "DBSCAN": dbscan.labels_})

# 3.d.
print(f"K-means silhouette score: {silhouette_score(x, kmeans.labels_)} \n "
      f"Agglomerative clustering silhouette score:{silhouette_score(x, agg.labels_)} \n"
      f"DBSCAN clustering silhouette score{silhouette_score(x, dbscan.labels_)} \n")
# DBSCAN presents the highest silouhette score. It is the only to leave
# data points as noise, rather than clustering all observations. Agglomerative and kmeans clustering
# excludes no data points to noise and assign all to mutually exclusive cluseters.
# Hence DBSCAN must be treated differently.

# 3.e.
df["sepal width"] = x_unscaled["sepal width (cm)"]
df["petal length"] = x_unscaled["petal length (cm)"]

# 3.f.
df["DBSCAN"] = df["DBSCAN"].replace(-1, "Noise")

# 3.g.
df["id"] = df.index
value_vars = ["Kmeans", "Agg",  "DBSCAN"]
id_vars = ["sepal width", "petal length"]
melted = df.melt(id_vars=id_vars, value_vars=value_vars)
fig = sns.relplot(data = melted, x="sepal width", y="petal length", hue="value", col="variable")
fig.figure.savefig("./output/cluster_petal.pdf.")

# There are points that appear as clear outliers omitted by DBCSAN.
# This is intuitive for a classification problem concerned with identifying groups within the data,
# rather than assigning all observations to the most similar.
