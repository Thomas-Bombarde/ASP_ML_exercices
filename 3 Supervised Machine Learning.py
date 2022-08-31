import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline

# Exercise 1

# 1.a.
polynoms = pd.read_csv("./output/polynomials.csv")

# 1.b.
X = polynoms.drop(columns="y")
y = polynoms["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 1.c.
lm = OLS().fit(X_train, y_train)
ridge = Ridge(alpha=0.1).fit(X_train, y_train)
lasso = Lasso(alpha=0.1).fit(X_train, y_train)

lm.score(X_test, y_test)
ridge.score(X_test, y_test)
lasso.score(X_test, y_test)

# LASSO returns the R-squared, so performs best in minimizing the variance.
# For this reason, LASSO arguably makes the best prediction.

# 1.d.
df = pd.DataFrame(lm.coef_, X.columns, columns=["OLS"])
df["Lasso"] = lasso.coef_
df["Ridge"] = ridge.coef_
n = sum((df["Lasso"] == 0) & (df["Ridge"] != 0))
print(f"There are {n} observations such that the lasso coefficient is zero and the Ridge coefficient is non-zero.")

# 1.e.
df.plot.barh().set_aspect(30 / 10)
plt.savefig('./output/polynomials.pdf')

# Exercise 2

# 2.a.
diabetes = load_diabetes()
X = diabetes["data"]
y = diabetes["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 2.b.
# Use the solver appropriate for smaller datasets.
algorithms = [("scaler", StandardScaler()),
              ("nn", MLPRegressor(solver="lbfgs", random_state=42, max_iter=1000))]
pipe = Pipeline(algorithms, verbose=True)
param_grid = {"nn__hidden_layer_sizes": [(75, 75), (90, 90), (100, 100)],
              "nn__alpha": [0.001, 0.0025, 0.005]}
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train, y_train)

# 2.c.
params = pd.DataFrame(grid.cv_results_)
best_params = grid.best_params_
grid.best_score_
print(f"The best parameters were: {best_params}")
print(f"They scored {grid.best_score_}")
# With a low score, the model generalises poorly.

# 2.d.
best_estimator = grid.best_estimator_
tmp = best_estimator._final_estimator
df = pd.DataFrame(tmp.coefs_[0])
sns.heatmap(df, yticklabels=diabetes["feature_names"])  # probably wrong

# Exercise 3

# 3.a.
cancer = load_breast_cancer()
X = cancer["data"]
y = cancer["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# 3.c.
algorithms = [("scaler", MinMaxScaler()),
              ("nn", MLPClassifier(solver="lbfgs", random_state=42,
                                   max_iter=1000))]  # use solver for a small dataset
pipe = Pipeline(algorithms, verbose=True)
param_grid = {"nn__hidden_layer_sizes": [(75, 75), (100, 100)],
              "nn__alpha": [0.001, 0.005]}
grid = GridSearchCV(pipe, param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)

print(f"The best estimator has the following parameters: \n {grid.best_estimator_} \n and score: \n {grid.best_score_}")
# The RoC-AuC score is close to 1. This means the total negative and total positive curves intersect very little.
# Therefore, the model distinguishes well between positive class and negative class.
# It is a good generalisation.


# 3.d.
preds = grid.predict(X_test)
matrix = confusion_matrix(y_test, preds)
sns.heatmap(matrix, annot=True,
            xticklabels=cancer["target_names"],
            yticklabels=cancer["target_names"])
plt.savefig("./output/nn_breast_confusion.pdf")
