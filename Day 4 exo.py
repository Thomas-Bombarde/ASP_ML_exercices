import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline

polynoms = pd.read_csv("./output/polynomials.csv")

X = polynoms.drop(columns="y")
y = polynoms["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Regression
lm = OLS().fit(X_train, y_train)
ridge = Ridge(alpha = 0.1).fit(X_train, y_train)
lasso = Lasso(alpha = 0.1).fit(X_train, y_train)

lm.score(X_test, y_test)
ridge.score(X_test, y_test)
lasso.score(X_test, y_test)

#LASSO returns the smallest value for R-squared, so presents the smallest residual mean squared error
#and so performs best in minimizing the variance

df = pd.DataFrame(lm.coef_, X.columns, columns=["OLS"])
df["Lasso"] = lasso.coef_
df["Ridge"] = ridge.coef_
sum((df["Lasso"] == 0) & (df["Ridge"] != 0)) # =15as plt
df.plot.barh().set_aspect(30/10)

"""
Exercice 2
"""
diabetes = load_diabetes()
X = diabetes["data"]
y = diabetes["target"]
diabetes[""]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Model Pipelines
algorithms = [("scaler", StandardScaler()),
               ("nn", MLPRegressor(solver="adam", random_state=42, max_iter=1000))] #MinMax is the variables, regrossor the values?
pipe = Pipeline(algorithms, verbose=True) #what does verbose mean
param_grid = {"nn__hidden_layer_sizes": [(75, 75), (90, 90), (100, 100)], #__ means it takes the key word
              "nn__alpha": [0.001, 0.0025, 0.005]}
grid = GridSearchCV(pipe, param_grid, cv=3)
grid.fit(X_train, y_train)
params = pd.DataFrame(grid.cv_results_)
best_params = grid.best_params_
grid.best_score_
#best paramters were: {'nn__alpha': 0.001, 'nn__hidden_layer_sizes': (75, 75)}
#they scored best with 0.4509621148158042

best_estimator = grid.best_estimator_
tmp = best_estimator._final_estimator
tmp.coefs_[0]
df = pd.DataFrame(tmp.coefs_[0])
#(d)
sns.heatmap(df,yticklabels=diabetes["feature_names"]) #probably wrong


"""Exercice 3"""
cancer = load_breast_cancer()
X = cancer["data"]
y = cancer["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


algorithms = [("scaler", MinMaxScaler()),
            ("nn", MLPRegressor(solver="lbfgs", random_state=42, max_iter=1000))] #use lbfgs because it is a small dataset
pipe = Pipeline(algorithms, verbose=True)
param_grid = {"nn__hidden_layer_sizes": [(75, 75), (100, 100)],
              "nn__alpha": [0.001, 0.005]}
grid = GridSearchCV(pipe, param_grid, scoring="roc_auc")
grid.fit(X_train, y_train)
grid.best_estimator_
#The best estimator is that with  alpha=0.001, hidden_layer_sizes = (100, 100). It's score is:
grid.best_score_
#because it is close to 1. This means the total negative and total positive curves intersect very little.
# Therefore, distinguishes well between positive class and negative class. It is a good generalisation.


#(d)
results = pd.DataFrame(grid.cv_results_)
scores = results["mean_test_score"].values.reshape(2,2)
sns.heatmap(scores, annot=True,
            xticklabels=param_grid["nn__hidden_layer_sizes"],
            yticklabels=param_grid["nn__alpha"])
