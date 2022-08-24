import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression as OLS
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.preprocessing import StandardScaler
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
plt.barh()
plt.barh(width = ["Lasso", "Ridge", "OLS"])
plt.bar(data = df, x=df.index, height="Ridge", color="blue")
plt.bar(data = df, x=df.index, height="OLS", color="yellow")

df = pd.DataFrame({'speed': speed,
                   'lifespan': lifespan}, index=index)

ax = df.plot.barh()

"""
Exercice 3
"""
cancer = load_breast_cancer()
X = cancer["data"]
y = cancer ["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

#Model Pipelines
algorithms = [ ("scaler", StandardScaler()),
               ("nn", MLPRegressor(solver="adam", random_state=42, max_iter=1000))] #MinMax is the variables, regrossor the values?
pipe = Pipeline(algorithms, verbose=True) #what does verbose mean
param_grid = {"nn__hidden_layer_sizes": [(75, 75), (90, 90), (100, 100)], #__ means it takes the key word
              "nn__alpha": [0.001, 0.0025, 0.005]}
grid = GridSearchCV(pipe, param_grid)
grid.fit(X_train, y_train)
#how to check the number of cross-validations??
#plot_feature_importances_cancer()

scores = grid.score(X_test, y_test)
sns.heatmap(scores, annot=True, xticklabels = param_grid["nn__hidden_layer_sizes"], yticklabels = param_grid["nn__alpha"])