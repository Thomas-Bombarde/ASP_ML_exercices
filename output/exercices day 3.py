from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import PolynomialFeatures

breast_cancer = load_breast_cancer()
X = breast_cancer["data"]
PolynomialFeatures.fit(X)
features = PolynomialFeatures(degree=2, *, interaction_only=False, include_bias=false, order='C')