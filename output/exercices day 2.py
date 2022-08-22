"""
#Exercice 1
"""

import pandas as pd
import seaborn as sns

df1 = pd.DataFrame(sns.load_dataset("tips"))
df1.head()
df1 = df1.replace({"Mon": "Monday", "Tue": "Tuesday", "Wed": "Wednesday",
                   "Thur": "Thursday", "Fri": "Friday", "Sat": "Saturday",
                   "Sun": "Sunday"})

fig = sns.relplot(x="tip", y="total_bill", data=df1, hue="day", row="sex")
fig.set(xlabel='tips in $',
        ylabel='total bill in $',
        title='Tips per total bill faceted by sex')

fig.savefig("./output/tips.pdf")

"""
#Exercice 2
"""

# (a)
df2 = pd.read_csv("https://raw.githubusercontent.com/justmarkham/DAT8/master/data/u.user", sep="|")

# (b)
last_10 = df2.iloc[(len(df2) - 10):len(df2)]
first_25 = df2.iloc[0:25]
print(f"The last 10 rows:\n {last_10} \n and the first 25:\n {first_25}")

# (c)
df2.dtypes

# (d)
occupations = df2["occupation"].value_counts()
print(f"There are {len(occupations)} diffrent occuptions."
      f"The most frequent occupation is {occupations.idxmax()}")

id = occupations.index
occupations = occupations.sort_index()
fig = sns.barplot(x=id, y=occupations)
# if occupations is not sorted by index, counts do not corresponde to hteir variable.
fig.figure.savefig("./output/occupations.pdf")

"""
Exercice 3
"""

iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                   names=["sepal length (in cm)", "sepal width (in cm)", "petal length (in cm)", "petal width (in cm)",
                          "class"])

iris.loc[10:29, "petal length (in cm)"] = "missing"
iris = iris.replace("missing", 1)
iris.to_csv("./output/iris.csv")
cols = ["sepal length (in cm)", "sepal width (in cm)", "petal length (in cm)", "petal width (in cm)", "class"]

#for i in range(0,len(cols)):
#sns.catplot(data=iris, y = cols[i], x = "class", kind="bar")
g = sns.FacetGrid(iris, col= iris.index)
g.map_dataframe(sns.catplot, x="class")

iris["id"] = iris.index
value_vars = ["sepal length (in cm)", "sepal width (in cm)", "petal length (in cm)", "petal width (in cm)"]
id_vars = ["class"]
melted = iris.melt(id_vars=id_vars, value_vars=value_vars)

fig = sns.catplot(data=melted, x="variable", y="value", hue="class")
fig.set_xticklabels(labels=["sepal length (in cm)", "sepal width (in cm)", "petal length (in cm)", "petal width (in cm)"], rotation = 45)
fig.figure.savefig("./output/iris.pdf.")

#Exercice 4

df3 = pd.read_csv("https://query.data.world/s/wsjbxdqhw6z6izgdxijv5p2lfqh7gx")
df3.info()
df3.info(memory_usage="deep")
#the second specifies in more detail the memory used
dobjects = df3.select_dtypes(include=[object])
dobjects.describe()
#day_of_week and acquisition info contain less than 7 unique values
#converting the object to category type would
df3["day_of_week"] = df3["day_of_week"].astype("category")
df3["acquisition_info"] = df3["acquisition_info"].astype("category")
#dtype could havev been specified in read_csv
dnumeric = df3.select_dtypes(include=[float, int])
dnumeric.to_csv("./output/dnumeric.csv")
dnumeric.to_feather("./output/dnumeric.csv")
