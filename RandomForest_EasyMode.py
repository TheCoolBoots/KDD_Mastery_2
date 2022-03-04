import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import random
from sklearn.model_selection import train_test_split

# returns random decision tree and lsit of out-of-bag datapoints
def createRandomTree(X:pd.DataFrame, y:pd.Series, min_split_count=30, numFeaturesPerStep = 5) -> dict:
    if y.std() < .0001:
        return y.mean().get(0)
    # if all rows have the same values
    elif len(X.value_counts()) == 1:
        # ASK WHAT HAPPENS IF 2 HAVE SAME NUMBER OF INSTANCES
        return y.mean().get(0)
    elif len(X.columns) == 0:
        return y.mean().get(0)
    elif len(X) < min_split_count:
        return y.mean().get(0)

    # choose numFeaturesPerStep columns to consider and remove them from consideration down the line
    if numFeaturesPerStep > len(X.columns):
        numFeaturesPerStep = len(X.columns)
    randomCols = random.sample(list(X.columns), numFeaturesPerStep)

    dt = DecisionTreeRegressor(max_depth=1)
    dt.fit(X, y)
    print(dt.criterion)
    print(dt.class_weight)
    # return {maxCol : {val : createRandomTree(X[X[maxCol] == val], y[X[maxCol] == val]) for val in X[maxCol].unique()}}

titanicDF = pd.read_csv('titanic.csv')[['Pclass', 'Age', 'Sex', 'Survived']].dropna()
X = titanicDF[['Pclass', 'Age', 'Sex']]
t = titanicDF['Survived']
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state = 0)

# print(titanicDF)

tree = createRandomTree(X_train, t_train)

# forest = createRandomForest(X_train, t_train, 2)
# jsonStr = json.dumps(forest)
# with open('forest.json', 'w+') as file:
#     file.write(jsonStr)