import copy
import json
from typing import Tuple

import numpy as np
import pandas as pd
import random
import Lab3_helper
import json

from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype


def entropy(y:pd.Series):
    valueCounts = y.value_counts().to_frame()
    valueCounts['probability'] = valueCounts[y.name] / len(y)
    
    # print(valueCounts)
    
    rollingSum = 0
    
    for val in y.unique():
        rollingSum += (-valueCounts.loc[val]['probability']) * np.log2(valueCounts.loc[val]['probability'])
        # print(rollingSum)
        
    return rollingSum

    # (0.9735190023846809, 1.4723461729538008)
    # Entropy(S) = sum(- p(y) log2(p(y))) for every y in set S
    # p(y) = probability of value y

def gain(y:pd.Series, x:pd.Series):
    frame = pd.DataFrame()
    frame['target'] = y
    frame['source'] = x
    weightedAvg = 0
    for name, group in frame.groupby('source'):
        weightedAvg += len(group)/len(frame) * entropy(group['target'])
    
    return entropy(y) - weightedAvg

    # (0.21410831283572307, 0.09400998456880616, 0.03712337530803511)

def gain_ratio(y:pd.Series, x:pd.Series):
    g = gain(y,x)
    return g/entropy(y)

    # (0.21993234062330022, 0.09656717982753726, 0.03813317995550127)


# if you want to print like me :)
def print_tree(tree:dict) -> list:
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))


def generate_rules(tree) -> list:

    if type(tree) != dict:
        if type(tree) == pd.Series:
            return tree.iloc[0]
        return tree
    
    output = []
    for col in tree.keys():
        for val in tree[col]:
            rule = (col, val)
            children = generate_rules(tree[col][val])
            if type(children) != list:
                output.append([rule, children])
            else:
                for child in children:
                    child.insert(0, rule)
                    output.append(child)
    return output


def select_split2(X,y):
    maxCol = None
    maxGR = -100000
    
    for col in X.columns:
        # if column is continuous
        if X[col].dtype == np.float64 or (X[col].dtype == np.int64 and len(X[col].unique()) > 20):
            sortedCol = X[col].sort_values()
            
            # iterate through every pair of consecutive values
            for i in range(len(sortedCol) - 1):
                
                # calc the midpoint between the two values
                splitpoint = (sortedCol.iloc[i] + sortedCol.iloc[i+1])/2
                
                # if X[col][i] < splitval, set it to left value, otherwise set it to right value
                # newCol = X[col].apply(lambda val: sortedCol.iloc[i] if val<splitpoint else sortedCol.iloc[i+1])
                newCol = X[col] < splitpoint
                
                # calculate the gain ratio of splitting at the current splitpoint
                # print(newCol.nunique()) # this is sometimes of length 1 not 2?
                gr = gain_ratio(y, newCol)
                
                # if it is the best split to make, apply the function to the dataframe
                if gr > maxGR:
                    maxCol = "%s<%.2f"%(col, splitpoint)
                    maxGR = gr
                    colReplace = newCol.astype(str)
        else:
            gr = gain_ratio(y, X[col])  
            if gr > maxGR:
                maxCol = col
                maxGR = gr
                colReplace = X[maxCol]

        
    # print(X)
    return maxCol, maxGR, colReplace


def make_tree2(X,y,min_split_count=30):
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
    
    maxCol, maxGR = select_split2(X, y)

    retDict = {maxCol : {}}
    for val in X[maxCol].unique():
        mask = X[maxCol] == val

        retDict[maxCol][val] = make_tree2(X[mask].drop(columns=[maxCol]), y[mask], min_split_count)

    return retDict


def make_prediction(rules:list[list[str]], xRow:pd.Series, default):
    # charges = list(filter(convert_and_filter, medical_charges))
    for rule in rules:
        allMatch = True
        for cond in rule[:-1]:
            # print(cond)
            # print(x)
            if xRow.loc[cond[0]] != cond[1]:
                allMatch = False
                break
        if allMatch:
            return rule[-1]

    print('default')
    return (default)


# returns random decision tree and lsit of out-of-bag datapoints
def createRandomTree(X:pd.DataFrame, y:pd.Series, min_split_count=30, numFeaturesPerStep = 5) -> dict:
    if len(X.columns) == 0:
        if len(y) == 0:
            return 0
        return y.mean().get(0)
    elif y.std() < .0001:
        return y.mean().get(0)
    # if all rows have the same values
    elif len(X.value_counts()) == 1:
        return y.mean().get(0)
    elif len(X) < min_split_count:
        return y.mean().get(0)
    
    if numFeaturesPerStep > len(X.columns):
        numFeaturesPerStep = len(X.columns)
    randomCols = random.sample(list(X.columns), numFeaturesPerStep)

    maxCol, maxGR, colReplace = select_split2(X[randomCols], y)
    if '<' in maxCol:
        baseCol = maxCol.split('<')[0]
        X = X.drop(columns=[baseCol])
    X[maxCol] = colReplace
    # print(X)

    retDict = {maxCol : {}}
    for val in X[maxCol].unique():
        mask = X[maxCol] == val

        retDict[maxCol][val] = make_tree2(X.loc[mask].drop(columns=[maxCol]), y.loc[mask], min_split_count)

    return retDict


def createRandomForest(X:pd.DataFrame, y:pd.Series, numTrees = 100):
    return [createRandomTree(X, y, 20) for i in range(numTrees)]


def predictRandomForest(trees:list[dict], xRow:pd.Series):
    predictionSum = 0

    for tree in trees:
        rules = generate_rules(tree)
        predictionSum += make_prediction(rules, xRow, 0)

    return predictionSum/len(trees)


def randomForestRegression(X:pd.DataFrame, y:pd.Series) -> pd.Series:
    forest = createRandomForest(X, y, 100)
    predictions = X.apply(lambda row: predictRandomForest(forest, row), axis=1)
    return predictions

titanicDF = pd.read_csv('titanic.csv')[['Pclass', 'Age', 'Sex', 'Survived']].dropna()
X = titanicDF[['Pclass', 'Age', 'Sex']]
t = titanicDF['Survived']
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state = 0)

# print(titanicDF)

forest = createRandomForest(X_train, t_train, 2)
jsonStr = json.dumps(forest)
with open('forest.json', 'w+') as file:
    file.write(jsonStr)

# y_id3 = X_test.apply(lambda xRow: predictRandomForest(forest, xRow, t_test), axis=1)
# cm_id3 = Lab3_helper.confusion_matrix(t_test,y_id3,labels=[0,1])
# stats_id3 = Lab3_helper.evaluation(cm_id3,positive_class=1)

# print(stats_id3)
