import pandas as pd
import numpy as np
from typing import Tuple
from RandomForestHelper import *
import random


def getBootstrapSample(data:pd.DataFrame):
    randlist = pd.DataFrame(index=np.random.randint(len(data), size=len(data)))
    return data.merge(randlist, left_index=True, right_index=True, how='right')

# returns random decision tree and lsit of out-of-bag datapoints
def createRandomTree(data:pd.DataFrame, xCols:list[str], targetCol:str, numFeaturesPerStep:int) -> Tuple[dict, pd.DataFrame]:
    # if all rows have the same target value
    if (data[targetCol] == data[targetCol].iloc[0]).all():
        return data[targetCol].iloc[0]
    # if all rows have the same x values
    elif len(data.drop(columns=[targetCol]).value_counts()) == 1:
        return data[targetCol].mode().get(0)
    else:
        # choose numFeaturesPerStep columns to consider and remove them from consideration down the line
        if numFeaturesPerStep > len(xCols):
            numFeaturesPerStep = len(xCols)
        randomCols = random.sample(xCols, numFeaturesPerStep)
        for col in randomCols:
            xCols.remove(col)
        randomCols.append(targetCol)
        maxCol, maxGR = select_split(data[randomCols], targetCol)

        if type(xCols) != list:
            xCols = [xCols]

        return {maxCol : {val : createRandomTree(data[data[maxCol] == val], xCols, targetCol, numFeaturesPerStep) for val in data[maxCol].unique()}}


# returns list of random decision trees and DF of out-of-bag datapoints
def createRandomForest(trainData:pd.DataFrame, targetCol:str, numTrees:int) -> Tuple[list[dict], pd.DataFrame]:
    xCols = trainData.columns.tolist()
    xCols.remove(targetCol)

    return [createRandomTree(trainData, xCols, targetCol, 2) for i in range(numTrees)]



# makes a prediction based on generated random forest
def predictRandomForest(datapoint:pd.DataFrame, trees:list[dict]):
    pass


# evaluates accuracy of random forest based on given datapoints
def evaluateRandomForest(testData:pd.DataFrame, targetCol:str, trees:list[dict]):
    pass



titanicDF = pd.read_csv('titanic.csv')[['Pclass', 'Sex','Survived']]
# print(titanicDF)
forest = createRandomForest(titanicDF, 'Survived', 3)