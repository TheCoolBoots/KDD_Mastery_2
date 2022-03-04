import pandas as pd
import numpy as np

def entropy(targets:pd.Series):
    valueCounts = targets.value_counts().to_frame()
    valueCounts['probability'] = valueCounts[targets.name] / len(targets)
    
    rollingSum = 0
    
    for val in targets.unique():
        rollingSum += (-valueCounts.loc[val]['probability']) * np.log2(valueCounts.loc[val]['probability'])
        
    return rollingSum

    # (0.9735190023846809, 1.4723461729538008)


def gain(data:pd.DataFrame, xCol:str, targetCol:str):
    frame = data[[xCol, targetCol]]
    weightedAvg = 0
    for name, group in frame.groupby(xCol):
        weightedAvg += len(group)/len(frame) * entropy(group[targetCol])
    
    return entropy(data[targetCol]) - weightedAvg

def gain_ratio(data:pd.DataFrame, xCol:str, targetCol:str):
    g = gain(data, xCol, targetCol)
    return g/entropy(data[targetCol])

    # (0.21993234062330022, 0.09656717982753726, 0.03813317995550127)

def select_split(data:pd.DataFrame, targetCol:str):
    maxCol = None
    maxGR = 0
    cols = data.columns.tolist()
    cols.remove(targetCol)

    for col in cols:
        gr = gain_ratio(data, col, targetCol)  
        if gr > maxGR:
            maxCol = col
            maxGR = gr

    return maxCol, maxGR
