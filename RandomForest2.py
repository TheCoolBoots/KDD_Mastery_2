import copy
import json
import numpy as np
import pandas as pd
import random
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


class RandomForestRegression():
    def __init__(self):
        self.forest = None
        self.default = None
        self.numDefault = 0


    def entropy(self, y:pd.Series):
        valueCounts = y.value_counts().to_frame()
        valueCounts['probability'] = valueCounts[y.name] / len(y)
        
        rollingSum = 0
        
        for val in y.unique():
            rollingSum += (-valueCounts.loc[val]['probability']) * np.log2(valueCounts.loc[val]['probability'])
            
        return rollingSum


    def gain(self, y:pd.Series, x:pd.Series):
        frame = pd.DataFrame()
        frame['target'] = y
        frame['source'] = x
        weightedAvg = 0
        for name, group in frame.groupby('source'):
            weightedAvg += len(group)/len(frame) * self.entropy(group['target'])
        
        return self.entropy(y) - weightedAvg


    def gain_ratio(self, y:pd.Series, x:pd.Series):
        g = self.gain(y,x)
        return g/self.entropy(y)


    # if you want to print like me :)
    def print_tree(self, tree:dict) -> list:
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


    def generate_rules(self, tree) -> list:

        if type(tree) != dict:
            if type(tree) == pd.Series:
                return tree.iloc[0]
            return tree
        
        output = []
        for col in tree.keys():
            for val in tree[col]:
                rule = (col, val)
                children = self.generate_rules(tree[col][val])
                if type(children) != list:
                    output.append([rule, children])
                else:
                    for child in children:
                        child.insert(0, rule)
                        output.append(child)
        return output


    def select_split2(self, X, y):
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
                    gr = self.gain_ratio(y, newCol)
                    
                    # if it is the best split to make, apply the function to the dataframe
                    if gr > maxGR:
                        maxCol = "%s<%.2f"%(col, splitpoint)
                        maxGR = gr
                        colReplace = newCol.astype(str)
            else:
                gr = self.gain_ratio(y, X[col])  
                if gr > maxGR:
                    maxCol = col
                    maxGR = gr
                    colReplace = X[maxCol]

            
        # print(X)
        return maxCol, maxGR, colReplace


    def make_prediction(self, rules, x, default):
        for rule in rules:
            matches = True  
            for cond in rule[:-1]:
                splitVal, xVal = cond
                if '<' not in splitVal:
                    if x.loc[splitVal] != xVal:
                        matches = False
                        break
                else:
                    col, val = splitVal.split('<')
                    val = float(val)
                    if xVal == 'True':
                        if x.loc[col] >= val:
                            matches = False
                            break
                    else:
                        if x.loc[col] < val:
                            matches = False
                            break
            if matches:
                return rule[-1]

        self.numDefault += 1
        return (default)


    # returns random decision tree and lsit of out-of-bag datapoints
    def createRandomTree(self, X:pd.DataFrame, y:pd.Series, min_split_count=30, numFeaturesPerStep = 5) -> dict:
        if len(X.columns) == 0:
            if len(y) == 0:
                return 0
            return y.mean()
        elif y.std() < .0001:
            return y.mean()
        # if all rows have the same values
        elif len(X.value_counts()) == 1:
            return y.mean()
        elif len(X) < min_split_count:
            return y.mean()
        
        if numFeaturesPerStep > len(X.columns):
            numFeaturesPerStep = len(X.columns)
        randomCols = random.sample(list(X.columns), numFeaturesPerStep)

        maxCol, maxGR, colReplace = self.select_split2(X[randomCols], y)
        if '<' in maxCol:
            baseCol = maxCol.split('<')[0]
            X = X.drop(columns=[baseCol])
        X[maxCol] = colReplace
        # print(X)

        retDict = {maxCol : {}}
        for val in X[maxCol].unique():
            mask = X[maxCol] == val

            retDict[maxCol][val] = self.createRandomTree(X.loc[mask].drop(columns=[maxCol]), y.loc[mask], min_split_count, numFeaturesPerStep)

        return retDict


    def createRandomForest(self, X:pd.DataFrame, y:pd.Series, numTrees = 30):
        return [self.createRandomTree(X, y, 10, 5) for i in range(numTrees)]


    def predictRandomForest(self, xRow:pd.Series):
        predictionSum = 0

        for tree in self.forest:
            rules = self.generate_rules(tree)
            predictionSum += self.make_prediction(rules, xRow, self.default)

        # print(f'default: {self.numDefault}')
        return predictionSum/len(self.forest)


    def fit(self, X_train:pd.DataFrame, y_train:pd.Series):
        self.default = y_train.mean()
        self.forest = self.createRandomForest(X_train, y_train)


    def predict(self, X_test:pd.DataFrame) -> pd.Series:
        predictions = X_test.apply(lambda row: self.predictRandomForest(self.forest, row), axis=1)
        return predictions


    def evaluate(self, X_test:pd.DataFrame, y_test:pd.DataFrame):
        predictions = X_test.apply(lambda row: self.predictRandomForest(row), axis=1)
        print(f'Mean Squared Error={mean_squared_error(y_test, predictions)}')
        print(f'Mean Absolute Error={mean_absolute_error(y_test, predictions)}')



# titanicDF = pd.read_csv('titanic.csv')[['Pclass', 'Age', 'Sex', 'Survived']].dropna()
# X = titanicDF[['Pclass', 'Age', 'Sex']]
# X['Sex'] = (X['Sex'] == 'male').astype(int)
# print(X)
# t = titanicDF['Survived']

# from preprocessing import importData
# ethDF = importData()
# X = ethDF.drop(columns='ETH')
# t = ethDF['ETH']

wineDataset = pd.read_csv('winequality-red.csv')
X = wineDataset.drop(columns=['quality'])
t = wineDataset['quality']

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.3, random_state = 1)

# print(titanicDF)

rfr = RandomForestRegression()
rfr.fit(X_train, t_train)

rfr.evaluate(X_test, t_test)

# numTrees = 30, minSplitCount = 20, featuresPerStep = 5
# Mean Squared Error=0.1352501462769949
# Mean Absolute Error=0.2632076205085186

from sklearn.ensemble import RandomForestRegressor

rfr2 = RandomForestRegressor()
rfr2.fit(X_train, t_train)
pred = rfr2.predict(X_test)
mse_RF = mean_squared_error(t_test, pred)
mae_RF = mean_absolute_error(t_test, pred)
print('Mean squared error using Random Forest: ', mse_RF)
print('Mean absolute error Using Random Forest: ', mae_RF)

# Mean squared error using Random Forest:  0.15666062142934958
# Mean absolute error Using Random Forest:  0.2442331995485894

# for wine dataset:
# Mean Squared Error=0.3894567688962042
# Mean Absolute Error=0.47783052193201153
# Mean squared error using Random Forest:  0.3277495833333333  
# Mean absolute error Using Random Forest:  0.42900000000000005