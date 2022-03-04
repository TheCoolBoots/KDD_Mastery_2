import unittest
import json
from RandomForest import *
from RandomForestHelper import *

class test_RandomForest(unittest.TestCase):



    def test_boostrap(self):
        data = {'a': [1,2,3,4],
                'b': [5,6,7,8],
                'c': [9,10,11,12]}
        df = pd.DataFrame(data)

        actual = getBootstrapSample(df)
        self.assertEqual(len(df), len(actual))

    def test_entropy(self):

        titanic_df = pd.read_csv("titanic.csv")

        features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Survived']
        titanic_df2 = titanic_df.loc[:,features]
        titanic_df2['CabinLetter'] = titanic_df2['Cabin'].str.slice(0,1)
        X = titanic_df2.drop('Cabin',axis=1)#.dropna()
        X['CabinLetter'] = X['CabinLetter'].fillna("?")
        X['Pclass'] = X['Pclass'].astype(str)
        X['SibSp'] = X['SibSp'].astype(str)
        X['Parch'] = X['Parch'].astype(str)
        X = X.dropna()

        X2 = X.copy()
        X2['Age'] = pd.cut(X2['Age'],bins=20).astype(str) # bin Age up
        X2['Age'].value_counts()

        X2['Fare'] = pd.cut(X2['Fare'],bins=20).astype(str) # bin Age up
        X2['Fare'].value_counts()

        t = titanic_df.loc[X2.index,'Survived']

        e1 = entropy(t)
        e2 = entropy(X2['CabinLetter'])
        self.assertAlmostEqual(e1, 0.9735190023846809)
        self.assertAlmostEqual(e2, 1.4723461729538008)

    def test_gain_ratio(self):

        titanic_df = pd.read_csv("titanic.csv")

        features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked','Survived']
        titanic_df2 = titanic_df.loc[:,features]
        titanic_df2['CabinLetter'] = titanic_df2['Cabin'].str.slice(0,1)
        X = titanic_df2.drop('Cabin',axis=1)#.dropna()
        X['CabinLetter'] = X['CabinLetter'].fillna("?")
        X['Pclass'] = X['Pclass'].astype(str)
        X['SibSp'] = X['SibSp'].astype(str)
        X['Parch'] = X['Parch'].astype(str)
        X = X.dropna()

        X2 = X.copy()
        X2['Age'] = pd.cut(X2['Age'],bins=20).astype(str) # bin Age up
        X2['Age'].value_counts()

        X2['Fare'] = pd.cut(X2['Fare'],bins=20).astype(str) # bin Age up
        X2['Fare'].value_counts()

        t = titanic_df.loc[X2.index,'Survived']

        g1 = gain_ratio(X2, 'Sex', 'Survived')
        g3 = gain_ratio(X2, 'Age', 'Survived')
        g2 = gain_ratio(X2, 'Pclass', 'Survived')
        self.assertAlmostEqual(g1, 0.21993234062330022)
        self.assertAlmostEqual(g2, 0.09656717982753726)
        self.assertAlmostEqual(g3, 0.03813317995550127)



if __name__ == '__main__':
    unittest.main()