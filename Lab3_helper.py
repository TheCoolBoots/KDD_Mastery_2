from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def activation(net):
    return 1/(1+np.exp(-net))

def train(X,t,nepochs=200,n=0.5,test_size=0.3,val_size=0.3,seed=0):
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size,random_state=seed)
    X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=val_size,random_state=seed)

    train_accuracy = []
    val_accuracy = []
    nfeatures = X.shape[1]
    np.random.seed(seed)
    w = 2*np.random.uniform(size=(nfeatures,)) - 1
    
    for epoch in range(nepochs):
        y_train2 = X_train2.apply(lambda x: activation(np.dot(w,x)), axis=1)
        y_val = X_val.apply(lambda x: activation(np.dot(w,x)), axis=1)

        train_accuracy.append(sum(t_train2 == np.round(y_train2))/len(t_train2))
        val_accuracy.append(sum(t_val == np.round(y_val))/len(t_val))
                
        for j in range(len(w)):
            w[j] -= n * np.dot((y_train2 - t_train2) * y_train2 * (1 - y_train2), X_train2.iloc[:,j])
            
    results = pd.DataFrame({"epoch": np.arange(nepochs)+1, 'train_accuracy':train_accuracy,'val_accuracy':val_accuracy,
                            "n":n,'test_size':test_size,'val_size':val_size,'seed':seed
                           }).set_index(['n','test_size','val_size','seed'])
    return w,X_test,t_test,results

def evaluate_baseline(t_test,t_train2,t_val):
    mode = t_train2.mode()
    frac_max_class = sum(t_train2.values == mode.values)/len(t_train2)
    accuracy_test = sum(t_test.values == mode.values)/len(t_test)
    accuracy_train2 = sum(t_train2.values == mode.values)/len(t_train2)
    accuracy_val = sum(t_val.values == mode.values)/len(t_val)
    return frac_max_class,accuracy_test,accuracy_train2,accuracy_val

def predict(w,X,threshold=0.5):
    # this predict function might be wrong; 
    y = X.apply(lambda row: 1 if activation(np.dot(w, row)) > threshold else 0, axis=1)
    return y

def confusion_matrix(t,y,labels):
    cm = pd.crosstab(t, y)#, rownames=[None], colnames=[None])
    
    return cm

def evaluation(cm,positive_class=1):
    
    # ASK PROFESSOR how to use positive_class?
    # positive class is simply used to decide what is positive
    # thus, can be changed
    # for gaussian distribution, take integral of fraction of SD before and after and get that probability instead of just getting value @ function
    TN = cm[1-positive_class][1-positive_class]
    TP = cm[positive_class][positive_class]
    FN = cm[positive_class][1-positive_class]
    FP = cm[1-positive_class][positive_class]
    
    sens = TP/(TP + FN)
    spec = TN/(TN+FP)
    prec = TP/(TP + FP)
    # print(TN,TP,FN,FP)
    stats = {'accuracy': (TP+TN)/(TN + TP + FN + FP),
            'sensitivity/recall': sens,
            'specificity': spec,
            'precision': prec,
            'F1':(2 * prec * sens)/(prec + sens)}
    return stats

def importance(X,t,seeds):
    importanceData = []
    # train the neural network
    w = {}
    X_test = {}
    t_test = {}
    for seed in seeds:
        w[seed],X_test[seed],t_test[seed],results1 = train(X,t,seed=seed)
        
    weights = np.zeros((len(X.columns),))
    for seed in seeds:
        curWeights = w[seed]
        absWeights = abs(curWeights)
        # print(absWeights, max(absWeights))
        newWeights = absWeights/max(absWeights)
        weights = weights + newWeights
    weights = weights/len(seeds)
    
    
    return pd.Series(weights, index=X.columns)
            
    # return pd.Series(importanceData, index=X.columns)

def getTestPermutationImportances(X, t, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in X.columns:
        importances[col] = 0
        
    # find the original accuracy
    w,X_test,t_test,results = train(X, t, seed=0)
    y_test = predict(w, X_test)
    cm = confusion_matrix(t_test, y_test, labels=[0,1])
    stats = evaluation(cm)
    orig_accuracy = stats['accuracy']
    
    # now carray out the feature importance work
    for col in X.columns:
        for perm in range(npermutations):
            Xtest2 = X_test.copy()
            Xtest2[col] = X_test[col].sample(frac=1, replace=False).values
            y_test = predict(w, Xtest2)
            cm = confusion_matrix(t_test, y_test, labels=[0,1])
            stats = evaluation(cm)
            newAccuracy = stats['accuracy']
            importances[col] += abs(orig_accuracy - newAccuracy)
        importances[col] = importances[col]/npermutations
    return importances    


"""
0 	0.593123 	0.581395 	0.593123 	0.613333
1 	0.538682 	0.623256 	0.538682 	0.680000
2 	0.573066 	0.623256 	0.573066 	0.600000
3 	0.581662 	0.604651 	0.581662 	0.606667
4 	0.598854 	0.576744 	0.598854 	0.606667
5 	0.558739 	0.604651 	0.558739 	0.660000
"""

"""
423    1
177    1
305    1
292    0
889    0
      ..
203    0
499    0
628    0
879    1
745    0
"""

"""
{'accuracy': 0.8046511627906977,
 'sensitivity/recall': 0.7666666666666667,
 'specificity': 0.832,
 'precision': 0.7666666666666667,
 'F1': 0.7666666666666667}
{'accuracy': 0.8046511627906977,
 'sensitivity/recall': 0.832,
 'specificity': 0.7666666666666667,
 'precision': 0.832,
 'F1': 0.832}
"""

"""
Pclass           0.213081
Age              1.000000
SibSp            0.367404
Parch            0.439871
Fare             0.023622
Sex_female       0.732359
Sex_male         0.712207
Embarked_C       0.069990
Embarked_Q       0.217206
Embarked_S       0.141142
CabinLetter_A    0.124921
CabinLetter_B    0.371960
CabinLetter_C    0.134666
CabinLetter_D    0.081472
CabinLetter_E    0.522909
CabinLetter_F    0.006352
CabinLetter_G    0.207904
CabinLetter_T    0.045736
dtype: float64
Pclass           0.811589
Age              0.136572
SibSp            0.416314
Parch            0.182515
Fare             0.027063
Sex_female       0.875793
Sex_male         1.000000
Embarked_C       0.193508
Embarked_Q       0.192596
Embarked_S       0.121983
CabinLetter_A    0.059189
CabinLetter_B    0.119585
CabinLetter_C    0.057409
CabinLetter_D    0.079080
CabinLetter_E    0.015075
CabinLetter_F    0.180748
CabinLetter_G    0.075277
CabinLetter_T    0.006886
dtype: float64
Pclass           0.851602
Age              0.479359
SibSp            0.315474
Parch            0.003320
Fare             0.136107
Sex_female       0.353828
Sex_male         0.459571
Embarked_C       0.010247
Embarked_Q       0.188633
Embarked_S       0.146189
CabinLetter_A    0.035832
CabinLetter_B    1.000000
CabinLetter_C    0.894399
CabinLetter_D    0.796835
CabinLetter_E    0.160814
CabinLetter_F    0.169318
CabinLetter_G    0.289735
CabinLetter_T    0.103127
dtype: float64
Pclass           0.439091
Age              0.363223
SibSp            0.434835
Parch            0.025707
Fare             1.000000
Sex_female       0.239137
Sex_male         0.235561
Embarked_C       0.026522
Embarked_Q       0.162617
Embarked_S       0.032344
CabinLetter_A    0.120816
CabinLetter_B    0.101603
CabinLetter_C    0.232047
CabinLetter_D    0.032499
CabinLetter_E    0.135755
CabinLetter_F    0.009449
CabinLetter_G    0.006223
CabinLetter_T    0.015989
dtype: float64
"""
