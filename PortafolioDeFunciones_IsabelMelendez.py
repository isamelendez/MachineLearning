'''
SUGGESTED TO PLOT OUTPUTS


output = GradientBoostMaxDepth(1, 33, 1, X_train, y_train, X_test, y_test)
output = np.array(output)
transposed = output.T
x, y = transposed
plt.plot(x, y)
plt.title('')
plt.xlabel('')
plt.ylabel('R^2')
plt.show()


'''


#Random quantitity numbers between range
#import random
def getRandNumbers(q, min, max):
  numbers = []
  for i in range(q):
    number = random.randint(min, max)
    numbers.append(number)
  return numbers

#usage
#get 10 random numbers between 10 and 20
#numbers = getRandNumbers(10, 10, 20)


#Random quantitity of numbers between 0 and 1
#import random
def getRandDecimals(q):
  numbers = []
  for i in range(q):
    number = random.random()
    numbers.append(number)
  return numbers

#usage
#numbers = getRandDecimals(100)


#KFolds K tunning
#import numpy as np
#from math import sqrt
#from sklearn.model_selection import KFold
#from sklearn.linear_model import LinearRegression
def KTestLinearR(min, max, step, X, y):
    RMSE_i = []
    results = []
    for i in np.arange(min, max, step):
        lm = LinearRegression()
        kfold = KFold(i, True, 1)
        splitdata = kfold.split(X, y)
        for train, test in splitdata:
            model = lm.fit(X[train], y[train])
            y_predicted = lm.predict(y[test])
            rms = (sum((y_predicted - y[test]) ** 2) / len(y_predicted)) ** (1 / 2)
            RMSE_i.append(rms)
        RMSE = np.mean(RMSE_i)
        values = (i, RMSE)
        results.append(values)

    return results
#usage
# output = KTestLinearR(2, 16, 1, X, y)



#Ridge Regression Alpha Tunning
# funcion para calcular con disintos alphas en un determinado rango

def rmseFunc(min, max, step, X_train, y_train, X_test, y_test):
    res = []
    for i in np.arange(min, max, step):
        ridgereg = Ridge(alpha=i)

        ridgeModel = ridgereg.fit(X_train, y_train)
        y_pred = ridgereg.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        total = (i, rmse)
        res.append(total)
    return res

# test = rmseFunc(0.01, 1, 0.01, X_train, y_train, X_test, y_test)
# output = np.array(test)



#Decision Tree Regressor hyperparameter functions
#from sklearn.tree import DecisionTreeRegressor
def decisionTreeRegMinLeaf(min, max, step, X_train, y_train, X_test, y_test):
    results = []
    for i in np.arange(min, max, step):
        regr = DecisionTreeRegressor(min_samples_leaf=i)
        regr.fit(X_train, y_train)
        y_pred = regr_1.predict(X_test)
        score = regr.score(X_test, y_test)
        values = (i, score)
        results.append(values)
    return results

# output = decisionTreeRegMinLeaf(2, 1000, 10, X_train, y_train, X_test, y_test)

def decisionTreeRegDepth(min, max, step, X_train, y_train, X_test, y_test):
    results = []
    for i in np.arange(min, max, step):
        regr = DecisionTreeRegressor(max_depth=i)
        regr.fit(X_train, y_train)
        y_pred = regr_1.predict(X_test)
        score = regr.score(X_test, y_test)
        values = (i, score)
        results.append(values)
    return results

# decisionTreeRegDepth(2, 1000, 10, X_train, y_train, X_test, y_test)




#Random Forest  Regressor hyperparameter tunning
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.metrics import r2_score

def RandomForestRegMinSamplesLeaf(min, max, step, X_train, y_train, X_test, y_test):
    results = []
    for i in np.arange(min, max, step):
        clf = RandomForestRegressor(n_estimators=100,
                                    oob_score=True,
                                    min_samples_leaf=i,
                                    n_jobs=-1,
                                    bootstrap=True)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        r2_score(y_test, pred)
        values = (i, r2_score)
        results.append(values)
    return results
#output = RandomForestRegMinSamplesLeaf(2, 1000, 2, X_train, y_train, X_test, y_test)

def RandomForestRegMaxDepth(min, max, step, X_train, y_train, X_test, y_test):
    results = []
    for i in np.arange(min, max, step):
        clf = ExtraTreesRegressor(n_estimators=100,
                                    max_depth = i,
                                    n_jobs=-1)
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        r2s = r2_score(y_test, pred)
        values = (i, r2s)
        results.append(values)
    return results
#output = RandomForestRegMaxDepth(2, 500, 2, X_train, y_train, X_test, y_test)




#XGBOost hyperparameters tunning
#from sklearn.metrics import r2_score
#import xgboost as xgb

def XGBoostMaxDepth(min, max, step, X_train, y_train, X_test, y_test):
    results = []
    for i in np.arange(min, max, step):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        param = {'booster': 'gbtree', 'max_depth': i, 'eta': 0.8, 'gamma': 2}
        num_round = 100
        bst = xgb.train(param, dtrain, num_round)
        y_train_pred = bst.predict(dtrain)
        y_test_pred = bst.predict(dtest)
        r2s = r2_score(y_test, y_test_pred)
        values = (i, r2s)
        results.append(values)
    return results
#output = XGBoostMaxDepth(2, 500, 2, X_train, y_train, X_test, y_test)



#Gradient Boosting hyperparameters tunning
#model = GradientBoostingClassifier()
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score

def GradientBoostLearning(min, max, step, x_train, y_train, x_test, y_test):
    results = []
    for i in np.arange(min, max, step):
        model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
        learning_rate=i, loss='deviance', max_depth=3,
        max_features=None, max_leaf_nodes=None,
        min_samples_leaf=1,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        n_estimators=100, presort='auto', random_state=None,
        subsample=1.0, verbose=0, warm_start=False)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        r2s = r2_score(y_test, y_pred)
        values = (i, r2s)
        results.append(values)
    return results
#output = GradientBoostLearning(0.05, 1.05, 0.05, X_train, y_train, X_test, y_test)

def GradientBoostNTrees(min, max, step, x_train, y_train, x_test, y_test):
    results = []
    for i in np.arange(min, max, step):
        model  = GradientBoostingClassifier(criterion='friedman_mse', init=None,
        learning_rate=0.8, loss='deviance', max_depth=3,
        max_features=None, max_leaf_nodes=None,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        n_estimators=i, presort='auto', random_state=None,
        subsample=1.0, verbose=0, warm_start=False)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2s = r2_score(y_test, y_pred)
        values = (i, r2s)
        results.append(values)
    return results

def GradientBoostMaxDepth(min, max, step, x_train, y_train, x_test, y_test):
    results = []
    for i in np.arange(min, max, step):

        model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
        learning_rate=0.8, loss='deviance', max_depth=i,
        max_features=None, max_leaf_nodes=None, min_samples_leaf=7,
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        n_estimators=100, presort='auto', random_state=None,
        subsample=1.0, verbose=0, warm_start=False)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        r2s = r2_score(y_test, y_pred)
        values = (i, r2s)
        results.append(values)
    return results
#output = GradientBoostMaxDepth(1, 33, 1, X_train, y_train, X_test, y_test)




#K Means K tunning
#from sklearn.cluster import KMeans
#import numpy as np

def KMeansKTunning(min, max, step, data):
    results = []
    for i in np.arange(min, max, step):
        clusters = KMeans(n_clusters=i).fit(data)
        sse = clusters.inertia_
        values = (i, sse)
        results.append(values)
    return results
#output = KMeansKTunning(2, 16, 1, data)