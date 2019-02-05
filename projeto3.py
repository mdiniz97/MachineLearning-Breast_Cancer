import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn import svm

dataset = load_breast_cancer()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

#dataset.keys()

#dataset['data'] # daset das features
#dataset['target'] # diagnósticos
#dataset['target_names'] # nomes dos diagnósticos
#dataset['DESCR'] # descrição do dataset 
#dataset['feature_names'] # nome das features


dfData = pd.DataFrame(dataset['data'], columns = dataset['feature_names'])
dfTarget = pd.DataFrame(dataset['target'], columns = ['diagnosis'])

dfData.head()

dfTarget.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dfData, np.ravel(dfTarget), test_size=0.30, random_state = 1)
# colocando o random state para sempre que reexecutar gerar a mesma combinação aleatória de treino e teste

import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

svm = LinearSVC()
#svm.kernel = 'linear'

svm.fit(X_train,y_train)

pred  = svm.predict(X_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

from sklearn.model_selection import GridSearchCV

paramGrid = {'C': [0.01, 0.1,1, 10, 100, 1000, 10000, 100000]}

grid = GridSearchCV(svm, paramGrid, verbose = 3, n_jobs = -1)

grid.fit(X_train, y_train)

print("Best params: ", grid.best_params_, end = "\n\n")
print("Best estimator: ", grid.best_estimator_, end = "\n\n")
print("Best score: ", grid.best_score_, end = "\n\n")

gridPred = grid.predict(X_test)
print(classification_report(y_test, gridPred))
print(confusion_matrix(y_test, gridPred))

listaScore = []
listaC = []
for score, C in zip(grid.cv_results_['mean_test_score'], grid.cv_results_['param_C']):
    listaScore.append(score)
    listaC.append(C)
    
plt.subplot()
plt.loglog(listaC, listaScore, basex=2)
plt.grid(True)
plt.yscale("linear")
plt.xlabel("C")
plt.ylabel("Test Score")
plt.title("Test Score x C")
plt.show()

from pandas.tools.plotting import scatter_matrix

result = pd.concat([dfData, dfTarget], axis=1, join_axes=[dfData.index])

result.head()




svmG = SVC()

svmG.fit(X_train,y_train)

predGaussian  = svmG.predict(X_test)
print(classification_report(y_test, predGaussian))
print(confusion_matrix(y_test, predGaussian))

from sklearn.model_selection import GridSearchCV

# sigma = 1/gamma

paramGridGaussian = {'C': [0.01, 0.1,1, 10, 100, 1000, 10000, 100000],
                    'gamma': [1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001]}

gridGaussian = GridSearchCV(svmG, paramGridGaussian, verbose = 3, n_jobs = -1)

gridGaussian.fit(X_train, y_train)

print("Best params for gaussian kernel: ", gridGaussian.best_params_, end = "\n\n")
print("Best estimator for gaussian kernel: ", gridGaussian.best_estimator_, end = "\n\n")
print("Best score for gaussian kernel: ", gridGaussian.best_score_, end = "\n\n")

gridPredGaussian = gridGaussian.predict(X_test)
print(classification_report(y_test, gridPredGaussian))
print(confusion_matrix(y_test, gridPredGaussian))

listaScoreG = []
listaCG = []
listaGammaG = []
dicio = {}
for score, C, gamma in zip(gridGaussian.cv_results_['mean_test_score'], gridGaussian.cv_results_['param_C'], gridGaussian.cv_results_['param_gamma']):
    if (score, gamma) not in dicio:
        listaScoreG.append(score)
        listaCG.append(C)
        listaGammaG.append(gamma)
        dicio[score, gamma] = 1
        #print(score, C, gamma)
        
plt.subplot()
plt.loglog(listaCG, listaScoreG, basex=2)
plt.grid(True)
plt.yscale("linear")
plt.xlabel("C")
plt.ylabel("Test Score")
plt.title("Test Score x C")
plt.show()



svmRBF = SVC(kernel = 'rbf')

svmRBF.fit(X_train,y_train)

predRBF  = svmRBF.predict(X_test)
print(classification_report(y_test, predRBF))
print(confusion_matrix(y_test, predRBF))

