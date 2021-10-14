# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 03:07:19 2021

@author: migue
"""
import numpy as np
import pandas as pd
from sklearn import neighbors 
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neural_network import MLPClassifier
def LeerCSV(adressTrain):
    W_P=pd.read_csv(adressTrain)
    X_prime = W_P.iloc[:,[0,1,2]].values
    Y = W_P.iloc[:,3].values
    X_prime
    
    return (X_prime, Y)
def CrearModelos(X_train,X_test,y_train,y_test):    
    ##      KNN
    ModeloKNN=neighbors.KNeighborsClassifier()
    ModeloKNN.fit(X_train, y_train)
    ##      Decision tree
    ModelosTree = tree.DecisionTreeClassifier()
    ModelosTree.fit(X_train, y_train)
    tree.plot_tree(ModelosTree)
    ##      Logistic Regression
    ModeloLR = LogisticRegression()
    ModeloLR.fit(X_train, y_train)
    ##      Support Vector Machines
    ModeloSVM = svm.SVC()
    ModeloSVM.fit(X_train, y_train)
    ##      Artificial Neural Network
    
    modelRed = MLPClassifier(hidden_layer_sizes=(7,), max_iter=150,activation = 'relu',solver='adam',random_state=1)
    modelRed.fit(X_train, y_train)

    
    return (ModeloKNN, ModelosTree, ModeloLR, ModeloSVM,modelRed)
def CrearPredicciones(ModeloKNN, ModelosTree, ModeloLR, ModeloSVM,modelRed):    
    ##      KNN
    y_predKNN   = ModeloKNN.predict(X_test)
    ##      Decision tree
    y_predTree  = ModelosTree.predict(X_test)
    ##      Logistic Regression
    y_predLR    = ModeloLR.predict(X_test)
    ##      Support Vector Machines
    y_predSVM   =ModeloSVM.predict(X_test)
    
    y_predRed   =modelRed.predict(X_test)
    ##      KNN
    ##print("accuracy del modelo KNN\n",metrics.accuracy_score(y_test, y_predKNN))
    print("accuracy del modelo KNN\n",metrics.classification_report(y_test, y_predKNN))
    ##      Decision tree}
    ##print("accuracy del modelo Decision tree\n",metrics.accuracy_score(y_test, y_predTree))
    print("accuracy del modelo Decision tree\n",metrics.classification_report(y_test, y_predTree))

    ##      Logistic Regression
    ##print("accuracy del modelo Logistic Regression\n",metrics.accuracy_score(y_test, y_predLR))
    print("accuracy del modelo Logistic Regression\n",metrics.classification_report(y_test, y_predLR))

    ##      Support Vector Machines
    ##print("accuracy del modelo Support Vector Machines\n",metrics.accuracy_score(y_test, y_predSVM))
    print("accuracy del modelo Support Vector Machines\n",metrics.classification_report(y_test, y_predSVM))
    
    ##      Artificial Neural Network
    ##print("accuracy del modelo red\n",metrics.accuracy_score(y_test, y_predRed))
    print("accuracy del modelo Artificial Neural Network\n",metrics.classification_report(y_test, y_predRed))
    
    #y_pred   =modelRed.predict(X_train)
    #print(metrics.classification_report(y_train, y_pred))
    
    return (y_predKNN, y_predTree, y_predLR, y_predSVM,y_predRed )
def Normalizar(X_prime):
    scaler=StandardScaler()
    scaler.fit(X_prime)
    X=scaler.transform(X_prime)
    return (X,scaler)



(X_prime, Y) = LeerCSV('NEW2.csv')

(X,scaler)=Normalizar(X_prime)

########    split del set de Datos
X_train,X_test,y_train,y_test= train_test_split(X,Y,test_size=.20,random_state=45)

########    Crear Modelos
(ModeloKNN, ModelosTree, ModeloLR, ModeloSVM,modelRed)   = CrearModelos(X_train,X_test,y_train,y_test)

CrearPredicciones(ModeloKNN, ModelosTree, ModeloLR, ModeloSVM, modelRed)









