# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:01:43 2018

@author: Nisha
"""

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SVMSMOTE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import ADASYN
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings("ignore")


rs = RandomState(123)

def dataset_loading():#for reading and loading the dataset
    df = pd.read_csv("path/of/dataset", na_values='?', delimiter=',')#path of the dataset
    df.columns = ["Age", "WorkClass", "fnlwgt", "Education", "EducationNum","MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
    "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"]# columns of the dataset

    return df

def clean_and_categorise_data(df):#preprocess the data
    
    
    imp_freq = SimpleImputer(missing_values=np.nan, strategy='most_frequent')#impute missing values using most_frequent strategy
    
    df[["Occupation"]] = imp_freq.fit_transform(df[["Occupation"]])#impute occupation column
    df[["WorkClass"]] = imp_freq.fit_transform(df[["WorkClass"]])#impute workclass column
    df[["NativeCountry"]] = imp_freq.fit_transform(df[["NativeCountry"]])#impute nativecountry column
    
    encoder = OneHotEncoder(sparse=False)#using onehotencoder to turn the below categorical columns to quantitative columns
  
    df = encoder.fit_transform(df[["WorkClass", "Education", "MaritalStatus", "Occupation", "Relationship", "Race", "Gender", "NativeCountry"]])
    
    scale = StandardScaler()# to scale the whole data
    
    data = scale.fit_transform(df)
    
    df = pd.DataFrame(data)#changed the scaled to dataframe again
    
    return df

def imbalance(df, y, i):# different smote techniques to tackle imbalance
    
    if i == 1:
        
        sm = SMOTE(random_state=123)#using smote
        
    if i == 2:
        sm = ADASYN()#using ADASYN
    
    if i==3:
        sm = BorderlineSMOTE()#using borderline Smote
        
    if i==4:
        sm = SVMSMOTE() #using SVMSmote
        
    X_train, y_train= sm.fit_sample(df, y)#fitting different techniques used above
    return X_train,y_train#return x_train and y_train

def feature_selection(df,y):#using feature selection
    
    a = []#list to store feature ranking
    estimator = LogisticRegression(random_state = 10)#Estimator is logistic regression
    selector = RFECV(estimator, cv=10,step=1, scoring='accuracy')#using greedy search RFECV technique
    selector.fit(df, y)
    for i in range(len(selector.ranking_)):#selecting only rank 1 features and discarding all others for every model
        if (selector.ranking_[i]) != 1:
            a.append(i)
    df.drop(a,axis = 1, inplace = True)
    return df
    
def splitting_data():#splitting the data into train and test
    
    df = dataset_loading()
    df["Income"] = df["Income"].map({ "<=50K": -1, ">50K": 1 })
    y = df["Income"].values#getting income in y
    df.drop("Income", axis=1, inplace=True)
 
    return df, y #returning whole dataset and income values

def tested_models(X_train, X_test, y_train, y_test):#different tested models
    
    models = []#list to store models
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('RandomForest', RandomForestClassifier(random_state = 32)))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC(random_state = 42)))
    models.append(('L_SVM', LinearSVC()))
    models.append(('SGDC', SGDClassifier()))

    accuracyScore = []#list to store accuracy score
    F1Score = []#list to store f1 score
    names = []#list to store names
    
    for name, model in models:#loop to predict accuracy and f1score for all models
          model.fit(X_train, y_train)
          predictions = model.predict(X_test)
          accuracy = accuracy_score(y_test, predictions)
          f1 = f1_score(y_test, predictions)
          F1Score.append(f1)
          accuracyScore.append(accuracy)
          names.append(name)
    print("Accuracy is :",accuracyScore,"F1 Score is ",F1Score,"Name of the model is :",names)

def main():
    df, y = splitting_data()#dataset and income values are returned from this function
    df = clean_and_categorise_data(df)#dataset is sent for preprocessing
    print("done")
    df = feature_selection(df,y)#sending whole dataset to feature selection function
    print("done")
    
    for i in range(1,5):#loop to run different imbalance techniques for all the models
        
        train_f, train_y = imbalance(df, y, i)
    
        X_train, X_test, y_train, y_test = train_test_split(
        train_f, train_y, test_size=0.25, random_state=20)#splitting data
    
        tested_models(X_train, X_test, y_train, y_test)#sending the splitted data to tested_model function
        
    print('Optimization Started')
    #hyperparameter optimization Features
    param_grid= {'bootstrap': [True, False],
     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'criterion':['gini','entropy'],
     'random_state': [122],
     'n_estimators': [100,150,200,250,300,350]}
    clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=10)#hyperparameter optimization  on randomforest classifier using gridsearchcv and 10 crossfold
    clf.fit(train_f, train_y)
    print("\n Best parameters set found on development set:")
    print(clf.best_params_ , "with a score of ", clf.best_score_)
    scores = model_selection.cross_val_score(clf.best_estimator_,train_f, train_y, cv=10)#scores for model
    print (scores.mean()) #printing mean
    
    
main()