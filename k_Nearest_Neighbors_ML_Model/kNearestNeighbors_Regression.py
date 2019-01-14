# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 14:32:08 2018

@author: Nisha
"""

import numpy as np

def calculateDistances(trainingData, testData):
    # this function is used to calculate euclidean distance
    a = np.sum((trainingData - testData)**2 , axis=1)# axis = 1 means horizontally. since distance is calculated on basis of rows axis is taken as 1
    dists = np.sqrt(a) # square root of distance
    sort_array = np.argsort(dists)# sorting the distances array
    return dists, sort_array # returning distance and the sorted array

def distanceWeightedRegression(trainingData, testData, k):
    #this function calculates distance weighted regression
    improvedtestData = testData[:-1] # removing the last column from the 1-d testData array
    improvedtrainingData = trainingData[:,:-1] # removing the last column from trainingData 
    dists,sort = calculateDistances(improvedtrainingData, improvedtestData)
    # calculating distance
    
    neighbors = []  # to store nearest neighbors 
    
    for x in range(k):	
        neighbors.append(trainingData[sort[x]])  # calculating nearest neighbours using the distance sorted array as 
        #the one with the nearest distance is nearest neighbour
    
    neighbors = np.array(neighbors) # converting the neighbors into array format
    
    numofw = 0 #numerator of weight
    denofw = 0 #denominator of weight
    
    for i in range(k):
        numofw = numofw + (1/(dists[sort[i]] ** 2)) * (neighbors[i,-1])
    # calculating the weighted distance of k-nearest neighbors with respect to the given test
    #instance.    
        
    for j in range(k):
        denofw = denofw + (1/(dists[sort[j]]**2))
    # calculating the inverse of distance squared for the k - nearest neighbors for the given test instance.    
    reg = numofw/denofw # substituting the values in the formula.

    return reg

def Regression(trainingData, testData, k):
     #this function calculates distance weighted regression
    improvedtestData = testData[:-1]# removing the last column from the 1-d testData array
    improvedtrainingData = trainingData[:,:-1]# removing the last column from trainingData 
    dists,sort = calculateDistances(improvedtrainingData, improvedtestData)
    # calculating distance
    
    neighbors = [] # to store nearest neighbors 
    avg = 0 #variable to store mean is initialised to zero
    
    for x in range(k):	
        neighbors.append(trainingData[sort[x]])  # calculating nearest neighbours using the distance sorted array as 
        #the one with the nearest distance is nearest neighbour
    
    neighbors = np.array(neighbors) # converting the neighbors into array format
    
    avg = np.mean(neighbors[:,-1]) # taking out mean of the regression value for the nearest neighbours

    return avg #returning the mean of the regression value

def getAccuracy(testData, predictions):
     # function calculates the accuracy
    tot_sum_sqr = 0 # total sum of squares is assigned as zero
    sum_sqr_res = 0 # sum of squared residuals is assigned as zero
    Regr = 0 # R^2 which is depicted as Regr in this code is assigned as zero
    avg = np.mean(testData[:,-1]) # mean of the last column of test data is taken
    
    for i in range(len(testData)):
        tot_sum_sqr = tot_sum_sqr + ((testData[i,-1] - avg)**2) # substituting the values in total sum of squares formula
        sum_sqr_res = sum_sqr_res + ((predictions[i] - testData[i, -1])**2) # substituting the values in sum of residual squares formula
    
    Regr = (1 - (sum_sqr_res/tot_sum_sqr)) # calculating R^2 by dividing total sum of squares and sum of residual squares and subtracting the result by 1
    
    return Regr # returning the value of R^2
     
def main():
   
    trainingData = np.genfromtxt("path/training_data", delimiter=",")
      # loading trainingData
    testData = np.genfromtxt("path/test_data", delimiter=",")
    # loading testData
    trainingData = np.array(trainingData, dtype = float)# converting the trainingData into array
    testData = np.array(testData, dtype = float)# converting the trainingData into array
    
    predictions = []
   
    k= 3
    
    configuration = 1
    #1->Regression
    #2->distanceWeightedRegression
    
    for x in range(len(testData)):# loop for passing data to different functions
        
        if configuration == 1:
            res = Regression(trainingData, testData[x], k)# to calculate the predicted value of Regression
            print(res)
            predictions.append(res)
            print('> predicted=' + repr(res) + ', actual=' + repr(testData[x][-1]))
            
        if configuration == 2:
            res = distanceWeightedRegression(trainingData, testData[x], k)# to calculate the predicted value of distanceWeightedRegression
            print(res)
            predictions.append(res)
            print('> predicted=' + repr(res) + ', actual=' + repr(testData[x][-1]))
            
    accuracy = getAccuracy(testData, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')# printing accuracy based on the predicted values and actual values of testData

main()    
    