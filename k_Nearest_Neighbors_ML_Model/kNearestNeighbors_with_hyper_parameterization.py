# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 11:12:51 2018

@author: Nisha
"""
import numpy as np

def normalisation(data):
# this function is used to normalise the data in order to avoid incorrect classification    
    
    for i in range(data.shape[1]-1):
        mn = np.min(data[:,i], axis = 0) # calculating minimum of first 5 columns. Since minimum is being calculated on columns axis is taken as 0.
        mx = np.max(data[:,i], axis = 0) # calculating minimum of first 5 columns. axis =0 means vertically
        for j in range(len(data)):
            data[j,i] = (data[j,i]-mn)/(mx-mn)# normalising formula
    
    return data   # returning the normalised dataset 

def calculateEuclideanDistance(trainingData, testData, n):
# this function is used to calculate euclidean distance    
    a = np.sum((trainingData - testData)**2 , axis=1) # axis = 1 means horizontally. since distance is calculated on basis of rows axis is taken as 1
    dists = np.sqrt(a) # square root of distance
    sort_array = np.argsort(dists) # sorting the distances array
    return dists, sort_array # returning distance and the sorted array

def calculateManhattanDistance(trainingData, testData, n):
# this function is used to calculate manhattan distance    
    a = abs(trainingData - testData)# abs is used so as to take only the positive part of the value
    dists = np.sum(a, axis=1)# sum of distances on the basis of rows
    sort_array = np.argsort(dists)# sorting the distances array
    
    return dists, sort_array  # returning distance and the sorted array

def calculateMinkowskiDistance(trainingData, testData, n):
# this function is used to calculate minkowski distance     
     a = np.sum(abs(trainingData - testData)**n , axis=1)# here n is a variable which can be initialised anything other than 1 or 2
     # if n is initialised as 1 the distance becomes manhattan distance and if the value of n = 2 it becomes euclidean distance
     dists = ((a) * (1/n))
     sort_array = np.argsort(dists)# sorting the distances array
     
     return dists, sort_array # returning distance and the sorted array
    

def weights(trainingData, testData, k, n):
 # this function calculates votes (that is the predicted value) depending upon the nearest neighbours and whether the cancer is benign(0.0) or malignant(1.0)   
 # benign values are treated as class0 and malignant values are treated as class1 
    class0 = [] # to store all benign values
    class1 = [] # to store all malignant values
    
    for i in range(len(trainingData)):
        if trainingData[i,-1] == 0.0: # if the ith row in the last column has value 0.0 that is benign append it to class 0
            class0.append(trainingData[i])
        else:   # else if the value is 1.0 that is malignant append it to class1
            class1.append(trainingData[i])
    
    class0 = np.array(class0, dtype=float) # class0 is converted to array
    class1 = np.array(class1, dtype=float) # class0 is converted to array       
    
    improvedtestData = testData[:-1] # removing the last column from the 1-d testData array
    improvedtrainingData0 = class0[:, :-1]  # removing the last column from class0
    improvedtrainingData1 = class1[:, :-1] # removing the last column from class0

    neighbors0 = [] # to store nearest neighbors of class0
    neighbors1 = [] # to store nearest neighbors of class1
    vote0 = 0 # calculate votes for class0
    vote1 = 0  # calculate votes for class0
    
    dists0,sort0 = calculateEuclideanDistance(improvedtrainingData0, improvedtestData,n)
    # calculating euclidean distance above on class0 
    # to calculate manhattan distance replace the function name with calculateManhattanDistance
    # to calculate minkowski distance replace the function name with calculateMinkowskiDistance
    
    dists1,sort1 = calculateEuclideanDistance(improvedtrainingData1, improvedtestData,n)
    # calculating euclidean distance above on class1 
    # to calculate manhattan distance replace the function name with calculateManhattanDistance
    # to calculate minkowski distance replace the function name with calculateMinkowskiDistance

    for x in range(k):	
        neighbors0.append(class0[sort0[x]])# calculating nearest neighbours of class0 using the distance sorted array as 
        #the one with the nearest distance is nearest neighbour
        vote0 += (1/(dists0[sort0[x]])) # here 1/dist denotes weight of neighbour so the more the weight the nearest the neighbor
    
    for y in range(k):
        neighbors1.append(class1[sort1[y]])# calculating nearest neighbours of class1 using the distance sorted array as 
        #the one with the nearest distance is nearest neighbour

        vote1 += (1/(dists1[sort1[y]])) # here 1/dist denotes weight of neighbour 
        
    if vote0 > vote1: # comparing votes of class0 and class1 and returning the maximum of the two as the predicted value
        return 0.0
    else:
        return 1.0
    
def getAccuracy(testData, predictions):
    # function calculates the accuracy     
    correct = 0
    
    for x in range(len(testData)):
        if testData[x,-1] == predictions[x]: # the last column values in testData are compared to the sorted votes which are taken as predicted values
            correct += 1 # if both the predicted value and actual value match it is treated as accurate
	# all the values which are accurate i.e actual matches the predicted are stored in the correct variable which is then printed to show the overall accuracy of the system
    
    return (correct/float(len(testData))) * 100.0
    
def main():
   
    trainingData = np.genfromtxt("path/training_data", delimiter=",")
    # loading trainingData
    trainingData = np.array(trainingData, dtype = float)# converting the trainingData into array
    trainingData1 = normalisation(trainingData)# converting all the values of trainingData to normalized values
    #the normalised values will only be passed to all functions for calculation
    
    testData = np.genfromtxt("path/test_data", delimiter=",")
    # loading testData
    testData = np.array(testData, dtype = float)# converting the testData into array
    testData1 = normalisation(testData)# converting all the values of testData to normalized values
    #the normalised values will only be passed to all functions for calculation
    
    
    predictions = []
   
    k = 18
    n = 3
    for x in range(len(testData1)):# loop for passing data to different functions
        res = weights(trainingData1, testData1[x], k, n)# to calculate the predicted value and here n is the variable taken for minkowski distance
        predictions.append(res)
        print('> predicted=' + repr(res) + ', actual=' + repr(testData1[x][-1]))# printing predicted and actual values
   
    accuracy = getAccuracy(testData1, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')# printing accuracy based on the predicted values and actual values of testData
    
main()