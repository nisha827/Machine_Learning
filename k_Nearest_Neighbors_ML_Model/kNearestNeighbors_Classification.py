# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 00:34:40 2018

@author: Nisha
"""
import numpy as np
import operator

def normalisation(data):
# this function is used to normalise the data in order to avoid incorrect classification    
    
    for i in range(5):
        mn = np.min(data[:,i], axis = 0) # calculating minimum of first 5 columns. Since minimum is being calculated on columns axis is taken as 0.
        mx = np.max(data[:,i], axis = 0) # calculating minimum of first 5 columns. axis =0 means vertically
        for j in range(len(data)):
            data[j,i] = ((data[j,i]-mn)/(mx-mn))# normalising formula
    
    return data   # returning the normalised dataset     

def calculateDistances(trainingData, testData):
# this function is used to calculate euclidean distance
    
    a = np.sum((trainingData - testData)**2 , axis=1)  # axis = 1 means horizontally. since distance is calculated on basis of rows axis is taken as 1
    dists = np.sqrt(a) # square root of distance
    sort_array = np.argsort(dists) # sorting the distances array
    return dists, sort_array # returning distance and the sorted array

def getNeighbors(trainingData, testData, k):
    improvedtestData = testData[:-1] # removing the last column from testData
    improvedtrainingData = trainingData[:,:-1]  # removing the last column from trainingData
    
    dists,sort = calculateDistances(improvedtrainingData, improvedtestData) # calling distances and the sorted array

    neighbors = []
    
    for x in range(k):	
        neighbors.append(trainingData[sort[x]]) # calculating nearest neighbours using the distance sorted array as 
        #the one with the nearest distance is nearest neighbour

    return neighbors

def getVotes(neighbors):
# function to calculate votes
    
    classVotes = {}
    
    for x in range(len(neighbors)):
        votes = neighbors[x][-1] # votes is calculated on the basis of last column
        
        if votes in classVotes:
            classVotes[votes] += 1
        else:
            classVotes[votes] = 1
    
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)# votes are sorted on the basis of number of votes
    #reverse sorting is also possible.operator.itemgetter is a callable function which fetches column 2 and sorts it in descending order.
    #reverse = true means sorting the column in descending order.
    
    return sortedVotes[0][0]

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
    testData = np.genfromtxt("path/test_data", delimiter=",")
    # loading testData
    trainingData = np.array(trainingData, dtype = float)# converting the trainingData into array
    trainingData1 = normalisation(trainingData)# converting all the values of trainingData to normalized values
    #the normalised values will only be passed to all functions for calculation
    
    testData = np.array(testData, dtype = float)# converting the testData into array
    testData1 = normalisation(testData)# converting all the values of testData to normalized values
    #the normalised values will only be passed to all functions for calculation
    
    k = 3
    
    predictions = []
   
    for x in range(len(testData1)):# loop for passing data to different functions
        n = getNeighbors(trainingData1, testData1[x], k)# to calculate neighbors a single row of test data with all rows of train ing data is sent everytime
        res = getVotes(n)# this variable stores votes
        predictions.append(res)# the votes calculated is treated as predicted values
        print('> predicted=' + repr(res) + ', actual=' + repr(testData1[x][-1]))# printing predicted and actual values
   
    accuracy = getAccuracy(testData1, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')# printing accuracy based on the predicted values and actual values of testData 

main()
