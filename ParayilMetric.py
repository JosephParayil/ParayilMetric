#   Joseph Parayil
#   5/3/2024
#   This is my analysis code for obtaining the Parayil Metric of a given algorithm 
#   on a given environmnet. 

#   This must be run as an interactive Jupyter Notebook in Google Colab

#   Please note that this code alone at its current state will not work. This is
#   because I have omitted the code segment responsible for training instances
#   of a QL or DQN agent on a given environment. This omitted segment was not my
#   own code. It was used directly from an online QL/DQN tutorial series on 
#   the Gym toy text environments, by @johnnycode on youtube. His github account
#   is johnnycode8. His repository can be found here:
#   https://github.com/johnnycode8/gym_solutions/tree/main

#   If you wish to run this code for yourself, you must substitute the omitted
#   part at the beginning with code for training a specific RL model on a specific
#   environment, preferably with a function called run(timesteps).

#   That being said, you will find that this analysis code that I have created
#   is highly extensive. The size of this analysis code actually far exceeds the
#   length of the code for the actual RL training code. It demonstrates an 
#   intelligently elegant method to obtain a Parayil Metric.



import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt # Not used
import pickle

import math
import time
from datetime import datetime

import json
from google.colab import drive
drive.mount('/content/drive')

#Based on google drive; change based on your own file path
resultFilePath = '/please/insert/your/file/path/results.txt'
dataFilePath = '/please/insert/your/file/path/testHistory.json'


#   IMPORTANT NOTE: Trying to run this code will not work. The code for the 
#   training of QL/DQN on one of the environments is omitted and must be included.


"""
OMITTED:
Insert relevant code for training QL or DQN on a given environment here
"""





#===============================================================#===============================================================#===============================================================#===============================================================

#           ANALYSIS CODE BEINGS HERE

#===============================================================#===============================================================#===============================================================#===============================================================



#===============================================================
#   CONSTANTS
#===============================================================

convergeTimestep = 2000         # At what timestep does our model reach convergence? (aka: what is our upper bound for our binary searches)
convergePerformance = 100       # What is the maximum performance (percent) that our model converges on
timestepTestTrials = 100        # How many trials for a single performance test at a given timestep?
performanceCheckTrials = 100    # In each timestep test trial, at the end of training, how many trials should we test the agent's win rate performance?
yIntervals = 5                  # What yIntervals for the Parayil Metric?
timeTestTrials = 100            # How many trials in the process of converting timestep values to actual computational time?

#Histories (to record the reuslts of tests conducted, so that we can refer to their result again without doing the same test again)
testHistory = []
timeTestHistory = []



#===============================================================#===============================================================#===============================================================#===============================================================
#===============================================================#===============================================================#===============================================================#===============================================================

# runTimestepTest(timesteps): Function for running a timestep test
# A Timestep test obtains the average performance after a specific amount of timesteps
# I run 100 (timestepTestTrials) repetitions of the training process for one timestep test
# For each of these trials, 100 (performanceCheckTrials) repetitions of evaluating the performance at the end of training is conducted
def runTimestepTest(timesteps):
  print("      Running test for timestep", timesteps, "   Time: ", datetime.now())
  performances = [0] * timestepTestTrials
  for i in range(timestepTestTrials):
    run(timesteps) 
    performances[i] = run(performanceCheckTrials,False)

  return sum(performances)/len(performances) #Return average performance


# getPerformance(c): Function to choose between running a timestep test or
# getting the existing test result if the test was conducted before and stored
# in testHistory
def getPerformance(c):
  global testHistory

  #Finding if there was past timestep test already done
  index = next((index for index, (x, y) in enumerate(testHistory) if x == c), None)

  if index is None:
    #Running new timestep test
    result = runTimestepTest(c)
    testHistory.append((c,result))

    # Writing to data file in real-time
    with open(dataFilePath, 'w') as file:
      json.dump(
          {
              "testHistory": [[x, y] for x, y in testHistory],
              "timeTestHistory": timeTestHistory
          }, file
    )


    return result
  else:
    #Getting result of already completed timestep test
    #print("Test already done")
    result = testHistory[index][1]
    return result


# binarySearch(targetValue, a, b): Function to recursively find at what timestep
# does the model reach a certain average performance (targetValue). The variables
# a and b represent the lower and upper bounds of the search, respectively.

# Even though binarySearch is used in the context of finding items in a sorted
# list, it can be just as well applied in finding the exact timestep at which a
# particular performance value is reached on average on a learning graph. A
# learning graph, or more specifically, an aggregated learning graph from many trials of
# training, can never go down. In this way, the graph can be said to be already
# "sorted". on average will  Instead of indexing a list, we use a "function",
# which is this theoretical learning graph. Of course, we do not have this
# learning graph in it's entirety at our fingertips, so we have to obtain it
# using time-intensive performance tests using the getPerformance and
# runTimestepTest functions

def binarySearch(targetValue, a, b):
  c = math.floor((b+a)/2)
  result =  getPerformance(c)

  print("      ",a,b,c,"Result: ", result, "   Time: ", datetime.now())

  if (b-a) == 1: #We narrowed down our search to return the final value

    #Two options remaining
    aPair = (a,getPerformance(a))
    bPair = (b,getPerformance(b))


    print(aPair,bPair,"targetValue:",targetValue,"   Time: ",datetime.now())
    #Which one is closer to the target value
    if (abs(targetValue - bPair[1]) < abs(targetValue - aPair[1])):
      thePair = bPair
    elif (abs(targetValue - aPair[1]) < abs(targetValue - bPair[1])):
      thePair = aPair
    elif ((targetValue - aPair[1]) > (targetValue - bPair[1])):
      thePair = bPair
    else: #If both are equally close to the target value
      if (targetValue-bPair[1]>0): #If the difference is greater
        thePair = bPair
      else:
        thePair = aPair

    print("FINAL RESULT: ", thePair,"   Time: ",datetime.now())
    return thePair[0]


  #Recursive narrowing of binary search algorithm
  if targetValue>result:
    return binarySearch(targetValue, c, b)
  else:
    return binarySearch(targetValue, a, c)


#Function to convert list of tuples to string form for printing
def tuples_to_indented_string(tuples_list, indent=4):
    # Check if tuples_list is iterable
    if not isinstance(tuples_list, (list, tuple)):
        raise TypeError("Input must be a list or tuple")

    indented_string = ""
    for item in tuples_list:
        # Check if each item is a tuple
        if not isinstance(item, tuple):
            raise TypeError("Elements of the input list must be tuples")

        # Unpack the tuple and format the string
        x, y = item
        indented_string += " " * indent + f"({x}, {y})\n"
    return indented_string


#===============================================================#===============================================================
#   ACTUAL MAIN CODE BEINGS HERE                    findParayilMetric function
#===============================================================#===============================================================

# findParayilMetric(yIntervals): Function to find the Parayil Metric given the
# specified yInterval value.

def findParayilMetric(yIntervals):
  global testHistory
  global timeTestHistory
  #Importing testHistory.json if it exists
  try:
    with open(dataFilePath, 'r') as file:
      data = json.load(file)

      testHistory = [tuple(item) for item in data['testHistory']]
      timeTestHistory = data['timeTestHistory']
      print("file loaded successfully:")
      print("testHistory:",testHistory)
      print("timeTestHistory:", timeTestHistory)
  except FileNotFoundError:
    #file doesn't exist yet
    testHistory = []
    timeTestHistory = []
    print("file does not exist yet")

  startTime = datetime.now()

  points = []

  #Running searches for each y interval
  for i in np.arange(yIntervals, convergePerformance+1, yIntervals):
    points.append((binarySearch(i,0,convergeTimestep),i))

  for i in np.arange(i+yIntervals,101,yIntervals):
    points.append((float('inf'),i))


  #Computing and printing the simple (incomplete) Parayil Metric
  print()
  print()
  print("-----------------------------------------------")
  print("RESULT:")
  print(*points, sep='\n')

  slopes=[]


  for i in range(len(points)):
    slopes.append(points[i][1]/points[i][0])

  print()
  print("-----------------------------------------------")
  print(*slopes, sep='\n')

  # IMPORTANT NOTE: This is not actual real Parayil Metric. One more step is
  # required. We must transform the X-axis from being timesteps to representing
  # actual time. The reason I transform from timesteps to actual computational
  # time is to account for the vast difference in computational time between
  # QL and DQN implementations in a single timestep.
  parayilMetric = sum(slopes)/len(slopes)

  #NOT FINAL
  print("")
  print("-----------------------------------------------")
  print("")
  print("PARAYIL METRIC:")
  print(parayilMetric)
  print("   Time: ",datetime.now())
  print("-----------------------------------------------")






  #==============================================================================================
  #                  Converting the X-axis from Iterations to Time taken to run
  #===============================================================================================

  # The reason we do this is to account for the vast difference in computational
  # time between QL and DQN implementations in a single timestep.

  # IMPORTANT NOTE: This is a step that can introduce some significant variability
  # and flaws to the overall experiment. Of course, computational time varies between
  # various computing systems and also due to a countless number of other factors.
  # I mitigated this problem as much as I could by ensuring that the CPU specs
  # used by Google colab were similar and running a 100 trials
  # A fundamental revision of this step is necessary when I decide to scale this experiment




  print("-----------------------------------------------")
  print("Converting the X-axis from Iterations to Time taken to run")
  print("-----------------------------------------------")



  pointsTime = points[:]

  for i in range(len(pointsTime)):
    iterations = pointsTime[i][0]

    #Finding what computational time it takes to train the model for the given amount of iterations

    averageTime = 0

    #If x is infinity
    if iterations == float('inf'):
      print("Timestep of", iterations , "takes", iterations, "on average to run")
      continue

    if i < len(timeTestHistory): #History present here
          averageTime = timeTestHistory[i]
    else: #NO history here yet
      print("Running time tests for timestep: " , iterations)
      #Running time test
      times = []
      for j in range(timeTestTrials):

          elapsedTime = 0

          startTime = datetime.now()
          run(iterations) """ <--- <--- <--- IMPORTANT NOTE: run Function is from the omitted johnnycode code snippit """
          elapsedTime = (datetime.now() - startTime).total_seconds()

          times.append(elapsedTime)
          print(elapsedTime)


      averageTime = sum(times)/len(times)

      timeTestHistory.append(averageTime)
      print("Writing to timeTestHistory in file")
      # Writing to data file
      with open(dataFilePath, 'w') as file:
        json.dump(
            {
                "testHistory": [[x, y] for x, y in testHistory],
                "timeTestHistory": timeTestHistory
            }, file
      )

    print("Timestep of", iterations , "takes", averageTime, "on average to run")

    #Applying changes on x values
    pointsTime[i] = (averageTime, pointsTime[i][1])



  #=======================
  #Making new slope values
  #=======================

  slopesTime=[]

  for i in range(len(pointsTime)):
    slopesTime.append(pointsTime[i][1]/pointsTime[i][0])


  #=================================
  #Calculating actual Parayil Metric
  #=================================
  parayilMetricTime = sum(slopesTime)/len(slopesTime)

  #Printing everything
  print(*pointsTime, sep='\n')
  print(*slopesTime, sep='\n')

  print("Final Parayil Metric with X-Axis mode: Time:")
  print(parayilMetricTime)


  #=================================================================
  #           Outputting Results to Google Drive
  #=================================================================

  elapsedTime = datetime.now() - startTime

  #To prevent overclocking of writing to google drive
  if (elapsedTime.total_seconds
   () < 0.5):
    time.sleep(0.5)

  with open(resultFilePath, 'a') as file:
    # Convert slopes and slopesTime to strings
    slopes_str = ', '.join(map(str, slopes))
    slopesTime_str = ', '.join(map(str, slopesTime))

    # Write to the file
    file.write("\n-----------------------------------------------")
    file.write("\nQL 4x4 TEST RESULTS")
    file.write("\n")
    file.write("\nTRIALS PER TEST:" + str(timestepTestTrials))
    file.write("\nY-INTERVALS: " + str(yIntervals))
    file.write("\n")
    file.write("\nTIME ELAPSED: " + str(elapsedTime))
    file.write("\nCURRENT TIME: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    file.write("\n")
    file.write("\n-----------------------")
    file.write("\nRESULTS WITH X-AXIS MODE: ITERATIONS")
    file.write("\n")
    file.write("\nPOINTS: ")
    file.write("\n" + tuples_to_indented_string(points))
    file.write("\n")
    file.write("\nSLOPES: ")
    file.write("\n" + slopes_str)
    file.write("\n")
    file.write("\nPARAYIL METRIC: ")
    file.write("\n" + str(parayilMetric))
    file.write("\n")
    file.write("\n-----------------------")
    file.write("\nRESULTS WITH X-AXIS MODE: TIME")
    file.write("\nTrials for each time test: " + str(timeTestTrials))
    file.write("\n")
    file.write("\nPOINTS: ")
    file.write("\n" + tuples_to_indented_string(pointsTime))
    file.write("\n")
    file.write("\nSLOPES: ")
    file.write("\n" + slopesTime_str)
    file.write("\n")
    file.write("\nPARAYIL METRIC: ")
    file.write("\n" + str(parayilMetricTime))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n-----------------------")
    file.write("\nRESULTS OF ALL TESTS PERFORMED (X-AXIS MODE: ITERATIONS):")
    file.write("\n" + tuples_to_indented_string(testHistory))
    file.write("\n")
    file.write("\n-----------------------")
    file.write("\n")
    file.write("\nEND OF QL 4x4 RESULTS.")
    file.write("\n-----------------------------------------------")

    return parayilMetricTime

"""
#Code for finding converge values and testing different trial amounts for accuracy
for i in range(10):
  performances = [0] * 500
  for i in range(500):
    run(2000)
    performances[i] = run(100,False)

  print(performances)
  print("Average")
  print(sum(performances)/len(performances))
"""


#======================================
#        FILE RESET FUNCTIONS
#
#           HANDLE WITH CARE
#======================================

def wipeData(filePath):
  with open(filePath, 'w') as file:
      file.write('')
  print("File cleared successfully:",filePath)

def wipeTestHistory():
    with open(dataFilePath, 'r') as file:
        data = json.load(file)

    data['testHistory'] = []

    with open(dataFilePath, 'w') as file:
        json.dump(data, file)

    print("testHistory cleared successfully.")

def wipeTimeTestHistory():
    with open(dataFilePath, 'r') as file:
        data = json.load(file)

    data['timeTestHistory'] = []

    with open(dataFilePath, 'w') as file:
        json.dump(data, file)

    print("timeTestHistory cleared successfully.")



#======================================#======================================#======================================#
#                                                    ACTUAL CODE                                                     #
#======================================#======================================#======================================#
"""
#Timing tests

resultsList = []
for i in range(50-44):
  resultsList.append(findParayilMetric(5))
  wipeTimeTestHistory()

average = sum(resultsList)/len(resultsList)

print("FINAL AVERAGE FINAL AVERAGE", average)

with open(resultFilePath, 'a') as file:
  file.write("\n\n\n\n\n\n\n\n\n\nAVERAGE FINAL AVERAGE FINAL:")
  file.write(str(average))

#Actual anew parayil metric testing

wipeTestHistory()
wipeTimeTestHistory()
"""


#   IMPORTANT NOTE: Trying to run the code will not work. The code for the 
#   training of QL/DQN on one of the environments is omitted and must be included.

findParayilMetric(5)

wipeTestHistory()
wipeTimeTestHistory()




