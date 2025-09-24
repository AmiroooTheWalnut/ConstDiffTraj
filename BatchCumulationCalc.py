import math

def batchCumulationCalc(totalIter, batchSize):
    outputCumulativeGradIters = (int)(math.ceil(totalIter/batchSize))

    return outputCumulativeGradIters