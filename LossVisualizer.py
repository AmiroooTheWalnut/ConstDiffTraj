import numpy
import matplotlib.pyplot as plt

class LossVisualizer():
    def __init__(self, numIterations):
        self.values=numpy.zeros(numIterations)

    def visualize(self,saveFileName=None, titleText="Loss value over iterations"):
        plt.plot(self.values)
        plt.title(titleText)
        if saveFileName!=None:
            plt.savefig(saveFileName)
        plt.show()