import numpy as np
import scipy
import abc


class mirarLinearOperator(abc.ABC):
    @abc.abstractmethod
    def dot(self,x):
        pass
    @abc.abstractmethod
    def Tdot(self,y): 
        pass
    def isTransposed(self):
        return False
    def transpose(self):
        if self.isTransposed():
            return self.operator
        else:
            return transposedOperator(self)

class transposedOperator(mirarLinearOperator):
    def __init__(self,operator):
        self.operator = operator
    def dot(self,x):
        return self.operator.Tdot(x)
    def Tdot(self,y):
        return self.operator.dot(y)



class linearSeries(mirarLinearOperator):
    def __init__(self,operators):
        self.operators = operators
    def dot(self,x):
        for iOperator in np.arange(len(self.operators)):
            x = self.operators[-1*iOperator - 1].dot(x)
        return x
    def Tdot(self,y):
        for iOperator in np.arange(len(self.operators)):
            y = self.operators[iOperator].Tdot(y)
        return y

class denseMatrix(mirarLinearOperator):
    def __init__(self,matrix):
        self.matrix = matrix
    def dot(self,x):
        return self.matrix.dot(x)
    def Tdot(self,y):
        return self.matrix.transpose().dot(y)

class diagonalScale(mirarLinearOperator):
    def __init__(self,scale):
        self.scale = scale
    def dot(self,x):
        return self.scale*x
    def Tdot(self,y):
        return self.scale*y

class fourierTransform(mirarLinearOperator):
    def __init__(self,axes):
        self.axes = axes
    def dot(self,x):
        return np.fft.fftn(x,axes=self.axes,norm='ortho')
    def Tdot(self,y):
        return np.fft.ifftn(y,axes=self.axes,norm='ortho').real

#class fourierFilter(mirarLinearOperator):
#    def __init__(self,axes,fourierScale):
#        op1 = fourierTransform(axes)



