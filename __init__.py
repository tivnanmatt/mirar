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

class linearSum(mirarLinearOperator):
    def __init__(self,operators):
        def dot(self,x):
            xOut = 0.0
            for iOperator in np.arange(len(self.operators)):
                xOut = xOut + self.operators[iOperator].dot(x)
            return xOut
        def dot(self,y):
            yOut = 0.0
            for iOperator in np.arange(len(self.operators)):
                yOut = yOut + self.operators[iOperator].Tdot(y)
            return yOut

class linearProduct(mirarLinearOperator):
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
    def __init__(self,axes=None):
        self.axes = axes
    def dot(self,x):
        return np.fft.fftn(x,axes=self.axes,norm='ortho')
    def Tdot(self,y):
        return np.fft.ifftn(y,axes=self.axes,norm='ortho').real

class fourierFilter(mirarLinearOperator):
    def __init__(self,fourierScale,axes=None):
        op1 = fourierTransform(axes)
        op2 = diagonalScale(fourierScale)
        op3 = op1.transpose()
        self.operator = linearProduct((op3,op2,op1))
    def dot(self,x):
        return self.operator.dot(x)
    def Tdot(self,y):
        return self.operator.Tdot(y)


class convolutionFilter(mirarLinearOperator):
    def __init__(self,kernel):
        self.kernel = kernel
    def dot(self,x):
        return scipy.convolve(x,kernel)
    def Tdot(self,y):
        return scipy.convolve(y,np.flip(kernel))


class mirarObjectiveFunction(abc.ABC):
    @abc.abstractmethod
    def objective(self,x):
        pass
    @abc.abstractmethod
    def gradient(self,x): 
        pass
    @abc.abstractmethod
    def hessian(self,x): 
        pass

class jointObjective(mirarObjectiveFunction):
    def __init__(self,objectiveTerms):
        self.objectiveTerms = objectiveTerms
    def objective(self,x):
        obj = 0.0
        for iObj in np.arange(len(self.objectiveTerms)):
            obj = obj + self.objectiveTerms[iObj].objective(x)
        return obj
    def gradient(self,x):
        grad = 0.0
        for iObj in np.arange(len(self.objectiveTerms)):
            grad = grad + self.objectiveTerms[iObj].gradient(x)
    def hessian(self,x):
        hessianOperators = tuple(obj.hessian(x) for obj in np.arange(len(self.objectiveTerms)))
        return linearSum(hessianOperators)

class multivariateGaussian(mirarObjectiveFunction):
    def __init__(self,meanX, invCovX):
        self.meanX = meanX 
        self.invCovX = invCovX
    def objective(x):
        return 0.5*(x - self.meanX)*self.covX.dot(x - self.meanX) 
    def gradient(x):
        return self.covX.dot(x - self.meanX)
    def hessian(x):
        return self.covX

class quadraticSmoothness(mirarObjectiveFunction):
    def __init__(self, beta, kernel):
        self.R = convolutionFilter(beta*kernel)
    def objective(x):
        return 0.5*x*self.R.dot(x)
    def gradient(x):
        return self.R.dot(x)
    def hessian(x):
        return self.R


