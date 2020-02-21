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
    def isTransposed(self):
        return True
    def dot(self,x):
        return self.operator.Tdot(x)
    def Tdot(self,y):
        return self.operator.dot(y)

class linearSum(mirarLinearOperator):
    def __init__(self,operators):
        self.operators = operators
    def dot(self,x):
        xOut = 0.0
        for iOperator in np.arange(len(self.operators)):
            xOut = xOut + self.operators[iOperator].dot(x)
        return xOut
    def Tdot(self,y):
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
        return scipy.signal.convolve(x,self.kernel, mode='same')
    def Tdot(self,y):
        return scipy.signal.convolve(y,np.flip(self.kernel), mode='same')

class reorder(mirarLinearOperator):
    def __init__(self, forwardReorder):
        reverseOrder = 0*newOrder
        for iEl in np.newOrder:
            reverseReorder[iEl] = np.where(forwardReorder==iEl)
        self.forwardReorder = forwardReorder
        self.reverseReorder = reverseReorder
    def dot(self,x):
        return x.transpose(forwardReorder)
    def Tdot(self,y):
        return y.transpose(reverseReorder)

class reshape(mirarLinearOperator):
    def __init__(self,forwardReshape, reverseReshape):
        self.forwardReshape = forwardReshape
        self.reverseReshape = reverseReshape
    def dot(self,x):
        return x.reshape(self.forwardReshape)
    def Tdot(self,y):
        return y.reshape(self.reverseReshape)





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
    def hessp(self,x,p):
        hess = self.hessian(x)
        return hess.dot(p)

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
        return grad
    def hessian(self,x):
        hessianOperators = tuple(obj.hessian(x) for obj in self.objectiveTerms)
        return linearSum(hessianOperators)

class lexicoVersion(mirarObjectiveFunction):
    def __init__(self, originalObjective, shapeX):
        self.originalObjective = originalObjective
        forwardReshape = shapeX
        reverseReshape = tuple([np.prod(shapeX)])
        self.U = reshape(forwardReshape, reverseReshape)
    def objective(self,x):
        return self.originalObjective.objective(self.U.dot(x))
    def gradient(self,x):
        return self.U.Tdot(self.originalObjective.gradient(self.U.dot(x)))
    def hessian(self,x):
        return linearProduct( (self.U.transpose(), self.originalObjective.hessian(self.U.dot(x)), self.U) )

class multivariateGaussian(mirarObjectiveFunction):
    def __init__(self,meanX, invCovX):
        self.meanX = meanX 
        self.invCovX = invCovX
    def objective(self, x):
        obj = 0.5*(x - self.meanX)*self.invCovX.dot(x - self.meanX) 
        return np.sum(obj)
    def gradient(self, x):
        return self.invCovX.dot(x - self.meanX)
    def hessian(self, x):
        return self.invCovX

class quadraticSmoothness(mirarObjectiveFunction):
    def __init__(self, beta, kernel):
        self.R = convolutionFilter(beta*kernel)
    def objective(self, x):
        obj = 0.5*x*self.R.dot(x)
        return np.sum(obj)
    def gradient(self, x):
        return self.R.dot(x)
    def hessian(self, x):
        return self.R

class huberPenalty(mirarObjectiveFunction):
    def __init__(self, delta, beta, neighborhood):
        self.beta = beta
        self.delta = delta
        self.neighborhood = neighborhood
    def objective(self,x):
        obj = 0.0*x
        for iNeighbor in np.arange(self.neighborhood.shape[0]):
            r = x - np.roll(x, tuple(self.neighborhood[iNeighbor]), tuple(np.arange(len(self.neighborhood[iNeighbor]))))
            idxLinear = np.abs(r) > self.delta
            obj[~idxLinear] += 0.5*r[~idxLinear]*r[~idxLinear]
            obj[idxLinear]  += self.delta*np.abs(r[idxLinear]) - 0.5*self.delta*self.delta
        return np.sum(self.beta*obj)
    def gradient(self,x):
        grad = 0.0*x
        for iNeighbor in np.arange(self.neighborhood.shape[0]):
            r = x - np.roll(x, tuple(self.neighborhood[iNeighbor]), tuple(np.arange(len(self.neighborhood[iNeighbor]))))
            idxLinear = np.abs(r) > self.delta
            grad[~idxLinear] += r[~idxLinear]
            grad[idxLinear]  += self.delta*np.sign(r[idxLinear]) 
        return self.beta*grad
    class huberHessian(mirarLinearOperator):
        def __init__(self, huberObjective, xOperatingPoint):
            self.huberObjective = huberObjective
            self.xOperatingPoint = xOperatingPoint
        def dot(self,x): 
            xOut = 0.0*x
            for iNeighbor in np.arange(self.huberObjective.neighborhood.shape[0]):
                rOperatingPoint = self.xOperatingPoint - np.roll(self.xOperatingPoint, tuple(self.huberObjective.neighborhood[iNeighbor]), tuple(np.arange(len(self.huberObjective.neighborhood[iNeighbor]))))
                idxLinear = np.abs(rOperatingPoint) > self.huberObjective.delta
                r = x - np.roll(x, tuple(self.huberObjective.neighborhood[iNeighbor]), tuple(np.arange(len(self.huberObjective.neighborhood[iNeighbor]))))
                xOut[~idxLinear] += r[~idxLinear]
                xOut[idxLinear]  += self.huberObjective.delta*np.sign(r[idxLinear]) 
            return self.huberObjective.beta*xOut
        def Tdot(self,y):
            return self.dot(x)
    def hessian(self,x):
        return self.huberHessian(self, x)





