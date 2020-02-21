import numpy as np
import mirar

import scipy

import skimage.transform
import skimage.data
from matplotlib import pyplot as plt


x = np.ones([3, 1],dtype=float)
matrix = np.ones([5,3],dtype=float)
A = mirar.denseMatrix(matrix)
ATAx = A.Tdot(A.dot(x))

x = np.ones([3, 1],dtype=float)
scale = np.arange(3).astype(float)
scale.shape = x.shape
B = mirar.diagonalScale(scale)
BTBx = B.Tdot(B.dot(x))

x = np.ones([3, 1],dtype=float)
operators = (A, B)
AB = mirar.linearProduct(operators)
BTATABx = AB.Tdot(AB.dot(x))


x = np.zeros([11,1],dtype=float)
x[3] = 1.0
x[7] = 1.0
FT = mirar.fourierTransform()
IFT = FT.transpose()
IFT.dot(FT.dot(x))


h = np.zeros([11,1],dtype=float)
h[-1] = 1.5
h[0] = 2.0
h[1] = 1.5 
H = np.sqrt(11)*FT.dot(h)
D = mirar.fourierFilter(fourierScale=H)
D.Tdot(D.dot(x))














P = skimage.data.shepp_logan_phantom()
P = skimage.transform.resize(P,[512,512])
varP = .03
P = P + np.random.normal(0,np.sqrt(varP),P.shape)

meanX = P
varX = varP
invCovX = mirar.diagonalScale(1.0/varX)
obj_likelihood = mirar.multivariateGaussian(meanX, invCovX)

beta = 10000
kernel = np.array( [[ 0, -1,  0],
                    [-1,  4, -1],
                    [ 0, -1,  0]]).astype(float)
obj_smoothness1 = mirar.quadraticSmoothness(beta, kernel)

neighborhood = np.array([ (+1, 0),
                          (-1, 0),
                          (0, +1),
                          (0, -1)]).astype(int)
beta = 1000
delta = 0.01
obj_smoothness2 = mirar.huberPenalty(delta, beta, neighborhood)


obj_smoothness = obj_smoothness2

obj_joint = mirar.jointObjective( (obj_likelihood, obj_smoothness) )

obj_final = mirar.lexicoVersion(obj_joint, P.shape)

x0 = 0*P
x0 = P


def cBack(xk):
    print("HELLO: ", obj_final.objective(xk))


res = scipy.optimize.minimize(      obj_final.objective, 
                                    x0, 
                                    method='Newton-CG', 
                                    jac=obj_final.gradient, 
                                    hessp=obj_final.hessp,
                                    options={'maxiter': 5, 'xtol': 1e-6, 'disp': True},
                                    callback = cBack)

print("D ", obj_smoothness2.objective(res.x.reshape(P.shape) ) )

plt.figure()
plt.subplot(1,2,1)
plt.imshow(res.x.reshape(P.shape) )
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(P.reshape(P.shape) )
plt.colorbar()
plt.show()

