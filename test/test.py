import numpy as np
import mirar

import skimage

x = np.ones([3, 1],dtype=float)
matrix = np.ones([5,3],dtype=float)
A = mirar.denseMatrix(matrix)
ATAx = A.Tdot(Ax)

x = np.ones([3, 1],dtype=float)
scale = np.arange(3).astype(float)
scale.shape = x.shape
BTBx = B.Tdot(Bx)

x = np.ones([3, 1],dtype=float)
operators = (A, B)
AB = mirar.linearProduct(operators)
BTATABx = AB.Tdot(ABx)


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




