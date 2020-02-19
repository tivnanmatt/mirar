import numpy as np
import mirar

x = np.ones([3, 1],dtype=float)
print('x:', x)

matrix = np.ones([5,3],dtype=float)
A = mirar.denseMatrix(matrix)
Ax = A.dot(x)
print('Ax: ', Ax)
ATAx = A.Tdot(Ax)
print('ATAx: ', ATAx)


scale = np.arange(3).astype(float)
scale.shape = x.shape
B = mirar.diagonalScale(scale)
Bx = B.dot(x)
print('Bx: ', Bx)
BTBx = B.Tdot(Bx)
print('BTBx: ', BTBx)


operators = (A, B)
AB = mirar.linearSeries(operators)
ABx = AB.dot(x)
print('ABx: ', ABx)
BTATABx = AB.Tdot(ABx)
print('BTATABx: ', BTATABx)


FT = mirar.fourierTransform(axes=(0,1))
IFT = FT.transpose()

print('FTx: ', FT.dot(x))
print('IFTFTx: ', IFT.dot(FT.dot(x)))


