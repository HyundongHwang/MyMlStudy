import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt



names = np.array(['bob', 'joe', 'will', 'bob', 'joe', 'will'])
result = np.unique(names)
print("np.unique(names) : ", result)



exlnv = np.array([[2,2,0],[-2,1,1],[3,0,1]])
result = lin.inv(exlnv)
print("lin.inv(exlnv) : ", result)



exSolveA = np.array([[3,2,1],[1,-1,3],[5,4,-2],])
exSolveB = np.array([7,3,1])
result = lin.solve(exSolveA, exSolveB)
print("lin.solve(exSolveA, exSolveB) : ", result)



xLstsq = np.array([0,1,2,3])
yLstsq = np.array([-1, 0.2, 0.9, 2.1])
plt.plot(xLstsq, yLstsq)
plt.grid(True)
plt.show()



data1 = np.array([[1,2,3,],[4,5,6,],[7,8,9,],])
print("data1 : ", data1)
print("data1.dtype : ", data1.dtype)



num_str = np.array(['1.25', '0.99', '44', '11'], dtype=None)