import numpy as np
import random
# a = np.random.rand(5)
# print(a.dtype)
# print(random.choices(range(5),a,k = 1))


X = np.zeros((2,2,3,1))
V = np.zeros((2,2,3,3))

a = np.array([[5,6],[7,8]])

b = np.array([[1],[2]])
print(a)
print(b)
print(a@b)

# print(a ** 2 )

# X[:,0,0] = np.ones(3)
# X[:,1,0] = a
# print(X[:,:,0])

# X[:,:,:,0] = np.array([a ** k for k in range(3)]).T
# print (X[:,:] @ X[:,:].T)
# print(X[0,0])
# print(X[0,0] @ X[0,0].T)
# print(V)
# print(X)

# b = np.zeros((3,2,2))
# c = np.ones((3,2,1))
# d = np.zeros((3,2,2))
# for i in range(3):
#     d[i] = c[i] @ c[i].T

# print(d)

def fun():
    return 1,2,3

A,B,C = fun()

print(A,B,C)