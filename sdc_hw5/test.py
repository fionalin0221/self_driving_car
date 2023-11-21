import numpy as np
def test():
    return 0

a = np.array([[1,2,3],[4,5,6]])
# print(a)
# print(a.size)
# print(a.shape)

# a = 10
b = a
print(a)
print(b)
a[1] = 0
print(a)
print(b)
# print(id(a))
# print(id(b))