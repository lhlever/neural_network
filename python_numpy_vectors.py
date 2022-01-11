import numpy as np
a=np.random.randn(5)#生成秩为1的数组，它既不是行向量也不是列向量
a=np.random.randn(5,1)#生成的是5*1的矩阵
print(a)
print(a.shape)
print(a.T)
a=a.T
assert(a.shape==(5,1))
# print(a)