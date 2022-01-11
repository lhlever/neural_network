# 用“广播”的方式，求各个元素所在列的百分比
import numpy as np
A=np.array([[56.0,0.0,4.4,68.0],
            [1.2,104.0,52.0,8.0],
            [1.8,135.0,99.0,0.9]])
print(A)
cal=A.sum(axis=0)#axis=0代表竖直方向,axis=1代表水平方向
print('-'*50)
print(cal)
# print(cal.shape)
print('-'*50)
print(cal.reshape(1,4))
percentage=100*A/cal.reshape(1,4)#A矩阵是3*4矩阵，除以一个1*4的矩阵，这是一种广播的应用
# percentage=100*A/cal#A矩阵是3*4矩阵，除以一个1*4的矩阵
print('-'*50)
print(percentage)
