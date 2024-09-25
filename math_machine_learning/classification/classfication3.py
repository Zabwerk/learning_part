import numpy as np 
import matplotlib.pyplot as plt

#读入训练数据
train=np.loadtxt("F:\尝试写一些东西\读书笔记\白话机器学习的数学\code\classification\data3.txt",delimiter='\t',skiprows=1)
train_x=train[:,0:2]
train_y=train[:,2]

#参数初始化
theta=np.random.rand(4)

#标准化
mu=train_x.mean(axis=0)
sigma=train_x.std(axis=0)
def standard(x):
    return (x-mu)/sigma

train_z=standard(train_x)

#增加x0和x3
def to_matrix(x):
    x0=np.ones((x.shape[0],1))
    x3=x[:,0,np.newaxis]**2
    return np.hstack([x0,x,x3])

X=to_matrix(train_z)

#sigmoid函数
def f(x):
    return 1/(1+np.exp(-np.dot(x,theta)))

#学习率
eta=1e-3

#重复次数
epoch=5000

#更新次数
count=0 

#重复学习
for _ in range(epoch):
    theta=theta-eta*np.dot(f(X)-train_y,X)
    count+=1
    print('第{}次更新权重为{}'.format(count,theta)) 

x1=np.linspace(-2,2,100)
x2=-(theta[0]+theta[1]*x1+theta[3]*x1**2)/theta[2]

# plt.plot(train_z[train_y==1,0],train_z[train_y==1,1],'o')
# plt.plot(train_z[train_y==0,0],train_z[train_y==0,1],'x')
# plt.plot(x1,x2,linestyle='dashed')
# plt.show()

#参数初始化
theta=np.random.rand(4)

#重复学习
for _ in range(epoch):
    p=np.random.permutation(X.shape[0])
    for x,y in zip(X[p,:],train_y[p]):
        theta=theta-eta*(f(x)-y)*x

#绘图确认
plt.plot(train_z[train_y==1,0],train_z[train_y==1,1],'o')
plt.plot(train_z[train_y==0,0],train_z[train_y==0,1],'x')
plt.plot(x1,x2,linestyle='dashed')
plt.show()