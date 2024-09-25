import numpy as np
import matplotlib.pyplot as plt

#读入训练数据
train =np.loadtxt('image2.txt',delimiter='\t',skiprows=1)
train_x=train[:,0:2] 
train_y=train[:,2]

#参数初始化
theta=np.random.rand(3)

#标准化
mu=train_x.mean(axis=0)
sigma=train_x.std(axis=0)
def standard(x):
    return (x-mu)/sigma

train_z=standard(train_x)

#增加x0
def to_matrix(x):
    x0=np.ones((x.shape[0],1))
    return np.hstack([x0,x])

X=to_matrix(train_z)

#sigmoid函数
def f(x):
    return 1/(1+np.exp(-np.dot(x,theta)))

#分类函数
def classify(x):
    return (f(x)>=0.5).astype(int)

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


#绘图确认
x0=np.linspace(-2,2,100)
plt.plot(train_z[train_y==1,0],train_z[train_y==1,1],'o')
plt.plot(train_z[train_y==0,0],train_z[train_y==0,1],'x')
plt.plot(x0,-(theta[0]+theta[1]*x0)/theta[2],linestyle='dashed')
plt.show()

