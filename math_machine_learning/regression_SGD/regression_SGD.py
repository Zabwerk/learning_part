import numpy as np
import matplotlib.pyplot as plt



#读入训练数据
train=np.loadtxt('rain.txt')
train_x=train[:,0]
train_y=train[:,1]
#标准化
mu=train_x.mean()
sigma=train_x.std()
def standardize(x):
    return (x-mu)/sigma

train_z=standardize(train_x)

#初始化参数
theta=np.random.rand(3)

#创建训练数据矩阵
def to_matrix(x):
    return np.vstack([np.ones(x.shape[0]),x,x**2]).T

X=to_matrix(train_z)
#学习率
ETA=1e-3

#误差差值
diff=1

#更新次数
count=0
#预测函数
def f(x):
    return np.dot(x,theta)
def E(x,y):
    return 0.5*np.sum(y-f(x))**2
#误差的差值
diff=1


def MSE(x,y):
    return (1/x.shape[0])*np.sum((y-f(x))**2)

#用随机值初始化参数
theta=np.random.rand(3)

#均方误差的历史记录
errors=[]

errors.append(MSE(X,train_y))

while(diff>1e-2):
    #为了调整训练数据的顺序，准备随机的序列
    p=np.random.permutation(X.shape[0])

    #随机取出训练数据，使用SGD更新参数
    for x,y in zip(X[p,:],train_y[p]):
        theta=theta-ETA*(f(x)-y)*x
    
    errors.append(MSE(X,train_y))
    diff=errors[-2]-errors[-1]

#绘制误差变化图
x=np.linspace(-3,3,100)

plt.plot(train_z,train_y,'o')
plt.plot(x,f(to_matrix(x)))
plt.show()