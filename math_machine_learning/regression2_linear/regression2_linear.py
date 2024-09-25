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

#重复学习
error=E(X,train_y)


while diff>1e-2:
    #更新参数
    theta=theta-ETA*np.dot(f(X)-train_y,X)
    #计算新的误差
    new_error=E(X,train_y)
    #计算误差的差值
    diff=error-new_error
    error=new_error
    count+=1
    #print('diff:',diff,'error:',error,'count:',count)
    log="第{}次：误差差值={:3f},误差={:3f}"
    print(log.format(count,diff,error))

x=np.linspace(-3,3,100)

plt.plot(train_z,train_y,'b.')
plt.plot(x,f(to_matrix(x)),'r-')
plt.show()