import numpy as np 
import matplotlib.pyplot as plt

#真正的函数
def g(x):
    return 0.1*(x**3+x**2+x)

#随意准备了一些向真正的函数加入了一点噪声的训练数据
train_x = np.linspace(-2,2,8)
train_y = g(train_x) + np.random.randn(train_x.size)*0.05

#绘图曲儿
x=np.linspace(-2,2,100)

#标准化
mu=train_x.mean()   
sigma=train_x.std()
def standardize(x):
    return (x-mu)/sigma

train_z=standardize(train_x)

#创建训练数据的矩阵
def to_matrix(x):
    return np.vstack([
        np.ones(x.size),
         x,
         x**2,
         x**3,
         x**4,
         x**5,
         x**6,
         x**7,
         x**8,
         x**9,
         x**10,
         ]).T

X=to_matrix(train_z)

#参数初始化
theta=np.random.randn(X.shape[1])
def f(x):
    return np.dot(x,theta)

#目标函数
def E(x,y):
    return 0.5*np.sum((y-f(x))**2)

#学习率s
eta=1e-4

#误差
diff=1

error=E(X,train_y)
#重复学习
while diff>1e-6:
    theta=theta-eta*np.dot(f(X)-train_y,X)
    current_error=E(X,train_y)
    diff=error-current_error
    error=current_error

#对结果绘图
z=standardize(x)    
# plt.plot(train_z,train_y,'o')
# plt.plot(z,f(to_matrix(z)))
# plt.show()

theta1=theta
theta=np.random.randn(X.shape[1])

#正则化常量
Lambda=1

#误差
diff=1

#重复学习
error=E(X,train_y)
while diff>1e-6:
    #正则化项，偏置项不适用正则化，所以为9
    reg_term=Lambda*np.hstack([0,theta[1:]])
    #应用正则化项，更新参数
    theta=theta-eta*np.dot(f(X)-train_y,X+reg_term)
    current_error=E(X,train_y)
    diff=error-current_error
    error=current_error

#对结果绘图
#z=standardize(x)    
# plt.plot(train_z,train_y,'o')
# plt.plot(z,f(to_matrix(z)))
# plt.show()

theta2=theta

plt.plot(train_z,train_y,'o')

#画出未应用正则化的结果
theta=theta1
plt.plot(z,f(to_matrix(z)),linestyle='dashed')

tehta=theta2
plt.plot(z,f(to_matrix(z)),linestyle='--')

plt.show()