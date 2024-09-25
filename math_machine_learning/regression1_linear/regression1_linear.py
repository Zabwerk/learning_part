import numpy as np 
import matplotlib.pyplot as plt
import chardet


# with open('G:\\rain.xlsx','rb') as f:
#     raw_data=f.read()
#     result=chardet.detect(raw_data)
#     encoding=result['encoding']    
# print(encoding)


#读入训练数据
train=np.loadtxt('rain.txt')
train_x=train[:,0]
train_y=train[:,1]

# #绘图
# plt.plot(train_x,train_y,'o')
# plt.show()

#参数初始化
theta0=np.random.rand()
theta1=np.random.rand()

#预测函数
def f(x):
    return theta0+theta1*x

def E(x,y):
    return 0.5*np.sum(y-f(x))**2

#标准化
mu=train_x.mean()
sigma=train_x.std()
def standardize(x):
    return (x-mu)/sigma

train_z=standardize(train_x)

#学习率
ETA=1e-3

#误差差值
diff=1

#更新次数
count=0

#重复学习
error=E(train_z,train_y)
while diff>1e-2:
    #更新结果保存到临时变量
    temp0=theta0-ETA*np.sum((f(train_z)-train_y))
    temp1=theta1-ETA*np.sum((f(train_z)-train_y)*train_z)
    #更新参数
    theta0=temp0
    theta1=temp1

    #计算与上一次误差的差值
    current_error=E(train_z,train_y)
    diff=error-current_error
    error=current_error

    #输出日志
    count+=1
    log="第{}次：theta0={:3f},theta1={:3f},差值={:4f}"
    print(log.format(count,theta0,theta1,diff))

x=np.linspace(-3,3,100)

plt.plot(train_z,train_y,'ro')
plt.plot(x,f(x))
plt.show()