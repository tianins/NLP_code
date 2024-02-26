import matplotlib.pyplot as pyplot
import math

X = [0.01 * i for i in range(100)]
Y = [2 * x**2 + 3*x + 4 for x in X]
# print(X)
# print(Y)
# pyplot.scatter(X, Y)
# pyplot.show()



def func(x):
    y = w1 * x**2 + w2 * x + w3
    return y

def loss(y_pred, y_true):
    return (y_pred - y_true) ** 2



# 权重随机初始化
w1, w2, w3 = 1, -1, -1

# 学习率
lr = 0.1


for epoch in range(1000):
    epoch_loss = 0
    for x, y_true in zip(X, Y):
        y_pred = func(x)
        epoch_loss += loss(y_pred, y_true)
        #梯度计算
        grad_w1 = 2 * (y_pred - y_true) * x ** 2
        grad_w2 = 2 * (y_pred - y_true) * x
        grad_w3 = 2 * (y_pred - y_true) 
        w1 = w1 - lr * grad_w1
        w2 = w2 - lr * grad_w2
        w3 = w3 - lr * grad_w3
        
    epoch_loss /= len(X)
    print("第%d轮， loss %f" %(epoch, epoch_loss))
    if epoch_loss < 0.001:
        break

print(f"训练后权重:w1:{w1} w2:{w2} w3:{w3}")


# Y1 = [func(i) for i in X]

# pyplot.scatter(X, Y, color="red")
# pyplot.scatter(X, Y1)
# pyplot.show()
