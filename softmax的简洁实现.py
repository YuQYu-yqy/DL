import torch
from torch import nn
from torch.utils import data
from torchvision import datasets, transforms
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
batch_size=256
# 加载Fashion-MNIST数据集
def load_data_fashion_mnist(batch_size):
    trans=transforms.ToTensor()
    mnist_train=datasets.FashionMNIST('./data',train=True,download=True,transform=trans)
    mnist_test=datasets.FashionMNIST('./data',train=False,download=True,transform=trans)
    train_iter=data.DataLoader(mnist_train,batch_size,shuffle=True,num_workers=0)
    test_iter=data.DataLoader(mnist_test,batch_size,shuffle=False,num_workers=0)
    return train_iter,test_iter
train_iter,test_iter=load_data_fashion_mnist(batch_size)
# 初始化模型参数
net=nn.Sequential(nn.Flatten(),nn.Linear(784,10))
net[1].weight.data.normal_(0,0.01)
net[1].bias.data.fill_(0)
# 损失函数：Softmax与交叉熵结合
loss=nn.CrossEntropyLoss(reduction='sum')
# 优化算法：小批量随机梯度下降
trainer=torch.optim.SGD(net.parameters(),lr=0.1)
# 计算准确率
def accuracy(y_hat,y):
    return (y_hat.argmax(1)==y).sum().item()
# 评估准确率
def evaluate_accuracy(net,data_iter):
    acc_sum,n=0.0,0
    for X,y in data_iter:
        acc_sum+=accuracy(net(X),y)
        n+=y.size(0)
    return acc_sum/n
# 训练
def train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer):
    for epoch in range(num_epochs):
        train_loss_sum,train_acc_sum,n=0.0,0.0,0
        for X,y in train_iter:
            y_hat=net(X)
            l=loss(y_hat,y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
            train_loss_sum+=l.item()
            train_acc_sum+=accuracy(y_hat,y)
            n+=y.size(0)
        test_acc=evaluate_accuracy(net,test_iter)
        print(f'epoch {epoch+1}, loss {train_loss_sum/n:.3f}, train acc {train_acc_sum/n:.3f}, test acc {test_acc:.3f}')
num_epochs=10
train_ch3(net,train_iter,test_iter,loss,num_epochs,trainer)