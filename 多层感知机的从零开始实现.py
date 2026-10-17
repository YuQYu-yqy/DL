import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
batch_size=256
num_inputs,num_outputs,num_hiddens=784,10,256
num_epochs,lr=10,0.1
#加载数据集
transform = transforms.ToTensor()
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform)

train_iter = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)
test_iter = DataLoader(test_dataset, batch_size=batch_size,
                       shuffle=False, num_workers=0)
#设参
W1=nn.Parameter(torch.randn(num_inputs,num_hiddens)*0.01)
b1=nn.Parameter(torch.zeros(num_hiddens))
W2=nn.Parameter(torch.randn(num_hiddens,num_outputs)*0.01)
b2=nn.Parameter(torch.zeros(num_outputs))
params=[W1,b1,W2,b2]
#激活函数
def relu(X):
    a=torch.zeros_like(X)
    return torch.max(X,a)
#模型
def net(X):
    X=X.reshape((-1,num_inputs))
    H=relu(X@W1+b1)
    return H@W2+b2
#损失函数
loss=nn.CrossEntropyLoss(reduction="none")
#累加程序
class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    def reset(self):
        self.data = [0.0] * len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]
#分类精度
def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())
#评估精度
def evaluate_accuracy(net, data_iter):
    if isinstance(net, nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]
#训练一个epoch
def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, nn.Module):
        net.train()
    metric = Accumulator(3)          # 训练损失之和、正确数、样本数
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]
#训练主函数
def train(net, train_iter, test_iter, loss, num_epochs, updater):
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        train_loss, train_acc = train_metrics
        print(f'epoch {epoch + 1}, '
              f'train loss {train_loss:.4f}, '
              f'train acc {train_acc:.4f}, '
              f'test acc {test_acc:.4f}')
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc
#预测
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def predict(net, test_iter, n=10):
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    for i in range(n):
        print(f'真实: {trues[i]:12s}  预测: {preds[i]}')
#训练
updater=torch.optim.SGD(params,lr=lr)
train(net,train_iter,test_iter,loss,num_epochs,updater)
predict(net,test_iter,n=10)