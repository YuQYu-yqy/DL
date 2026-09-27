import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
#数据加载
def load_data_fashion_mnist(batch_size,resize=None):
    trans=[transforms.ToTensor()]
    if resize:
        trans.insert(0,transforms.Resize(resize))
    trans=transforms.Compose(trans)
    mnist_train=datasets.FashionMNIST(root='./data',train=True,download=True,transform=trans)
    mnist_test=datasets.FashionMNIST(root='./data',train=False,download=True,transform=trans)
    return (DataLoader(mnist_train,batch_size,shuffle=True,num_workers=0),DataLoader(mnist_test,batch_size,shuffle=False,num_workers=0))
batch_size=256
train_iter,test_iter=load_data_fashion_mnist(batch_size)
#SGD
def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param-=lr*param.grad/batch_size
            param.grad.zero_()
#标签映射
def get_fashion_mnist_labels(labels):
    text_labels=['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle boot']
    return [text_labels[int(i)] for i in labels]
#显示图像
def show_images(imgs,num_rows,num_cols,titles=None,scale=1.5):
    figsize=(num_cols*scale,num_rows*scale)
    _,axes=plt.subplots(num_rows,num_cols,figsize=figsize)
    axes=axes.flatten() if hasattr(axes,'flatten') else [axes]
    for i,(ax,img) in enumerate(zip(axes,imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.detach().numpy(),cmap='gray')
        else:
            ax.imshow(img,cmap='gray')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes
#初始化模型参数
num_inputs=784
num_outputs=10
W=torch.normal(0,0.01,size=(num_inputs,num_outputs),requires_grad=True)
b=torch.zeros(num_outputs,requires_grad=True)
#softmax
def softmax(X):
    X_exp=torch.exp(X)
    partition=X_exp.sum(1,keepdim=True)
    return X_exp/partition
X_demo=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print("sum(0, keepdim=True):",X_demo.sum(0,keepdim=True))
print("sum(1, keepdim=True):",X_demo.sum(1,keepdim=True))
X=torch.normal(0,1,(2,5))
X_prob=softmax(X)
print("softmax 输出：",X_prob)
print("每行和：",X_prob.sum(1))
#模型
def net(X):
    return softmax(torch.matmul(X.reshape((-1,W.shape[0])),W)+b)
#交叉熵
def cross_entropy(y_hat,y):
    return -torch.log(y_hat[torch.arange(len(y_hat)),y])
y=torch.tensor([0,2])
y_hat=torch.tensor([[0.1,0.3,0.6],[0.3,0.2,0.5]])
print("y_hat[[0, 1], y] =",y_hat[[0,1],y])
print("cross_entropy =",cross_entropy(y_hat,y))
#精度
def accuracy(y_hat,y):
    if len(y_hat.shape)>1 and y_hat.shape[1]>1:
        y_hat=y_hat.argmax(axis=1)
    cmp=y_hat.type(y.dtype)==y
    return float(cmp.type(y.dtype).sum())
print("演示精度：",accuracy(y_hat,y)/len(y))
#累加器
class Accumulator:
    def __init__(self,n):
        self.data=[0.0]*n
    def add(self,*args):
        self.data=[a+float(b) for a,b in zip(self.data,args)]
    def reset(self):
        self.data=[0.0]*len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]
#评估精度
def evaluate_accuracy(net,data_iter):
    if isinstance(net,torch.nn.Module):
        net.eval()
    metric=Accumulator(2)
    with torch.no_grad():
        for X,y in data_iter:
            metric.add(accuracy(net(X),y),y.numel())
    return metric[0]/metric[1]
print("随机权重下的测试精度：",evaluate_accuracy(net,test_iter))
#训练一个迭代周期
def train_epoch_ch3(net,train_iter,loss,updater):
    if isinstance(net,torch.nn.Module):
        net.train()
    metric=Accumulator(3)
    for X,y in train_iter:
        y_hat=net(X)
        l=loss(y_hat,y)
        if isinstance(updater,torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()),accuracy(y_hat,y),y.numel())
    return metric[0]/metric[2],metric[1]/metric[2]
#训练函数
def train_ch3(net,train_iter,test_iter,loss,num_epochs,updater):
    train_losses,train_accs,test_accs=[],[],[]
    for epoch in range(num_epochs):
        train_metrics=train_epoch_ch3(net,train_iter,loss,updater)
        test_acc=evaluate_accuracy(net,test_iter)
        train_losses.append(train_metrics[0])
        train_accs.append(train_metrics[1])
        test_accs.append(test_acc)
        print(f"epoch {epoch+1:2d}: train loss {train_metrics[0]:.4f}, train acc {train_metrics[1]:.4f}, test acc {test_acc:.4f}")
    epochs=range(1,num_epochs+1)
    plt.figure(figsize=(6,4))
    plt.plot(epochs,train_losses,'-',label='train loss')
    plt.plot(epochs,train_accs,'m--',label='train acc')
    plt.plot(epochs,test_accs,'g-.',label='test acc')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.ylim(0.3,0.9)
    plt.legend()
    plt.grid()
    plt.show()
    train_loss,train_acc=train_metrics
    assert train_loss<0.5,train_loss
    assert 0.7<train_acc<=1,train_acc
    assert 0.7<test_acc<=1,test_acc
#优化器与训练
lr=0.1
def updater(batch_size):
    return sgd([W,b],lr,batch_size)
num_epochs=10
train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,updater)
#预测
def predict_ch3(net,test_iter,n=6):
    for X,y in test_iter:
        break
    trues=get_fashion_mnist_labels(y)
    preds=get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles=[true+'\n'+pred for true,pred in zip(trues,preds)]
    show_images(X[0:n].reshape((n,28,28)),1,n,titles=titles[0:n])
    plt.show()
predict_ch3(net,test_iter)