import math
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
#生成数据集
max_degree = 20
n_train, n_test = 100, 100
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])
features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 10)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)
true_w, features, poly_features, labels = [
    torch.tensor(x, dtype=torch.float32)
    for x in [true_w, features, poly_features, labels]
]
print(features[:2], poly_features[:2, :], labels[:2])
#训练和测试
def evaluate_loss(net, data_iter, loss):
    net.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for X, y in data_iter:
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            total_loss += l.sum().item()
            total_samples += l.numel()
    return total_loss / total_samples
def train_epoch(net, train_iter, loss, trainer):
    net.train()
    total_loss = 0.0
    total_samples = 0
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        trainer.zero_grad()
        l.mean().backward()
        trainer.step()
        total_loss += l.sum().item()
        total_samples += l.numel()
    return total_loss / total_samples
def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_dataset = TensorDataset(train_features, train_labels.reshape(-1, 1))
    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(test_features, test_labels.reshape(-1, 1))
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    epochs_list = []
    train_loss_list = []
    test_loss_list = []
    for epoch in range(num_epochs):
        train_epoch(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            epochs_list.append(epoch + 1)
            train_loss_list.append(evaluate_loss(net, train_iter, loss))
            test_loss_list.append(evaluate_loss(net, test_iter, loss))
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print('weight:', net[0].weight.data.numpy())
    #绘图
    plt.figure()
    plt.semilogy(epochs_list, train_loss_list, label='train')
    plt.semilogy(epochs_list, test_loss_list, label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlim(1, num_epochs)
    plt.ylim(1e-3, 1e2)
    plt.legend()
    plt.grid(True)
    plt.show()
#三阶多项式函数拟合
train(
    poly_features[:n_train, :4],
    poly_features[n_train:, :4],   
    labels[:n_train],
    labels[n_train:]
)
#线性函数欠拟合
train(
    poly_features[:n_train, :2],
    poly_features[n_train:, :2],
    labels[:n_train],
    labels[n_train:]
)
#高阶多项式函数过拟合
train(
    poly_features[:n_train, :],
    poly_features[n_train:, :],
    labels[:n_train],
    labels[n_train:],
    num_epochs=1500
)