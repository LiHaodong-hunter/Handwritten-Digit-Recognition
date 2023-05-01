# 机 构：中国科学院大学
# 程序员：李浩东
# 时 间：2023/3/27 17:19

import torch
import torchvision
from torch import nn

# 定义超参数
EPOCHS = 20
BATCH_SIZE = 256
LR = 0.001

device = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])
train_data = torchvision.datasets.MNIST('data/', train=True, transform=transform, download=True)  # 训练集
test_data = torchvision.datasets.MNIST('data/', train=False, transform=transform)  # 测试集
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一个卷积池化层：输入图像大小(1,28,28),输出图像大小(16,14,14)
        self.cov1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.5),  # 正则化，丢掉50%的神经元，防止过拟合
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 第二个卷积池化层：输入图像大小(16,14,14),输出图像大小(32,7,7)
        self.cov2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.5),  # 正则化，丢掉50%的神经元，防止过拟合
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # 第三个卷积池化层：输入图像大小(32,7,7),输出图像大小(64,2,2)
        self.cov3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=1),
            nn.Dropout(p=0.5),  # 正则化，丢掉50%的神经元，防止过拟合
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            # 两个全连接层
            nn.Linear(64 * 2 * 2, 64),
            nn.Dropout(p=0.5),  # 正则化，丢掉50%的神经元，防止过拟合
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        # print(x.shape)  # torch.Size([256, 1, 28, 28])  BATCH_SIZE = 256
        x = self.cov1(x)
        # print(x.shape)  # torch.Size([256, 16, 14, 14])
        x = self.cov2(x)
        # print(x.shape)  # torch.Size([256, 32, 7, 7])
        x = self.cov3(x)
        # print(x.shape)  # torch.Size([256, 64, 2, 2])
        # 平化
        x = x.view(x.size(0), -1)  # 把x变换成BATCH_SIZE行个tensor
        # print(x.shape)  # torch.Size([256, 256])
        # nn.Flatten(x)  # torch.Size([21632, 256])
        output = self.fc(x)
        return output


cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR, weight_decay=0.001)  # 优化器
loss_func = torch.nn.CrossEntropyLoss()  # 损失函数

# 训练模型
for epoch in range(EPOCHS):
    cnn.train(True)
    for step, (train_img, train_label) in enumerate(train_loader):
        train_img = train_img.to(device)
        train_label = train_label.to(device)

        train_output = cnn(train_img)
        train_loss = loss_func(train_output, train_label)
        optimizer.zero_grad()  # 清除梯度
        train_prediction = torch.argmax(train_output, dim=1)
        train_accuracy = torch.sum(train_prediction == train_label) / train_label.shape[0]
        train_loss.backward()  # 反向传播
        optimizer.step()  # 应用梯度

        # 测试模型
        if step % 50 == 0:
            correct = total = 0
            cnn.train(False)
            with torch.no_grad():
                for _, (test_img, test_label) in enumerate(test_loader):
                    test_img = test_img.to(device)
                    test_label = test_label.to(device)

                    test_output = cnn(test_img)
                    test_loss = loss_func(test_output, test_label)
                    # predictions = torch.argmax(outputs, dim=1)
                    pred_label = torch.max(test_output, 1)[1].detach().numpy()
                    correct += float((pred_label == test_label.detach().numpy()).astype(int).sum())
                    total += float(test_label.size(0))

            test_accuracy = correct / total
            print('Epoch: ', epoch+1, '| train loss: %.4f' % train_loss.data.numpy(), '| train accuracy: %.4f' % train_accuracy.data.numpy(),
                  '| tset loss: %.4f' % test_loss.data.numpy(),'| test accuracy: %.4f' % test_accuracy)

# 保存模型
torch.save(cnn.state_dict(), 'cnn.pth')

