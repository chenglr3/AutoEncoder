import torch
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
from torch import nn,optim

from VAE import VAE

import visdom

def main():
    mnist_train  = datasets.MNIST(
        'mnist',True,transform=transforms.Compose([transforms.ToTensor()]),download=True
    )
    #DataLoader负责数据的分批，采样和传输
    mnist_train = DataLoader(mnist_train,batch_size=32,shuffle=True)

    mnist_test  = datasets.MNIST(
        'mnist',False,transform=transforms.Compose([transforms.ToTensor()]),download=True
    )
    mnist_test = DataLoader(mnist_test,batch_size=32,shuffle=True)

    # #获取下一批次的数据
    # # x,_ = next(iter(mnist_train))
    # x , _ = iter(mnist_train).next()
    # print('x形状大小:',x.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #实例化自编码器
    model = VAE().to(device)
    #评价标准
    criterion = nn.MSELoss()
    #优化器
    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    print('model:\n',model)
    #可视化
    viz = visdom.Visdom()


    for epoch in range(10):
        #遍历训练集的数据
        '''
        mnist_train作为一个DataLoader对象，其输出形式是一个可迭代的批次数据。当你在训练循环中迭代mnist_train时，每次迭代都会输出一个批次的数据，这个批次包含一对元素：输入数据和对应的标签。
        每个批次的数据形式为：(batch_images, batch_labels)，其中：
        batch_images 是一个形状为 [batch_size, 1, 28, 28] 的张量，表示一批图像数据。1 表示图像是单通道（灰度图），28, 28 是图像的高度和宽度。
        batch_labels 是一个形状为 [batch_size] 的张量，表示这批图像对应的标签（数字0到9）。
        '''
        for batch_idx ,(x,_) in enumerate(mnist_train):
            # [b,1,28,28]
            x = x.to(device)
            #相当于调用forward函数
            x_hat,kld= model(x)
            loss = criterion(x_hat,x)
            
            if kld is not None:
                '''
                原先的loss:重构误差 VS kld 误差
                重构过程是希望没噪声的，KL loss是希望有噪声的，二者互相对抗
                当重构loss > KLd loss，就降低噪声，使得拟合更加容易一点
                当重构loss < Kld loss,就增加噪声，拟合困难，这是decoder就要想办法提高生成能力
                '''
                elbo = - loss - 1.0 * kld
                loss = - elbo

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(epoch,'loss:',loss.item(),'kld:',kld.item())

        #测试集上可视化
        x_test,_ = next(iter(mnist_test))
        x_test = x_test.to(device)
        with torch.no_grad():
            x_test_hat,kld = model(x_test)

        viz.images(x_test,nrow=8,win='x_test',opts=dict(title='x_test'))
        viz.images(x_test_hat,nrow=8,win='x_test_hat',opts=dict(title='x_test_hat'))







if __name__ == '__main__':
    main()


