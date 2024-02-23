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
        for batch_idx ,(x,_) in enumerate(mnist_train):
            # [b,1,28,28]
            x = x.to(device)
            #相当于调用forward函数
            x_hat,kld= model(x)
            loss = criterion(x_hat,x)
            
            if kld is not None:
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


