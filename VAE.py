import torch
from torch import nn
import numpy as np

class VAE(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        #[b,784] -> [b,20]
        # mu:[b,10]
        #sigma:[b,10]
        self.encoder = nn.Sequential(
            nn.Linear(784,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64,20),
            nn.ReLU()
        )

        #[b,20] ->  [b,784]
        self.decoder = nn.Sequential(
            nn.Linear(10,64),
            nn.ReLU(),
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,784),
            nn.Sigmoid()
        )

    def forward(self,x):
        """
        x:[B,C=1,H=28,W=28]
        B: batch size
        C: channel
        H: 矩阵的高度
        W: 矩阵的宽
        """
        batch_size = x.size(0)
        #flatten
        x  = x.view(batch_size,784)
        #ecnoder,[b,20] including mean and sigma
        h_ = self.encoder(x)
        #[b,20] -> [b,10] and [b,10]
        mu,sigma = h_.chunk(2,dim=1)
        #reparametrize trick
        h = mu + sigma * torch.randn_like(sigma)
        #decoder
        x = self.decoder(h)
        #reshape
        x = x.view(batch_size,1,28,28)

        #KL divergence
        kld = 0.5 * torch.sum(
            torch.pow(mu,2) +
            torch.pow(sigma,2) - 
            torch.log(1e-8 + torch.pow(sigma,2)) - 1
        ) /  (batch_size*28*28)

        return x,kld


