import torch
from torch import nn

class AE(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        #[b,784] -> [b,20]
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
            nn.Linear(20,64),
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
        #ecnoder
        x = self.encoder(x)
        #decoder
        x = self.decoder(x)
        #reshape
        x = x.view(batch_size,1,28,28)

        return x


