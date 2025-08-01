import torch
import torch.nn.functional as F
from torch import nn, optim

class TemperatureScaling:
    def __init__(self):
        self.temperature = torch.nn.Parameter(torch.ones(1))

    def fit(self, logits, labels):
        # 使用交叉熵损失来优化温度参数
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        # optimizer = torch.optim.AdamW([self.temperature], lr=0.01)

        def loss_fn():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(loss_fn)
    
    def predict(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)
