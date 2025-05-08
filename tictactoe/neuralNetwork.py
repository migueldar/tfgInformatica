import torch
from torch import nn, Tensor

class TicTacToeNN(nn.Module):
    importance = 0.9

    def __init__(self):
        super().__init__() 
        # self.weights = nn.Parameter(torch.randn(10))
        self.weights = nn.Parameter(torch.tensor([0.5736, 0.2790, 0.5403, 0.4037, 0.7111, 0.2733, 0.4696, 0.3405, 0.5313, 0.1369]))

    def forward(self, x: Tensor) -> Tensor:
        return torch.tanh((self.weights * x).sum(dim=-1))
    
    def loss_fn(preds: Tensor, result: int) -> Tensor:
        l = len(preds)
        realT = torch.tensor([result] * l)
        importancePowers = [TicTacToeNN.importance ** i for i in range(l)]
        importancePowers.reverse()
        importanceT = torch.tensor(importancePowers)
        errors = (preds - realT) ** 2 * importanceT
        return errors.mean()
    
    def save(self, file: str):
        torch.save(self.state_dict(), file)

    def load(self, file: str):
        self.load_state_dict(torch.load(file))


NeuralNetwork: TicTacToeNN = TicTacToeNN()
optimizer = torch.optim.SGD(NeuralNetwork.parameters(), lr=0.02)

#states es un tensor de tensores, donde cada tensor es una posicion
def train(states: Tensor, result: int):
    NeuralNetwork.train()

    preds = NeuralNetwork.forward(states)

    loss = TicTacToeNN.loss_fn(preds, result)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
