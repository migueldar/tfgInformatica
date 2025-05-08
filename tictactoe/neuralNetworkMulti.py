import torch
from torch import nn, Tensor

class TicTacToeNN(nn.Module):
    importance = 0.9

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 6, False)
        self.layer2 = nn.Linear(6, 6, False)
        self.layer3 = nn.Linear(6, 6, False)
        self.layer4 = nn.Linear(6, 6, False)
        self.layerOut = nn.Linear(6, 1, False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layerOut(x))
        return x
    
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

optimizer = torch.optim.Adam(NeuralNetwork.parameters(), lr=1e-3)

#states es un tensor de tensores, donde cada tensor es una posicion
def train(states: Tensor, result: int):
    NeuralNetwork.train()

    preds = NeuralNetwork.forward(states)

    loss = TicTacToeNN.loss_fn(preds, result)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
