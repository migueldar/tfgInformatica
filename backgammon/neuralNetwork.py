import torch
from torch import nn, Tensor

class BackgammonNN(nn.Module):
    importance = 0.98

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(29, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 64)
        self.layer4 = nn.Linear(64, 64)
        self.layer5 = nn.Linear(64, 64)
        self.layerOut = nn.Linear(64, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))
        x = torch.tanh(self.layerOut(x))
        return x
    
    def loss_fn(preds: Tensor, result: int) -> Tensor:
        l = len(preds)
        realT = torch.tensor([result] * l)
        importancePowers = [BackgammonNN.importance ** i for i in range(l)]
        importancePowers.reverse()
        importanceT = torch.tensor(importancePowers)
        errors = (preds - realT) ** 2 * importanceT
        return errors.mean()
    
    def save(self, file: str):
        torch.save(self.state_dict(), file)

    def load(self, file: str):
        self.load_state_dict(torch.load(file))


NeuralNetwork: BackgammonNN = BackgammonNN()

optimizer = torch.optim.Adam(NeuralNetwork.parameters(), lr=1e-3)

#states es un tensor de tensores, donde cada tensor es una posicion
def train(states: Tensor, result: int):
    NeuralNetwork.train()

    preds = NeuralNetwork.forward(states)

    loss = BackgammonNN.loss_fn(preds, result)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
