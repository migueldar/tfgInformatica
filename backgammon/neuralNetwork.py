import torch
from torch import nn, Tensor

class BackgammonNN(nn.Module):
    importance = 0.98

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(29, 15, False)
        self.layer2 = nn.Linear(15, 15, False)
        self.layer3 = nn.Linear(15, 15, False)
        self.layer4 = nn.Linear(15, 15, False)
        self.layer5 = nn.Linear(15, 15, False)
        self.layer6 = nn.Linear(15, 15, False)
        self.layer7 = nn.Linear(15, 15, False)
        self.layer8 = nn.Linear(15, 15, False)
        self.layerOut = nn.Linear(15, 1, False)

    def forward(self, x: Tensor) -> Tensor:
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        x = torch.tanh(self.layer4(x))
        x = torch.tanh(self.layer5(x))
        x = torch.tanh(self.layer6(x))
        x = torch.tanh(self.layer7(x))
        x = torch.tanh(self.layer8(x))
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
NeuralNetwork.to(device)

optimizer = torch.optim.Adam(NeuralNetwork.parameters(), lr=1e-3)

#states es un tensor de tensores, donde cada tensor es una posicion
def train(states: Tensor, result: int):
    NeuralNetwork.train()

    preds = NeuralNetwork.forward(states)

    loss = BackgammonNN.loss_fn(preds, result)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()
