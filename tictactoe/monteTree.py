from board import TicTacToeBoard
from neuralNetworkMulti import NeuralNetwork, train
# from neuralNetwork import NeuralNetwork, train
import torch
import math
import random

def chooseWithProbabilities(options: list, probabilities: list[float]):
    ran = random.random()
    curr = 0

    for i in range(len(options)):
        curr += probabilities[i]
        if ran < curr:
            return options[i]
    return options[-1]

class Arista:
    exploreTreeConst = 10

    def __init__(self, P: float):
        self.N = 0
        self.Q = 0
        self.P = P

    def __repr__(self):
        return f"N: {self.N}, Q: {self.Q:.2f}, P: {self.P:.2f}"
    
    def treePolicy(self, n: int, player: int):
        return player * self.Q + Arista.exploreTreeConst * self.P * math.sqrt(math.log(n + 1) / (self.N + 1))
    
    def addValue(self, val: float):
        self.Q = (self.Q * self.N + val) / (self.N + 1)
        self.N += 1

    def printAristas(aristas: list["Arista"]):
        for a in aristas:
            print(a)


class MonteTree:
    maxTurnsSimulation = 9
    exploreActionConst = 1
    simulationsPerTurn = 1000
    mu = 10e5
    # mu = 1
    alpha = 4

    def __init__(self, parent: "MonteTree", board: TicTacToeBoard):
        self.board: TicTacToeBoard = board
        self.parent: "MonteTree" = parent
        self.aristas: list[Arista] = []
        self.children: list["MonteTree"] = []

    def __repr__(self):
        ret = f"\nBoard: {self.board}\n"
        ret += f"Aristas: {self.aristas}\n"
        ret += f"Children:"
        for c in f"{self.children}".split("\n"):
            ret += f"\t{c}\n"
        return ret
    

    def strTree(self):
        ret = f"{self.board}\n"
        ret += f"{sum(a.N for a in self.aristas)}\n"
        for c in self.children:
            for l in f"{c.strTree()}".split("\n"):
                ret += f"\t{l}\n"
        return ret
        

    def nodeSelection(self) -> "MonteTree":
        if len(self.children) == 0:
            return self

        sumN = sum(a.N for a in self.aristas)

        #little hack, so that if none of the children have been played
        #the one with the highest P will be chosen
        if sumN == 0:
            sumN = 1

        argmax, valmax = 0, self.aristas[0].treePolicy(sumN, self.board.player)

        for i in range(1, len(self.aristas)):
            if self.aristas[i].N == 0:
                return self.children[i].nodeSelection()
            valaux = self.aristas[i].treePolicy(sumN, self.board.player)
            if valaux > valmax:
                argmax = i
                valmax = valaux

        return self.children[argmax].nodeSelection()


    #añade aristas e hijos
    def expansion(self, mul = 1):
        possibleMoves = self.board.calculatePosibleMoves()
        with torch.inference_mode():
            possibleMovesValues = []
            for p in possibleMoves:
                possibleMovesValues.append(NeuralNetwork(p.toTensor()))
                self.children.append(MonteTree(self, p))
            #aqui multiplico por el jugador porque si es -1 los valores mejores son los mas negativos
            possibleMovesValuesT = torch.tensor(possibleMovesValues) * mul * self.board.player
            possibleMovesValuesSoft = possibleMovesValuesT.softmax(-1)
            for p in possibleMovesValuesSoft:
                self.aristas.append(Arista(p.item()))


    #para las acciones en la simulacion seleccionamos con probabilidad P
    def simulationActionSelection(self) -> "MonteTree":
        self.expansion(MonteTree.alpha)
        return chooseWithProbabilities(self.children, [a.P for a in self.aristas])


    def simulation(self) -> float:
        simTree = MonteTree(None, self.board)
        moves = 0

        while simTree.board.isOver() == False and moves < MonteTree.maxTurnsSimulation:
            simTree = simTree.simulationActionSelection()
            moves += 1
        
        if simTree.board.isOver():
            return simTree.board.winner()
        NeuralNetwork.eval()
        return NeuralNetwork(simTree.board.toTensor()).item()
        

    def backup(self, value: float):
        if self.parent == None:
            return
        #busco la arista que me conecta con mi padre
        for i in range(len(self.parent.children)):
            if self.parent.children[i].board == self.board:
                self.parent.aristas[i].addValue(value)
                break
        return self.parent.backup(value)


    def actionSelection(self) -> "MonteTree":
        qs = [MonteTree.mu * c.Q for c in self.aristas]
        qsT = torch.tensor(qs) * self.board.player
        qsSoft = qsT.softmax(-1)

        # Arista.printAristas(self.aristas)
        # print([f"{x:.3f}" for x in qs])
        # print(qsSoft)

        return chooseWithProbabilities(self.children, qsSoft)


    def playTurn(self) -> "MonteTree":
        for _ in range(MonteTree.simulationsPerTurn):
            selected = self.nodeSelection()
            selected.expansion()
            if len(selected.children) != 0:
                selected = selected.nodeSelection()
            simVal = selected.simulation()
            selected.backup(simVal)
        return self.actionSelection()


def playGame(playerStart = None, board = [0,0,0,0,0,0,0,0,0,0]) -> tuple[torch.Tensor, int]:
    if playerStart == None:
        playerStart = random.choice([1,-1])

    states = []
    mt = MonteTree(None, TicTacToeBoard(playerStart, board))

    while not mt.board.isOver():
        # print(mt.board)
        mt = mt.playTurn()
        states.append(mt.board.board + [mt.board.player])
    # print(mt.board)

    return torch.tensor(states, dtype=torch.float32), mt.board.winner()


def playSelf():
    board = [0] * 9
    r = int(random.random() * 9) % 10
    board[r] = random.choice([1, -1])
    _, p = playGame(board[r] * -1, board)
    print(p, flush=True)


def playRandom():
    tc = TicTacToeBoard(1)
    mt = MonteTree(None, tc)
    monTurn = False

    while not tc.isOver():
        if monTurn:
            mt = mt.playTurn()
            tc = mt.board
        else:
            tc = random.choice(tc.calculatePosibleMoves())
            for child in mt.children:
                if child.board == tc:
                    mt = child
                    break
            else:
                mt = MonteTree(None, tc)
        monTurn = not monTurn
    print(tc.winner(), flush=True)


def playMe():
    tc = TicTacToeBoard(1)
    mt = MonteTree(None, tc)
    monTurn = False

    while not tc.isOver():
        if monTurn:
            mt = mt.playTurn()
            tc = mt.board
        else:
            tc = tc.calculatePosibleMoves()[int(input("Movimiento: "))]
            for child in mt.children:
                if child.board == tc:
                    mt = child
                    break
            else:
                mt = MonteTree(None, tc)
        monTurn = not monTurn
        print(tc)


NeuralNetwork.load("modelWeights")

# boards = [
#         [-1,-1, 1, 0, 1, 0, 0, 0, 0],
#         [-1, 1,-1,-1, 1, 0, 0, 0, 1],
#         [ 1,-1,-1, 0, 0, 0, 0, 0, 1],

#         [-1, 1, 1, 0,-1, 0, 0, 0, 0],
#         [-1, 1, 0, 0,-1,-1, 1,-1, 1],
#         [-1, 1, 0,-1, 1, 0, 0,-1, 0],

#         [ 1,-1, 1, 0, 0,-1, 0, 0,-1],
#         [ 1,-1,-1, 0, 0, 1, 0, 0, 0],
#         [ 0, 0, 1, 0, 0,-1,-1, 1,-1],

#         [ 0, 0, 0,-1, 1, 0, 0, 0, 0],
#         [ 0, 0, 1, 0, 0, 0,-1, 0, 0],
#         [ 0, 0, 1, 0, 0, 0, 0,-1, 0],
# ]

# i = 0
# for b in boards:
#     print(i)
#     print(MonteTree(None, TicTacToeBoard(1, b)).playTurn().board)
#     i += 1
#     print(flush=True)

# for i in range(1, 5001):
#     print(i)
#     st, res = playGame()
#     # print(r, s)
#     train(st, res)

# NeuralNetwork.save("modelWeights")