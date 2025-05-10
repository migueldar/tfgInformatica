from board import BackgammonBoard
from neuralNetwork import NeuralNetwork, train
from board import rollDices, dicesToStr, movesToGame
import torch
import math
import random
import time
import signal
import sys
from pathlib import Path
import os
import ast

calculatePossibleMovesTime = 0
elseTime = 0

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


class DiceArista:
    def __init__(self, dices):
        self.dices = dices
    def __eq__(self, other):
        if len(self.dices) == 4:
            return self.dices == other.dices
        return  (self.dices[0] == other.dices[0] and self.dices[1] == other.dices[1]) or \
                (self.dices[0] == other.dices[1] and self.dices[1] == other.dices[0])
    def __repr__(self):
        return f"Dices: {self.dices}"

#los nodos hoja siempre tienen hasDices a False
class MonteTree:
    maxTurnsSimulation = 20
    simulationsPerTurn = 500
    mu = 10e5
    alpha = 4
    # mu = 1

    def __init__(self, parent: "MonteTree", board: BackgammonBoard, hasDices: bool):
        self.board: BackgammonBoard = board
        self.parent: "MonteTree" = parent
        self.hasDices: bool = hasDices
        self.aristas: list[Arista | DiceArista] = []
        self.children: list["MonteTree"] = []


    def __repr__(self):
        ret = f"\nBoard: {self.board}\n"
        ret += f"Aristas: {self.aristas}\n"
        ret += f"Children:"
        for c in f"{self.children}".split("\n"):
            ret += f"\t{c}\n"
        return ret
    

    def strTree(self):
        ret = ""
        if self.aristas:
            ret += f"{self.aristas}\n"
        for c in self.children:
            for l in f"{c.strTree()}".split("\n"):
                ret += f"\t{l}\n"
        return ret
        

    def nodeSelection(self) -> "MonteTree":
        if self.hasDices == False:
            aristaNewDices = DiceArista(rollDices())

            for i in range(len(self.aristas)):
                if self.aristas[i] == aristaNewDices:
                    return self.children[i].nodeSelection()
                
            self.aristas.append(aristaNewDices)
            newBoard = self.board.copy()
            newBoard.dices = aristaNewDices.dices
            self.children.append(MonteTree(self, newBoard, True))

            return self.children[-1]

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


    #añade aristas e hijos, siempre se llama desde un nodo con hasDices = True
    def expansion(self, mul = 1):
        possibleMoves = self.board.calculatePossibleMoves()
        with torch.no_grad():
            possibleMovesArray = [p.toArray() for p in possibleMoves]
            for p in possibleMoves:
                self.children.append(MonteTree(self, p, False))
            #aqui multiplico por el jugador porque si es -1 los valores mejores son los mas negativos
            possibleMovesValuesT = NeuralNetwork(torch.tensor(possibleMovesArray, dtype=torch.float32)).squeeze(1) * mul * self.board.player
            possibleMovesValuesSoft = possibleMovesValuesT.softmax(-1)
            for p in possibleMovesValuesSoft.tolist():
                self.aristas.append(Arista(p))


    def simulationActionSelection(self) -> "MonteTree":
        #para hacer lo de tirar los dados etc
        self = self.nodeSelection()
        self.expansion(MonteTree.alpha)
        return chooseWithProbabilities(self.children, [a.P for a in self.aristas])


    def simulation(self) -> float:
        simTree = MonteTree(None, self.board, self.hasDices)
        moves = 0

        while simTree.board.isOver() == False and moves < MonteTree.maxTurnsSimulation:
            simTree = simTree.simulationActionSelection()
            moves += 1
        
        if simTree.board.isOver():
            return simTree.board.winner()
        with torch.no_grad():
            return NeuralNetwork(simTree.board.toTensor()).item()
        

    def backup(self, value: float):
        if self.parent == None:
            return
        if self.hasDices == True:
            return self.parent.backup(value)
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

def playGame(playerStart = None) -> list[str]:
    if playerStart == None:
        playerStart = random.choice([1,-1])

    moves = []
    mt = MonteTree(None, BackgammonBoard(), True)

    w = 0
    while not mt.board.isOver():
        # if w % 10 == 0:
        print(w)
        w += 1

        moves.append(f"{dicesToStr(mt.board.dices)}: ")

        mt = mt.playTurn()
        moves[-1] += f"{mt.board.moveStr}"
    
        if mt.board.isOver():
            break
        newDices = rollDices()
        for i in range(len(mt.aristas)):
            if mt.aristas[i].dices == newDices:
                mt = mt.children[i]
                break
        else:
            mt.board.dices = newDices
            mt = MonteTree(None, mt.board, True)

    print("Turnos:", w)
    return moves


WEIGHTSFILE = "modelWeights"
LOGDONEFILE = "/root/done/doneLogs"
GAMESDIR = "/root/logGames/"
FILEPARTIDALARGA = "/root/tfgInfo/backgammon/partida"
DONEPARTIDALARGA= "/root/done/donePartidaLarga"
DONEWEIGHTS = "/root/done/doneModelWeights"


if Path(WEIGHTSFILE).exists():
    print("loading weights")
    NeuralNetwork.load(WEIGHTSFILE)

while True:
    while not Path(LOGDONEFILE).exists():
        time.sleep(2)
    os.remove(LOGDONEFILE)
    if Path(FILEPARTIDALARGA).exists():
        os.remove(FILEPARTIDALARGA)
    if Path(DONEPARTIDALARGA).exists():
        os.remove(DONEPARTIDALARGA)

    games = []
    for i in range(1,5):
        fg = open(f"/root/logGames/log{i}", "r")

        line = "\n"
        win = None
        states = None
        while line != "":
            for i in range(3):
                line = fg.readline()
                if line == "":
                    break
                if i == 0:
                    win = int(line.removesuffix("\n"))
                elif i == 1:
                    states = ast.literal_eval(line.removesuffix("\n"))
                    games.append([states, win])


    for g in games:
        train(torch.tensor(g[0], dtype=torch.float32), g[1])
    NeuralNetwork.save(WEIGHTSFILE)
    open(DONEWEIGHTS, "w").close()

    moves = playGame()
    fp = open(FILEPARTIDALARGA, "w")
    fp.write(movesToGame(moves))
    fp.close()
    open(DONEPARTIDALARGA, "w").close()
   