import random
import torch
import time

def rollDice() -> int:
    return random.randint(1,6)

def rollDices() -> list[int]:
    dices = [rollDice(), rollDice()]
    if dices[0] == dices[1]:
        dices = [dices[0]] * 4
    return dices

def dicesToStr(dices) -> str:
    if dices[0] == dices[1]:
        return f"{dices[0]}{dices[0]}"
    if dices[0] > dices[1]:
        return f"{dices[0]}{dices[1]}"
    return f"{dices[1]}{dices[0]}"

def movesToGame(moves: list[str]) -> str:
    ret =  "1 point match\n\n"
    ret += "Game 1\n"
    ret += "ai1:0 ai2:0\n"

    for i in range(len(moves) // 2):
        ret += f"{i+1}) {moves[i * 2]} {moves[i * 2 + 1]}\n"
    if len(moves) % 2 == 1:
        ret += f"{len(moves) // 2 + 1}) {moves[-1]}\n"
    return ret


class BackgammonBoard:
    def __init__(self, points = [-2, 0, 0, 0, 0, 5, 0, 3, 0, 0, 0, -5, 5, 0, 0, 0, -3, 0, -5, 0, 0, 0, 0, 2],
                 bar = [0,0], bearoff = [0,0], dices = None, player = None):
        self.points = points
        self.bar = bar
        self.bearoff = bearoff
        if dices:
            self.dices = dices
        else:
            self.dices = [rollDice(), rollDice()]
            while self.dices[0] == self.dices[1]:
                self.dices = [rollDice(), rollDice()]
        if player:
            self.player = player
        else:
            self.player = 1 if self.dices[0] > self.dices[1] else -1
        self.moveStr = ""

    def __repr__(self):
        return f"\nP:\t{self.points[0:6]}\n\t{self.points[6:12]}\n\t{self.points[12:18]}\n\t{self.points[18:24]}\n" + \
                f"Bar: {self.bar}\nBearoff: {self.bearoff}\nDices: {self.dices}\nTurn: {self.player}\nMoveStr: {self.moveStr}"

    def copy(self):
        return BackgammonBoard(self.points.copy(), self.bar.copy(), self.bearoff.copy(), self.dices.copy(), self.player)

    #these 2 methods are defined for the set to work well
    #hash doesnt have to be perfect bc both __hash__ and __eq__ are checked before adding to set
    #however the more precise the hash the better, as the set will be more efficient
    def __hash__(self):
        sumBarBear = self.bar[0] + self.bar[1] * 16 + self.bearoff[0] * (16 ** 2) + self.bearoff[1] * (16 ** 3)
        ret = 0
        for i in range(24):
            ret += self.points[i] * (8 ** i)
        ret += sumBarBear * (8 ** 25)
        return ret

    def __eq__(self, other):
        return self.points == other.points and self.bar == other.bar and self.bearoff == other.bearoff

    def rollDices(self):
        self.dices = [rollDice(), rollDice()]
        if self.dices[0] == self.dices[1]:
            self.dices = [self.dices[0]] * 4

    #returns possible moves for current player, and writtes them in self.possibleMoves
    #if different die, will calculate die1 -> die2 and die2 -> die1
    #If only some possibilities allow max moves they are forced
    def calculatePossibleMoves(self) -> list["BackgammonBoard"]:
        if not self.dices:
            raise RuntimeError("Cant calculate moves if dices not rolled")

        ret: set["BackgammonBoard"] = set()
        if len(self.dices) == 2:
            validMoves1 = BackgammonBoard.validMoves(self, self.dices[0], True)
            if validMoves1:
                for v in validMoves1:
                    validMoves2 = BackgammonBoard.validMoves(v, self.dices[1], False)
                    if validMoves2:
                        ret.update(validMoves2)
            validMoves3 = BackgammonBoard.validMoves(self, self.dices[1], True)
            if validMoves3:
                for v in validMoves3:
                    validMoves4 = BackgammonBoard.validMoves(v, self.dices[0], False)
                    if validMoves4:
                        ret.update(validMoves4)
            if not ret:
                ret.update(validMoves1)
                ret.update(validMoves3)

        if len(self.dices) == 4:
            validMoves1 = BackgammonBoard.validMoves(self, self.dices[0], True)
            if validMoves1:
                for v1 in validMoves1:
                    validMoves2 = BackgammonBoard.validMoves(v1, self.dices[0], False)
                    if validMoves2:
                        for v2 in validMoves2:
                            validMoves3 = BackgammonBoard.validMoves(v2, self.dices[0], False)
                            if validMoves3:
                                for v3 in validMoves3:
                                    validMoves4 = BackgammonBoard.validMoves(v3, self.dices[0], False)
                                    if validMoves4:
                                        ret.update(validMoves4)
                                    else:
                                        ret.update(validMoves3)
                            else:
                                ret.update(validMoves2)
                    else:
                        ret.update(validMoves1)

        if not ret:
            ret.add(self.copy())

        for p in ret:
            p.player = self.player * -1
            p.dices = []
        return list(ret)

    @staticmethod
    def validMoves(game: "BackgammonBoard", die: int, isFirst: bool) -> list["BackgammonBoard"]:
        if game.player == 1:
            if game.bar[0] > 0:
                if game.points[-die] >= -1:
                    gameCp = game.copy()
                    gameCp.bar[0] -= 1
                        
                    gameCp.moveStr = f"25/{25 - die}" if isFirst else f"{game.moveStr} 25/{25 - die}"

                    if game.points[-die] == -1:
                        gameCp.points[-die] = 1
                        gameCp.bar[1] += 1

                        gameCp.moveStr += "*"
                    else:
                        gameCp.points[-die] += 1
                    return [gameCp]
                return []
        
            ret = []
            hasPiece = [i for i in range(len(game.points)) if game.points[i] > 0]
            if not hasPiece:
                return []
            isBearing = True if max(hasPiece) < 6 else False

            for i in hasPiece:
                if i - die >= 0 and game.points[i - die] >= -1:
                    gameCp = game.copy()
                    gameCp.points[i] -= 1

                    gameCp.moveStr = f"{i + 1}/{i + 1 - die}" if isFirst else f"{game.moveStr} {i + 1}/{i + 1 - die}"

                    if game.points[i - die] == -1:
                        gameCp.points[i - die] = 1
                        gameCp.bar[1] += 1

                        gameCp.moveStr += "*"
                    else:
                        gameCp.points[i - die] += 1
                    ret.append(gameCp)
                elif isBearing and (i - die == -1 or (i - die < -1 and i == hasPiece[len(hasPiece) - 1] and not ret)):
                    gameCp = game.copy()
                    gameCp.points[i] -= 1
                    gameCp.bearoff[0] += 1

                    gameCp.moveStr = f"{i + 1}/0" if isFirst else f"{game.moveStr} {i + 1}/0"

                    ret.append(gameCp)

        else:
            if game.bar[1] > 0:
                if game.points[die - 1] <= 1:
                    gameCp = game.copy()
                    gameCp.bar[1] -= 1

                    gameCp.moveStr = f"25/{25 - die}" if isFirst else f"{game.moveStr} 25/{25 - die}"

                    if game.points[die - 1] == 1:
                        gameCp.points[die - 1] = -1
                        gameCp.bar[0] += 1

                        gameCp.moveStr += "*"
                    else:
                        gameCp.points[die - 1] -= 1
                    return [gameCp]
                return []
            
            ret = []
            hasPiece = [i for i in range(len(game.points)) if game.points[i] < 0]
            if not hasPiece:
                return []
            hasPiece.reverse()
            isBearing = True if min(hasPiece) >= 18 else False

            for i in hasPiece:
                if i + die < 24 and game.points[i + die] <= 1:
                    gameCp = game.copy()
                    gameCp.points[i] += 1

                    gameCp.moveStr = f"{24 - i}/{24 - i - die}" if isFirst else f"{game.moveStr} {24 - i}/{24 - i - die}"
                
                    if game.points[i + die] == 1:
                        gameCp.points[i + die] = -1
                        gameCp.bar[0] += 1

                        gameCp.moveStr += "*"
                    else:
                        gameCp.points[i + die] -= 1
                    ret.append(gameCp)
                elif isBearing and (i + die == 24 or (i - die > 24 and i == hasPiece[len(hasPiece) - 1] and not ret)):
                    gameCp = game.copy()
                    gameCp.points[i] += 1
                    gameCp.bearoff[1] += 1

                    gameCp.moveStr = f"{24 - i}/0" if isFirst else f"{game.moveStr} {24 - i}/0"

                    ret.append(gameCp)
        return ret

    def diff(self, post):
        diffpoints = []
        diffbar = self.bar != post.bar
        diffbearoff = self.bearoff != post.bearoff
        for i in range(24):
            if self.points[i] != post.points[i]:
                diffpoints.append((i, self.points[i], post.points[i]))
        ret = f"points: {diffpoints}"
        if diffbar:
            ret += f", bar: {post.bar}"
        if diffbearoff:
            ret += f", bearoff: {post.bearoff}"
        return ret    

    def winner(self):
        if self.bearoff[0] == 15:
            return 1
        if self.bearoff[1] == 15:
            return -1
        return 0
    
    def isOver(self):
        return self.winner() != 0
    
    def toArray(self) -> list[int]:
        return self.points + self.bar + self.bearoff + [self.player]
    
    def toTensor(self) -> torch.Tensor:
        return torch.tensor(self.points + self.bar + self.bearoff + [self.player], dtype=torch.float32)

# dados = input("Dices: ")
# moves = game.calculatePossibleMoves()
# for i in range(len(moves)):
#     print(i, ":", game.diff(moves[i]))
# move = int(input("Move: "))
# game.playMove(move)

# board = BackgammonBoard()

# print(board.calculatePossibleMoves())

# for i in range(100):
#     while game.winner() == 0:
#         moves = game.calculatePossibleMoves()
#         if len(moves) == 0:
#             game.passMove()
#         else:
#             game.playMove(0)
#         game.rollDices()
#     # print(game)


# print(media / 100)

