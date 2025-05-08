import random
import torch

class TicTacToeBoard:
	#0 empty 1 player -1 other player
	def __init__(self, player, board: list[int] = [0,0,0,0,0,0,0,0,0]):
		self.board = board
		self.player = player

	def __repr__(self):
		ret = "\n"
		for i in range(3):
			ret += "["
			for j in range(3):
				if self.board[i * 3 + j] == 1:
					ret += " 1"
				elif self.board[i * 3 + j] == 0:
					ret += " 0"
				else:
					ret += f"{self.board[i * 3 + j]}"
				if j != 2:
					ret += ", "
			ret += "]"
			if i != 2:
				ret += "\n"
		ret += f", {self.player}"
		return ret
	
	def __eq__(self, other):
		return self.board == other.board and self.player == other.player
	
	# def putPiece(self, piece: int, pos: int):
	# 	if self.board[pos] != 0:
	# 		raise RuntimeError("This tile already has a piece")
	# 	self.board[pos] = piece
	
	# def putPieceRandom(self):
	# 	free = self.freeSlots()
	# 	if free == 0:
	# 		raise RuntimeError("Board is full")
	# 	pos = random.randint(0, free - 1)
	# 	for i in range(len(self.board)):
	# 		if self.board[i] == 0:
	# 			if pos == 0:
	# 				self.board[i] = self.player
	# 				break
	# 			pos -= 1

	# def freeSlots(self) -> int:
	# 	free = 0
	# 	for v in self.board:
	# 		if v == 0:
	# 			free += 1
	# 	return free
	
	def winner(self) -> int:
		for p in [1,-1]:
			for i in range(3):
				if self.board[i*3] == p and self.board[i*3+1] == p and self.board[i*3+2] == p:
					return p
				if self.board[i] == p and self.board[3+i] == p and self.board[6+i] == p:
					return p
			if self.board[0] == p and self.board[4] == p and self.board[8] == p:
				return p
			if self.board[2] == p and self.board[4] == p and self.board[6] == p:
				return p
		return 0

	def isFull(self) -> bool:
		for i in range(9):
			if self.board[i] == 0:
				return False
		return True
	
	def isOver(self) -> bool:
		return self.isFull() or self.winner() != 0
	
	def toTensor(self) -> torch.Tensor:
		return torch.tensor(self.board + [self.player], dtype=torch.float32)
	

	def calculatePosibleMoves(self) -> list["TicTacToeBoard"]:
		ret = []
		if self.isOver():
			return ret
		for i in range(len(self.board)):
			if self.board[i] == 0:
				auxBoard = self.board.copy()
				auxBoard[i] = self.player
				ret.append(TicTacToeBoard(self.player*-1, auxBoard))
		return ret

# t = TicTacToeBoard(1)
# print(t.calculatePosibleMoves()[0].calculatePosibleMoves())
