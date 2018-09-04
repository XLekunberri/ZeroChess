import chess
import chess.pgn
import numpy as np
import os
import collections
from time import gmtime, strftime
from copy import deepcopy
from collections import defaultdict


BLACK = 0
WHITE = 1

PAWN = 0
ROOK = 1
KNIGHT = 2
BISHOP = 3
QUEEN = 4
KING = 5

QUEEN_CASTLING = 0
KING_CASTLING = 1

PIECES_DICT = {
    'p': PAWN,
    'r': ROOK,
    'n': KNIGHT,
    'b': BISHOP,
    'q': QUEEN,
    'k': KING,
}


def as_fen(board):
    fen = str(board.fen).split("'")[1]
    if board is not None:
        return fen
    else:
        return None


def as_board(fen, move=None):
    board = chess.Board()
    board.set_fen(fen)
    if move is not None:
        board.push(move)
    return board


def process_nnet_output(pi, v, moves):
    pi = np.squeeze(pi, axis=0)

    compressed_pi = np.zeros(shape=len(moves))
    new_pi = np.zeros(shape=[64, 64])

    for move, i in zip(moves, range(len(moves))):
        compressed_pi[i] = pi[move.from_square, move.to_square]
        new_pi[move.from_square, move.to_square] = pi[move.from_square, move.to_square]

    compressed_pi = compressed_pi/np.sum(compressed_pi)
    new_pi = new_pi/np.sum(new_pi)

    return new_pi, v[0, 0], compressed_pi


def get_move_based_on_pi(legal_moves, pi):
    probs = np.zeros(shape=len(legal_moves))
    for move, i in zip(legal_moves, range(len(legal_moves))):
        probs[i] = pi[move.from_square, move.to_square]

    probs = probs/np.sum(probs)

    return np.random.choice(legal_moves, p=probs)


def moves_to_boards(games):
    boards = []
    for game in games:
        board = chess.Board()
        for move in game:
            if move is not None:
                board.push(move)
            boards.append(deepcopy(board))

    return boards


def get_date():
    return strftime("%Y-%m-%d %H:%M:%S", gmtime())


def save_pgn(board, white, black, n_game, name, n_iter):
    game = chess.pgn.Game()
    game.headers["Event"] = 'Test'
    game.headers["Site"] = 'Test'
    game.headers["Date"] = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    game.headers["Round"] = n_game
    game.headers["White"] = white
    game.headers["Black"] = black
    game.headers["Result"] = board.result()

    moves = [move for move in board.move_stack]
    node = game.add_variation(moves.pop(0))
    n_move = 1
    for move in moves:
        node = node.add_variation(move)
        n_move += 1

    path = 'E:\\TFG\\ZeroChess\\models\\' + name + '\\games'
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + '\\{n_iter}.pgn'.format(n_iter=n_iter), 'a') as pgn:
        pgn.write(game.accept(chess.pgn.StringExporter()))
        pgn.write('\n\n')


def check_repetitions(board):
    transposition_key = board._transposition_key()
    transpositions = collections.Counter()
    transpositions.update((transposition_key,))

    switchyard = collections.deque()
    while board.move_stack:
        move = board.pop()
        switchyard.append(move)

        if board.is_irreversible(move):
            break

        transpositions.update((board._transposition_key(),))

    while switchyard:
        board.push(switchyard.pop())

    return transpositions[transposition_key]
