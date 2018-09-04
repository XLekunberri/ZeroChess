from ZeroChess.neural_network import NeuralNetwork
from ZeroChess.mcts import MCTS
from copy import deepcopy
from ZeroChess.utils import as_fen, save_pgn, get_date
import os
import random
import string
import pickle
import sys
import chess
from tqdm import tqdm


class SelfPlay:
    def __init__(self, load_session=None):
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

        if load_session is not None:
            path = self.root_path + '\\models\\' + load_session + '\\self_play.pkl'
            with open(path, 'rb') as handle:
                self.n_iter = pickle.load(handle)
                self.n_games_per_iteration = pickle.load(handle)
                self.mcts_simulations_per_state = pickle.load(handle)
                self.games_to_compare = pickle.load(handle)

                # Self-Play data
                self.name = pickle.load(handle)
                self.actual_iter = pickle.load(handle)

                # Neural Network arguments
                self.nnet_args = pickle.load(handle)

                # Neural Networks
                self.current_nnet = NeuralNetwork(name='Old', args=self.nnet_args)
                self.new_nnet = NeuralNetwork(name='Trained', args=self.nnet_args)

                self.current_nnet.load_model(path=self.root_path + '\\models\\' + self.name)
                self.new_nnet.load_model(path=self.root_path + '\\models\\' + self.name)

                # MCTS arguments
                self.c_puct = pickle.load(handle)

            # MCTS
            self.mcts = None
        else:
            # Self-Play arguments
            self.n_iter = 150
            self.n_games_per_iteration = 5
            self.mcts_simulations_per_state = 20
            self.games_to_compare = 25

            # Self-Play data
            self.name = ''.join(random.choice(string.ascii_uppercase) for _ in range(4))
            self.actual_iter = 0

            # Neural Network arguments
            self.nnet_args = {
                # Model args
                'state_history': 8,
                'n_residual': 7,
                'lr': 0.02,
                'momentum': 0.7,
                'l2': 0.01,
                # Fit args
                'batch_size': 32,
                'epochs': 30,
                'validation_split': 0.3,
                'verbose': 0
            }

            # Neural Networks
            self.current_nnet = NeuralNetwork(name='Old', args=self.nnet_args)
            self.new_nnet = NeuralNetwork(name='Trained', args=self.nnet_args)

            # MCTS arguments
            self.c_puct = 0.25

            # MCTS
            self.mcts = None

    def policy_iteration(self):
        tqdm.write('\nSesion name: {}'.format(self.name))
        for i in range(self.actual_iter, self.n_iter):
            tqdm.write('Iteration {i} of {total} ({date})'.format(i=i+1, total=self.n_iter, date=get_date()))
            examples = []
            self.mcts = MCTS(self.new_nnet, c_puct=self.c_puct)
            for _ in tqdm(range(self.n_games_per_iteration), desc="MCTS Simulation", unit="sim"):
                examples.append(self.mcts_one_match())

            self.new_nnet.train(examples)

            self.compare_nnets(i)
            self.update_nnets()

            self.actual_iter += 1
            self.save_status()
            tqdm.write('')

    def compare_nnets(self, it):
        new_wins = 0
        draws = 0
        for i in tqdm(range(self.games_to_compare), desc="Comparing NNets", unit="match"):
            board = chess.Board()
            white = self.new_nnet if i % 2 == 0 else self.current_nnet
            black = self.current_nnet if i % 2 == 0 else self.new_nnet

            while True:
                move = white.make_a_move(board)
                board.push(move)

                if board.is_game_over():
                    break

                move = black.make_a_move(board)
                board.push(move)

                if board.is_game_over():
                    break

            if white.name == self.new_nnet.name and board.result() == '1-0':
                new_wins += 1
            elif black.name == self.new_nnet.name and board.result() == '0-1':
                new_wins += 1
            elif board.result() == '1/2-1/2':
                draws += 1

            save_pgn(board, white=white.name, black=black.name, n_game=i+1, name=self.name, n_iter=it+1)

        try:
            result = new_wins/(self.games_to_compare - draws)
        except ZeroDivisionError:
            result = 0

        return result

    def mcts_one_match(self):
        new_state = chess.Board()
        new_node = MCTS.Node(self.mcts, parent=None, state=new_state, move=None)
        examples = []

        while True:
            actual_state = new_state
            actual_node = new_node
            for s in range(self.mcts_simulations_per_state):
                self.mcts.search(actual_node)

            examples.append([actual_node.move, self.mcts.P[as_fen(actual_state)], None])
            best_move = actual_node.select_best_child().move

            new_state = deepcopy(actual_state)
            new_state.push(best_move)
            new_node = MCTS.Node(self.mcts, parent=actual_node, state=new_state, move=best_move)

            if new_state.is_game_over():
                examples = self.set_winner(examples, new_state.result())
                return examples

    def set_winner(self, examples, result):
        if result == '1/2-1/2':
            result = 0
        else:
            result = -1

        results = []
        for i in range(len(examples)):
            x = 1 if i % 2 == 0 else -1
            results.append(result * x)
        results = results[::-1]

        for i in range(len(examples)):
            examples[i] = [examples[i][0], examples[i][1], results[i]]

        return examples

    def update_nnets(self):
        self.new_nnet.save_model(path=self.root_path, filename='tmp')
        self.current_nnet.load_model(path=self.root_path, filename='tmp')
        os.remove(path=self.root_path + '\\tmp.json')
        os.remove(path=self.root_path + '\\tmp_weights.h5')

    def save_status(self):
        path = self.root_path + '\\models\\'

        if not os.path.exists(path + '\\' + self.name):
            os.makedirs(path + '\\' + self.name)

        with open(path + '\\' + self.name + '\\' + 'self_play.pkl', 'wb') as handle:
            pickle.dump(self.n_iter, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.n_games_per_iteration, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.mcts_simulations_per_state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.games_to_compare, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.name, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.actual_iter, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.nnet_args, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.c_puct, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.new_nnet.save_model(path=path + '\\' + self.name + '\\')

    def load_status(self):
        pass


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    sys.setrecursionlimit(10000)
    SelfPlay(load_session='FNRQ').policy_iteration()
