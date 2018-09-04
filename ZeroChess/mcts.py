import math
from collections import defaultdict
from ZeroChess.utils import as_fen, as_board, process_nnet_output


class MCTS:
    def __init__(self, nnet, c_puct=0):
        self.nodes = {}
        self.nnet = nnet
        self.c_puct = c_puct

        self.Q = defaultdict(lambda: 0)
        self.N = defaultdict(lambda: 0)
        self.P = defaultdict(lambda: 0)

    def add_node(self, node):
        self.nodes[(node.parent, node.state, node.move)] = node
        for move in node.unexplored_children:
            node.add_child(move)
            self.Q[(node, move)] = 0
            self.N[(node, move)] = 0

    def search(self, node):
        # If the game is over, return the result
        if as_board(node.state).is_game_over():
            if as_board(node.state).result() == '1/2-1/2':
                return 0
            else:
                return -1

        # If this is the new state, evaluate with the neural network and update (EXPANSION)
        if node not in self.nodes.values():
            self.add_node(node)

            pi, v = self.nnet.predict(as_board(node.state))

            legal_moves = list(as_board(node.state).legal_moves)
            pi, v, compressed_pi = process_nnet_output(pi, v, legal_moves)

            self.P[node.state] = pi
            for move, i in zip(legal_moves, range(len(legal_moves))):
                self.P[(node.state, move)] = compressed_pi[i]

            return -v

        # SELECTION
        best_child = node.select_best_child()
        best_move = best_child.move

        # SIMULATION
        v = self.search(best_child)

        # BACKPROPAGATION
        actual_q = self.Q[(node.state, best_move)]
        actual_n = self.N[(node.state, best_move)]

        self.Q[(node.state, best_move)] = (actual_n * actual_q + v) / (actual_n + 1)
        self.N[(node.state, best_move)] += 1

        return -v

    class Node:
        def __init__(self, mcts_instance, parent=None, state=None, move=None):
            self.MCTS = mcts_instance

            self.parent = parent
            self.state = as_fen(state)
            self.move = move
            self.children = []
            self.unexplored_children = list(state.legal_moves)

        def get_ucb(self, state, move):
            q = self.MCTS.Q[(state, move)]
            p = self.MCTS.P[(state, move)] * math.sqrt(sum(self.MCTS.N.values())) / (1 + self.MCTS.N[(state, move)])

            return q + self.MCTS.c_puct * p
        
        def select_best_child(self):
            return sorted(self.children, key=lambda child: self.get_ucb(self.state, child.move))[0]

        def add_child(self, child_move):
            child_state = as_board(self.state, move=child_move)
            child = MCTS.Node(mcts_instance=self.MCTS, parent=self, state=child_state, move=child_move)
            self.children.append(child)
            self.unexplored_children.remove(child_move)
