import keras
import math
import numpy as np
import ZeroChess.utils as utils
from copy import deepcopy
from keras.layers import Input, Dense, Concatenate, Add, Activation, BatchNormalization, Conv3D, Flatten, Reshape, Multiply
from keras.models import Model
from ZeroChess.utils import check_repetitions, moves_to_boards, get_move_based_on_pi
from keras_tqdm import TQDMCallback


class NeuralNetwork:
    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.model = None

    def new_model(self):
        def add_residual_block(input_model):
            model = Conv3D(filters=256, kernel_size=3, strides=1, data_format='channels_first', padding='same')(input_model)
            model = BatchNormalization(axis=1)(model)
            model = Activation('relu')(model)
            model = Conv3D(filters=256, kernel_size=3, strides=1, data_format='channels_first', padding='same')(model)
            model = BatchNormalization(axis=1)(model)
            model = Add()([model, input_model])
            model = Activation('relu')(model)

            return model

        p1_pieces = Input(shape=(self.args['state_history'], 6, 8, 8), dtype='float32', name='p1_pieces')
        p2_pieces = Input(shape=(self.args['state_history'], 6, 8, 8), dtype='float32', name='p2_pieces')
        helpers = Input(shape=(self.args['state_history'], 1, 8, 8), dtype='float32', name='helpers')
        legal_moves = Input(shape=(64, 64), dtype='float32', name='legal_moves')

        inputs = Concatenate(axis=2)([p1_pieces, p2_pieces, helpers])

        model = Conv3D(filters=256, kernel_size=3, strides=1, data_format='channels_first', padding='same')(inputs)
        model = BatchNormalization(axis=1)(model)
        model = Activation('relu')(model)

        for i in range(self.args['n_residual']):
            model = add_residual_block(model)

        pi_output = Conv3D(filters=2, kernel_size=1, strides=1, data_format='channels_first', padding='same')(model)
        pi_output = BatchNormalization(axis=1)(pi_output)
        pi_output = Activation('relu')(pi_output)
        pi_output = Flatten(data_format='channels_first')(pi_output)
        pi_output = Dense(units=4096, activation='softmax')(pi_output)
        pi_output = Reshape(target_shape=(64, 64), name='pi_output')(pi_output)
        # pi_output = Multiply(name='pi_output')([pi_output, legal_moves])

        v_output = Conv3D(filters=1, kernel_size=1, strides=1, data_format='channels_first', padding='same')(model)
        v_output = BatchNormalization(axis=1)(v_output)
        v_output = Activation('relu')(v_output)
        v_output = Dense(units=256)(v_output)
        v_output = Activation('relu')(v_output)
        v_output = Flatten(data_format='channels_first')(v_output)
        v_output = Dense(units=1, activation='tanh', name='v_output')(v_output)

        self.model = Model(inputs=[p1_pieces, p2_pieces, helpers, legal_moves], outputs=[pi_output, v_output])

    def save_model(self, path, filename='neural_network'):
        model_json = self.model.to_json()
        with open('{}\\{}.json'.format(path, filename), 'w') as json_file:
            json_file.write(model_json)

        self.model.save_weights('{}\\{}_weights.h5'.format(path, filename))

    def load_model(self, path, filename='neural_network'):
        with open('{}\\{}.json'.format(path, filename), 'r') as json_file:
            model_json = json_file.read()

        self.model = keras.models.model_from_json(model_json)
        self.model.load_weights('{}\\{}_weights.h5'.format(path, filename))
        self.compile_model()

    def compile_model(self):
        def pi_loss_function(pi_true, pi_pred):
            return keras.backend.sum(-pi_true * keras.backend.log(pi_pred), axis=-1)

        def v_loss_function(v_true, v_pred):
            return keras.losses.mean_squared_error(v_true, v_pred)

        optimizer = keras.optimizers.SGD(lr=self.args['lr'], momentum=self.args['momentum'])
        losses = {'pi_output': pi_loss_function, 'v_output': v_loss_function}

        self.model.compile(optimizer=optimizer, loss=losses, metrics=['accuracy'])

    def train(self, targets):
        if self.model is None:
            self.new_model()
            self.compile_model()

        moves = [[targets[x][i][0] for i in range(len(targets[0]))] for x in range(len(targets))]
        boards = moves_to_boards(moves)
        nninputs = [NNInput(board, self.args['state_history']).as_input() for board in boards]

        input_games = {
            'p1_pieces': np.asarray([game['p1_pieces'] for game in nninputs]),
            'p2_pieces': np.asarray([game['p2_pieces'] for game in nninputs]),
            'helpers': np.asarray([game['helpers'] for game in nninputs]),
            'legal_moves': np.asarray([game['legal_moves'] for game in nninputs])
        }

        target_outputs = {
            'pi_output': np.asarray([targets[x][i][1] for i in range(len(targets[0])) for x in range(len(targets))]),
            'v_output': np.asarray([targets[x][i][2] for i in range(len(targets[0])) for x in range(len(targets))])
        }

        self.model.fit(x=input_games,
                       y=target_outputs,
                       batch_size=self.args['batch_size'],
                       epochs=self.args['epochs'],
                       validation_split=self.args['validation_split'],
                       verbose=self.args['verbose'],
                       callbacks=[TQDMCallback()]
                       )

    def predict(self, board):
        if self.model is None:
            self.new_model()
            self.compile_model()

        input_array = NNInput(board, self.args['state_history']).as_input(expand=True)
        pi, v = self.model.predict(input_array)

        return pi, v

    def make_a_move(self, board):
        pi, _ = self.predict(board)

        legal_moves = list(board.legal_moves)
        return get_move_based_on_pi(legal_moves, np.squeeze(pi, axis=0))


class NNInput:
    def __init__(self, board, t):
        self.original_board = board

        white_board, black_board, helpers = self.process_board(deepcopy(board), t, [], [], [])

        self.p1_pieces = white_board if int(board.turn) == utils.WHITE else black_board
        self.p2_pieces = black_board if int(board.turn) == utils.WHITE else white_board
        self.helpers = helpers
        self.legal_moves = self.get_legal_moves(list(board.legal_moves))

    def as_input(self, expand=False):
        if expand:
            return {
                'p1_pieces': np.expand_dims(self.p1_pieces, axis=0),
                'p2_pieces': np.expand_dims(self.p2_pieces, axis=0),
                'helpers': np.expand_dims(self.helpers, axis=0),
                'legal_moves': np.expand_dims(self.legal_moves, axis=0)
            }
        else:
            return {
                'p1_pieces': self.p1_pieces,
                'p2_pieces': self.p2_pieces,
                'helpers': self.helpers,
                'legal_moves': self.legal_moves
            }

    def process_board(self, board, t, white_stack, black_stack, helpers_stack):

        white_board, black_board = self.get_color_boards(board)
        helpers = self.get_helpers(board)

        white_stack.append(white_board)
        black_stack.append(black_board)
        helpers_stack.append(helpers)

        if t > 1:
            if board.move_stack:
                board.pop()
            self.process_board(board, t - 1, white_stack, black_stack, helpers_stack)

        return np.asarray(white_stack), np.asarray(black_stack), np.asarray(helpers_stack)

    def get_color_boards(self, board):
        board_string = str(board).replace(' ', '').replace('\n', '')
        white_board = np.zeros((6, 8, 8), dtype='float32')
        black_board = np.zeros((6, 8, 8), dtype='float32')

        for piece, pos in zip(board_string, range(0, len(board_string) + 1)):
            if piece.lower() in utils.PIECES_DICT:
                x = math.floor(pos / 8)
                y = pos % 8

                if piece.islower():
                    black_board[utils.PIECES_DICT[piece]][x][y] = 1
                else:
                    white_board[utils.PIECES_DICT[piece.lower()]][x][y] = 1

        return np.asarray(white_board), np.asarray(black_board)

    def get_helpers(self, board):
        helpers = np.zeros(shape=(1, 8, 8), dtype='float32')

        p1_castling = np.zeros(2, dtype='float32')
        p1_castling[utils.QUEEN_CASTLING] = int(board.has_queenside_castling_rights(self.original_board.turn))
        p1_castling[utils.KING_CASTLING] = int(board.has_kingside_castling_rights(self.original_board.turn))

        p2_castling = np.zeros(2, dtype='float32')
        p2_castling[utils.QUEEN_CASTLING] = int(board.has_queenside_castling_rights(not self.original_board.turn))
        p2_castling[utils.KING_CASTLING] = int(board.has_kingside_castling_rights(not self.original_board.turn))

        color = int(self.original_board.turn)
        move_count = board.fullmove_number
        no_progress = board.halfmove_clock
        repetitions = check_repetitions(board)

        np.put(helpers[0], [0, 1], p1_castling)
        np.put(helpers[0], [2, 3], p2_castling)
        np.put(helpers[0], 4, color)
        np.put(helpers[0], 5, move_count)
        np.put(helpers[0], 6, no_progress)
        np.put(helpers[0], 7, repetitions)

        return np.asarray(helpers)

    def get_legal_moves(self, moves):
        matrix = np.zeros(shape=(64, 64), dtype='float32')
        for move in moves:
            matrix[move.from_square, move.to_square] = 1

        return np.asarray(matrix)
