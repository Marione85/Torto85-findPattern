from .MyLogic import Board
import numpy as np
import numpy.ma as ma
import math

class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    """
    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        # return initial board (numpy board)
        b = Board(self.n)
        return b.pieces

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        # return number of actions
        # coppia (interrogazione, scoperta) più guess pattern si/no
        return pow(self.n,4) + 2

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            move: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        # player takes action on mask of other player and his own info
        # action must be a valid move and return (board, next player)

        if action < self.n**4:
            move = ([action//self.n**3, (action%self.n**3)//self.n**2], [(action%self.n**2)//self.n, action%self.n])
        elif action == self.n**4:
            move = "guess_pattern"
        else:
            move = "guess_nonpattern"

        board.execute_move(move, player)
        
        return (board, -player)

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        # return a fixed size binary vector
        # invoca get_legal_moves e ritorna tutti i possibili casi ottenuti
        valids = [0]*self.getActionSize()

        legalMoves =  board.get_legal_moves(player)

        for (i,j),(h,k) in legalMoves[:-2]:
            app = i*pow(self.n,3) + j*pow(self.n,2) + h*self.n + k
            valids[app] = 1
        valids[-1] = 1
        valids[-2] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1

        # verificare se la casella self.guess1 è diversa da 0
        # se self.guess1 == 1 and self.case == 1
            # player1 vince -> return 1
            # altrimenti player2 vince -> return -1
        # se self.guess1 == -1 and self.case == 0
            # player1 vince -> return 1
            # altrimenti player2 vince -> return -1
        # non è stato fatto guess -> return 0
        if board.guess1 == board.case:
            # vinto
            return 1
        elif board.guess1 == -board.case:
            # perso
            return -1
        if board.guess2 == board.case:
            # vinto
            return -1
        elif board.guess2 == -board.case:
            # perso
            return 1
        return 0

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        # returns the board on which the mask and the info are applied
        
        if player == 1:
            # a self.pieces viene applicata la self.mask1 per limitare al player
            # la visione delle tessere a lui concesse, poi si applica self.info1
            # così da poter inserire anche le risposte ottenute fino ad ora

            # board.info1[i][j][0] = bianco
            # board.info1[i][j][1] = nero
            mx = ma.masked_array(board, mask=board.mask1)
            for i in range(self.n):
                for j in range(self.n):
                    if mx.mask[i][j] == True:
                        # la funzione di stima è della forma x/1+|x|
                        mx[i][j] = self.stimaValore(board.info1[i][j][0],board.info1[i][j][1])
        else:
            # si effettua la stessa cosa per il player2
            mx = ma.masked_array(board, mask=board.mask2)
            for i in range(self.n):
                for j in range(self.n):
                    if mx.mask[i][j] == True:
                        # la funzione di stima è della forma x/1+|x|
                        mx[i][j] = self.stimaValore(board.info2[i][j][0],board.info2[i][j][1])
        return mx.data
    
    def stimaValore(self, bianco, nero):
        somma = bianco + nero
        if nero == bianco:
          return 0.5
        elif nero > bianco:
          return 0.5 + (1-math.exp(-((nero-bianco)/(1+(np.abs(nero-bianco))))))/2
        else:
          return 1 - (0.5 + (1-math.exp(-((bianco-nero)/(1+(np.abs(bianco-nero))))))/2)
#---- riscrivere funzione nel commento


    # SERVE?
    def getSymmetries(self, board, pi):
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        # mirror, rotational
        assert(len(pi) == self.n**4+2)  # 2 for guess
        pi_board = np.reshape(pi[:-2], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board, player):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        # take the canonical board, on which the mask and the info are applied
        return board.tostring()
