class Game():
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.

    See othello/OthelloGame.py for an example implementation.
    __init__(self): Il costruttore della classe. Può essere utilizzato per inizializzare eventuali variabili di istanza necessarie.

    getInitBoard(self): Restituisce una rappresentazione del tabellone all'inizio del gioco.
    getBoardSize(self): Restituisce le dimensioni del tabellone come una tupla di valori (x, y).
    getActionSize(self): Restituisce il numero di tutte le possibili azioni.
    getNextState(self, board, player, action): Restituisce lo stato successivo del tabellone dopo che un giocatore ha effettuato un'azione.
    getValidMoves(self, board, player): Restituisce un vettore binario che indica le mosse valide per un determinato giocatore sul tabellone corrente.
    getGameEnded(self, board, player): Restituisce lo stato del gioco, che può essere 0 se il gioco non è ancora finito, 1 se il giocatore ha vinto, -1 se il giocatore ha perso e un valore non nullo per una patta.
    getCanonicalForm(self, board, player): Restituisce la forma canonica del tabellone, che dovrebbe essere indipendente dal giocatore. 
    getSymmetries(self, board, pi): Restituisce una lista di tuple contenenti forme simmetriche del tabellone e i corrispondenti vettori di policy. Questo metodo è utile durante l'addestramento della rete neurale.
    stringRepresentation(self, board): Restituisce una rapida conversione del tabellone in un formato stringa, necessario per l'hashing utilizzato da algoritmi come MCTS (Monte Carlo Tree Search).
    """
    def __init__(self):
        pass

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        pass

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        pass

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        pass

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        pass

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
        pass

    def getGameEnded(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.
               
        """
        pass

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
        pass

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
        pass

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        pass
