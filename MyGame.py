from .MyLogic import Board
import numpy as np
import numpy.ma as ma
import math

class Game():
    """ Questa classe specifica il Game. Questo Game è pensato per
    essere: two-player, adversarial and turn-based.
    Usa 1 per il player1 e -1 per il player2."""

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        "Ritorna startBoard: una rappresentazione della board"
        # ritorna la board iniziale, di tutti gli elementi inizializzati
        # serve tenersi inizialmnte solo la board (di tipo numpy.array)
        
        self.b = Board(self.n)
        return self.b.pieces

    def getBoardSize(self):
        "Ritorna (x,y): tupla che indica la dimensione della board"
        
        return (self.n, self.n)

    def getActionSize(self):
        "Ritorna actionSize: numero di tutte le possibili azioni"
        # coppia (interrogazione, scoperta) più guess pattern si/no
        # per l'interrogazione abbiamo n^2, lo stesso vale per la
        # scoperta, quindi tutte le possibili coppie sono n^4

        return pow(self.n,4) + 2

    def getNextState(self, board, player, action):
        """
        Input: board corrente, il player corrente (1 o -1) e la move,
        l'azione compiuta dal player corrente

        Ritorna la board ed il nextPlayer, il player che gioca nel
        turno successivo (basta fare -player)
        """
        # player esegue l'azione sulla mask dell'avversario e sulle proprie info
        # è però necessario, data la action, ricavare gli indici da passare come
        # mossa, sempre nella forma (interrogazione, scoperta)
        # action must be a valid move and return (board, next player)
        
        if action < self.n**4:
            move = ([action//self.n**3, (action%self.n**3)//self.n**2],
                [(action%self.n**2)//self.n, action%self.n])
        elif action == self.n**4:
            move = "guess_pattern"
        else:
            move = "guess_nonpattern"

        self.b.execute_move(move, player)
        
        return (board, -player)

    def getValidMoves(self, board, player):
        """
        Input: board e player (1 o -1) correnti

        Ritorna validMoves: un vettore binario di lunghezza self.getActionSize(),
        con 1 nel caso la mossa è valida, 0 altrimenti
        """
        # invoca get_legal_moves e ritorna tutti i possibili casi ottenuti

        valids = [0]*self.getActionSize()

        legalMoves =  self.b.get_legal_moves(player)

        # legalMoves contiene tutte le mosse valide, si deve in seguito
        #trovare la posizione che deve occupare nel vettore binario valids
        for (i,j),(h,k) in legalMoves[:-2]:
            app = i*pow(self.n,3) + j*pow(self.n,2) + h*self.n + k
            valids[app] = 1
        valids[-1] = 1
        valids[-2] = 1

        return np.array(valids)

    def getGameEnded(self, board, player):
        """
        Input: board e player (1 o -1) correnti

        Ritorna r: 0 se game non è finito. 1 se il player vince,
        -1 se player perde,       
        """
        # verificare se la casella self.guess1 è diversa da 0
        # se self.guess1 == 1 and self.case == 1
            # player1 vince -> return 1
            # altrimenti player2 vince -> return -1
        # se self.guess1 == -1 and self.case == 0
            # player1 vince -> return 1
            # altrimenti player2 vince -> return -1
        # non è stato fatto guess -> return 0
        
        if self.b.guess1 == self.b.case:
            # vinto
            return 1
        elif self.b.guess1 == -self.b.case:
            # perso
            return -1
        
        # lo stesso vale per il player2
        if self.b.guess2 == self.b.case:
            # vinto
            return -1
        elif self.b.guess2 == -self.b.case:
            # perso
            return 1
        return 0

    def getCanonicalForm(self, board, player):
        """
        Input: board e player (1 o -1) correnti

        Ritorna la canonicalBoard: La canonical form deve essere indipendente
        dal player ed equivale alla board su cui viene applicata la mask e le
        info ottenute fino a quel punto.
        """
        
        if player == 1:
            # a self.pieces viene applicata la self.mask1 per limitare al player
            # la visione delle tessere a lui concesse, poi si applica self.info1
            # così da poter inserire anche le risposte ottenute fino ad ora

            # si ricorda inoltre che il formato delle info è:
            # board.info1[i][j][0] = bianco
            # board.info1[i][j][1] = nero
            mx = ma.masked_array(board, mask=self.b.mask1, dtype=float)
            for i in range(self.n):
                for j in range(self.n):
                    if mx.mask[i][j] == True:
                        # si delega il computo della stima legato alle informazioni
                        # ottenute alla funzione stimaValore
                        mx[i][j] = np.round(self.stimaValore(board.info1[i][j][0],
                                    board.info1[i][j][1]), 3)
        else:
            # si effettua la stessa operazione per il player2
            mx = ma.masked_array(board, mask=self.b.mask2, dtype=float)
            for i in range(self.n):
                for j in range(self.n):
                    if mx.mask[i][j] == True:
                        mx[i][j] = np.round(self.stimaValore(board.info2[i][j][0],
                                    board.info2[i][j][1]), 3)
        return mx.data
    
    def stimaValore(self, bianco, nero):
        # la funzione che stima il peso delle informazioni ottenute è
        # della forma 1-e^-((col1-col2)/1+|col1-col2|)
        # al valore così ottenutosi aggiunge 0.5 che corrisponde all'incertezza
        if nero == bianco:
            # se il loro valore è uguale allora si ha completa incertezza
            return 0.5
        elif nero > bianco:
            return 0.5 + (1-math.exp(-((nero-bianco)/(1+(np.abs(nero-bianco))))))/2
        else:
            # 1 - funzione perchè si deve ottenere l'inverso
            return 1 - (0.5 + (1-math.exp(-((bianco-nero)/(1+(np.abs(bianco-nero))))))/2)


    def getSymmetries(self, board, pi):
        """
        Input: board corrente e il vettore delle policy lungo self.getActionSize()

        Ritorna symmForms: una lista di [(board,pi)] dove ogni tupla è una forma simmetrica
        della board e del vettore pi corrispondente. Si usa quando si addestra la
        neural network attraverso gli esempi.
        """
        # mirror, rotational
        
        assert(len(pi) == self.n**4+2)  # 2 per il guess
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
        Input: board corrente

        Ritorna boardString: una conversione rapida della board in formato stringA.
        Richiesto dalla MCTS per l'hashing.
        """
        # take the canonical board, on which the mask and the info are applied
        
        return board.tostring()
