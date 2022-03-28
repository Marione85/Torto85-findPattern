import numpy as np
import numpy.ma as ma
import itertools

class Board():

    def __init__(self, n):
        "Configurazione iniziale della board"

        self.n = n

        # si tiene traccia del guess di uno dei due player
        # 0 scelta non presa
        # 1 se sceglie pattern
        # -1 se sceglie non pattern
        self.guess1 = 0
        self.guess2 = 0

        # si creano le due maschere per suddividere la board
        # 1 visibile, 0 non visibile
        self.mask2 = np.zeros(self.n*self.n, dtype=int)
        self.mask2[:(self.n*self.n)//2] = 1
        np.random.shuffle(self.mask2)
        self.mask2 = np.reshape(self.mask2, (self.n,self.n))

        self.mask1 = np.where(self.mask2==1, 0, 1)
        
        # parte relativa alla gestione delle informazioni
        # in posizione (i,j,0) abbiamo il numero di bianchi ottenuti come risposte interrogando la casella (i,j)
        # in posizione (i,j,1) abbiamo il numero di neri ottenuti come risposte interrogando la casella (i,j)
        self.info1 = np.zeros((self.n,self.n,2), dtype=int)
        self.info2 = np.zeros((self.n,self.n,2), dtype=int)

        # il numero di caselle nere è legato alla dimensione della board
        self.n_black = int(np.ceil(self.n*self.n*0.22))

        # Creazione della board iniziale
        # 0 white, 1 black
        self.pieces = np.zeros(self.n*self.n, dtype=int)
        self.pieces[:self.n_black] = 1
        np.random.shuffle(self.pieces)
        self.pieces = np.reshape(self.pieces, (self.n,self.n))
        
        # Si tiene traccia anche della presenza, o meno, del pattern
        # 0 non pattern, 1 pattern
        self.case = self.existPattern()

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def existPattern(self):
        "Gestisce i due possibili casi di pattern analizzati"
        
        if(self.n < 6):
            return self.checkPattern2()
        else:
            return self.checkPattern4()
    
    def checkPattern2(self):
        # pattern di almeno due celle adiacenti
        
        for i in range(self.n):
            for j in range(self.n):
                if j < self.n - 1:
                    if self.pieces[i][j] == 1 and self.pieces[i][j+1] == 1:
                        return 1
                if i < self.n - 1:
                    if self.pieces[i][j] == 1 and self.pieces[i+1][j] == 1:
                        return 1
        return 0

    def checkPattern4(self):
        # pattern con forma di tetramino a S o Z
        
        for i in range(self.n):
            for j in range(self.n):
                if j<self.n-2 and i<self.n-1:
                    if (self.pieces[i+1][j] == 1 and self.pieces[i][j+1] == 1 and
                        self.pieces[i+1][j+1] == 1 and self.pieces[i][j+2] == 1) or (
                        self.pieces[i][j] == 1 and self.pieces[i][j+1] == 1 and
                        self.pieces[i+1][j+1] == 1 and self.pieces[i+1][j+2] == 1):
                            return 1
                elif j<self.n-1 and i<self.n-2:
                    if (self.pieces[i+1][j] == 1 and self.pieces[i+2][j] == 1 and
                        self.pieces[i][j+1] == 1 and self.pieces[i+1][j+1] == 1) or (
                        self.pieces[i][j] == 1 and self.pieces[i+1][j] == 1 and
                        self.pieces[i+1][j+1] == 1 and self.pieces[i+2][j+1] == 1):
                            return 1       
        return 0

    def countDiff(self, player):
        """Conta il numero di celle nere che il player vede
        (1 per player1, -1 per player2)"""

        # le maschere servono per non tenere in considerazione le celle
        # ancora invisibili al rispettivo player
        mx1 = ma.masked_array(self.pieces, mask=self.mask1)
        mx2 = ma.masked_array(self.pieces, mask=self.mask2)

        if player == 1:
            return mx1.sum() - mx2.sum()
        else:
            return mx2.sum() - mx1.sum()

    def get_legal_moves(self, player):
        """Ritorna tutte le mosse legali per il player
        (0 per white, 1 per black)
        """
        
        # Prende tutte le coppie di mosse per il player.
        if player == 1:
            # il primo elemento della coppia risultante è quello da scoprire
            # il secondo elemento della coppia risultante è quello da interrogare
            return list(itertools.product(np.argwhere(self.mask2 == 1).tolist(),np.argwhere(self.mask1 == 1).tolist())) + ["guess_pattern", "guess_nonpattern"]
        else:
            # il primo elemento della coppia risultante è quello da scoprire
            # il secondo elemento della coppia risultante è quello da interrogare
            return list(itertools.product(np.argwhere(self.mask1 == 1).tolist(),np.argwhere(self.mask2 == 1).tolist())) + ["guess_pattern", "guess_nonpattern"]

    def execute_move(self, move, player):
        """Esegue le due mosse sulla "board"; in realtà modifica esclusivamente
        le maschere e le informazioni, a seconda del player. (0=white,1=black)
        """
        
        if player == 1:

            # move può contenere il guess oppure nell'ordine: gli indici
            # della cella da scoprire e poi quelli della cella da interrogare

            if move == "guess_pattern":
                self.guess1 = 1
            elif move == "guess_nonpattern":
                self.guess1 = -1
            else:
                # aggiornamento della maschera dell'avversario
                i,j = move[0]
                self.mask2[i][j] = 0
                
                # interrogazione con risposta random
                # 70% corretta, 30% altrimenti
                h,k = move[1]
                col = self.pieces[h][k]
                if np.random.uniform(0,1) <= 0.7 :
                    self.info1[h][k][col] += 1
                else:
                    if col == 1:
                        self.info1[h][k][0] += 1
                    else:
                        self.info1[h][k][1] += 1

        else:

            if move == "guess_pattern":
                self.guess2 = 1
            elif move == "guess_nonpattern":
                self.guess2 = -1
            else:
                # aggiornamento della maschera dell'avversario
                i,j = move[0]
                self.mask1[i][j] = 0

                # interrogazione con risposta random
                # 70% corretta, 30% altrimenti
                h,k = move[1]
                col = self.pieces[h][k]
                if np.random.uniform(0,1) <= 0.7 :
                    self.info2[h][k][col] += 1
                else:
                    if col == 1:
                        self.info2[h][k][0] += 1
                    else:
                        self.info2[h][k][1] += 1
