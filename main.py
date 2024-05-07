import logging

import coloredlogs

from Coach import Coach
from MyGame import MyGame as Game
from MyNN import NNetWrapper as nn
from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 10,           #numero di iterazioni totali previste
    'numEps': 100,              #Numero di partite complete di auto-gioco da simulare durante una nuova iterazione
    'tempThreshold': 15,        #soglia di partite da vincere
    'updateThreshold': 0.6,     # Durante i playoff dell'arena, la nuova rete neurale sarà accettata se vince più della soglia specificata di partite.
    'maxlenOfQueue': 2000,    # Numero massimo di esempi di gioco per addestrare le reti neurali.
    'numMCTSSims': 3,           # Numero di simulazioni di MCTS (Monte Carlo Tree Search) per mossa di gioco.
    'arenaCompare': 40,         # Numero di partite da giocare durante il confronto dell'arena per determinare se la nuova rete sarà accettata.
    'cpuct': 1,                 #Parametro per il calcolo dell'upper confidence bound nell'algoritmo MCTS.

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20, #Numero di iterazioni per la cronologia degli esempi di addestramento.

})
    #Carica la classe del gioco (Game) e istanzia un oggetto di questo tipo (g) con un parametro che sarà la n per costituire la board nxn.
    #Carica la rete neurale (nn) e istanzia un oggetto di questo tipo (nnet), passando l'oggetto del gioco come argomento.
    #Se la variabile args.load_model è impostata su True, carica un modello di rete neurale preaddestrato da un checkpoint specificato.
    #Se la variabile args.load_model è impostata su True, carica anche gli esempi di allenamento (trainExamples) dal file.
    #Crea un oggetto della classe Coach, passando l'oggetto del gioco, l'oggetto della rete neurale e gli argomenti (args).
    #Avvia il processo di apprendimento chiamando il metodo learn() sull'oggetto del Coach.

def main():

    # print("rinnovato")

    log.info('Loading %s...', Game.__name__)
    g = Game(3)

    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()


if __name__ == "__main__":
    main()
