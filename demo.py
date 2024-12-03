#%%
import mathableLib as ml
import cv2 as cv



def run_demo(sourceFolder:str='training_data'):
    emptyBoard = ml.get_empty_board()

    for game in range(1, 5):
        print(f'Game {game}:')
        board = emptyBoard
        
        with open(f'training_data/{game}_turns.txt', 'r') as file:
            players = []
            rounds = []
            for line in file:
                players.append(line.strip().split()[0])
                rounds.append(int(line.strip().split()[1]))

        score = 0
        virtualBoard = ml.get_empty_virtual_board()
        print(f'{players[0]} turn:')
        for round in range(1, 51):

            if round in rounds[1:]:
                print(f'{players[rounds.index(round)-1]} obtained {score} points')
                print(f'{players[rounds.index(round)]} turn:')
                score = 0

            nextImage = cv.imread(f'{sourceFolder}/{game}_{round:02}.jpg')
            nextBoard = ml.get_board(nextImage)

            x, y, _ = ml.find_different_cell(board, nextBoard)

            guess = str(x+1) + str(ml.numToLetterDict[y+1])
            cell = ml.get_cell(nextBoard, x, y, showType=None)

            pred, scores = ml.classify_number_in_cell(cell)
            
            score += ml.calculate_score(virtualBoard, pred, x, y)
            virtualBoard[x][y] = int(pred)
            
            guess += ' ' + pred.strip()
            print(guess)
            board = nextBoard

        print(f'{players[-1]} obtained {score} points')
        print('End of game\n')



if __name__ == '__main__':
    run_demo()