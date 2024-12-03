# imports
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# show image functions
def show_image_cv(image):
    image = cv.resize(image,(0,0),fx=0.4,fy=0.4)
    cv.imshow('image', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def show_image_plt(image, rgb=True):
    if rgb:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

# function to mask a brown-redish table and keep the blue game board
def mask_table(image):
    imgHsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    lower = np.array([85, 0, 0])
    upper = np.array([255, 255, 255])
    maskTable = cv.inRange(imgHsv, lower, upper)

    lower = np.array([0, 0, 0])
    upper = np.array([40, 255, 255])
    maskBoard = cv.inRange(imgHsv, lower, upper)
    maskBoard = cv.bitwise_not(maskBoard)
    
    finalMask = cv.bitwise_and(maskTable, maskBoard)
    maskedImage = cv.bitwise_and(image, image, mask=finalMask)
    return maskedImage

# function to sort corners of a polygon
def sort_points(points):
    points = sorted(points, key=lambda x: (x[0], x[1]))
    up = sorted(points[:2], key=lambda x: x[1])
    down = sorted(points[2:], key=lambda x: x[1])
    return np.float32([up[0], up[1], down[0], down[1]])

# function to crop image on board only
def get_board(image):
    imageMasked = mask_table(image)
    imageGrey = cv.cvtColor(imageMasked, cv.COLOR_BGR2GRAY)
    _, imageBin = cv.threshold(imageGrey, 1, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(imageBin, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv.drawContours(mask, contours, -1, 255, thickness=cv.FILLED)
    mainContour = max(contours, key=cv.contourArea)
    epsilon = 0.02 * cv.arcLength(mainContour, True)
    polygone = cv.approxPolyDP(mainContour, epsilon, True)
    sourcePoints = sort_points(np.float32([p[0] for p in polygone]))
    boardDimension = 1900
    destinationPoints = np.float32([[0, 0], [0, boardDimension-1], [boardDimension-1, 0], [boardDimension-1, boardDimension-1]])
    perspective = cv.getPerspectiveTransform(sourcePoints, destinationPoints)
    result = cv.warpPerspective(image, perspective, (boardDimension, boardDimension))
    return result[252:1652, 251:1651]

# function to get the empty board image
def get_empty_board(imagePath='auxiliary_images/empty_board.jpg'):
    image = cv.imread(imagePath)
    return get_board(image)

# function to get the cell of a freshly added piece
def find_different_cell(image, nextImage, grey=False):
    if grey:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        nextImage = cv.cvtColor(nextImage, cv.COLOR_BGR2GRAY)
    maxDiff = 0
    x, y = None, None
    for i in range(14):
        for j in range(14):
            cell = get_cell(image, i, j)
            nextCell = get_cell(nextImage, i, j)
            diff = cv.absdiff(cell, nextCell)
            diffSum = diff.sum()
            if maxDiff < diffSum:
                maxDiff = diffSum
                x, y = i, j
    return x, y, maxDiff

# function the image a cell of the board
def get_cell(boardImage, line, column, cellSize=100, showType=None):
    x = cellSize * column
    y = cellSize * line
    result = boardImage[y:y+cellSize, x:x+cellSize]
    if showType == 'cv':
        show_image_cv(result)
    elif showType == 'plt':
        show_image_plt(result)
    return result

# function to show all cells of the board
def show_cells(image):
    for i in range(0, 13, 4):
        for j in range(0, 13, 4):
            show_image_plt(image[i*100:(i+1)*100, j*100:(j+1)*100])#[5:95, 5:95])

# function to classify a number in a cell
def classify_number_in_cell(patch, sourceFolder='template_images', binariseTemplates=False, cleanPatch=True):
        patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
        patch = cv.adaptiveThreshold(patch,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,29,6)
        
        if cleanPatch:
            patch = cv.erode(patch, np.ones((3,3),np.uint8), iterations=1)
            patch = cv.dilate(patch, np.ones((3,3),np.uint8), iterations=1)
        
        files = os.listdir(sourceFolder)
        maxi = -np.inf
        best = None
        scores = []
        
        for file in files:
            imgTemplate = cv.imread(sourceFolder + '/' + file, cv.IMREAD_GRAYSCALE)

            if binariseTemplates:
                imgTemplate = cv.adaptiveThreshold(imgTemplate,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,29,6)
            
            match = cv.matchTemplate(patch, imgTemplate,  cv.TM_CCOEFF_NORMED)
            match = np.max(match)
            scores.append([file[:-4], match])
            
            if match > maxi:
                best = file[:-4]
                maxi = match

        scores.sort(key=lambda x: x[1], reverse=True)
        return best, scores

# function to initialize a virtual board
def get_empty_virtual_board():
    board = [['_' for i in range(14)] for j in range(14)]
    # bonuses
    board[0][0] = 'b 3'
    board[0][6] = 'b 3'
    board[0][7] = 'b 3'
    board[0][13] = 'b 3'
    board[6][0] = 'b 3'
    board[6][13] = 'b 3'
    board[7][0] = 'b 3'
    board[7][13] = 'b 3'
    board[13][0] = 'b 3'
    board[13][6] = 'b 3'
    board[13][7] = 'b 3'
    board[13][13] = 'b 3'
    board[1][1] = 'b 2'
    board[1][12] = 'b 2'
    board[2][2] = 'b 2'
    board[2][11] = 'b 2'
    board[3][3] = 'b 2'
    board[3][10] = 'b 2'
    board[4][4] = 'b 2'
    board[4][9] = 'b 2'
    board[9][4] = 'b 2'
    board[9][9] = 'b 2'
    board[10][3] = 'b 2'
    board[10][10] = 'b 2'
    board[11][2] = 'b 2'
    board[11][11] = 'b 2'
    board[12][1] = 'b 2'
    board[12][12] = 'b 2'
    # constrainis
    board[1][4] = '/'
    board[1][9] = '/'
    board[2][5] = '-'
    board[2][8] = '-'
    board[3][6] = '+'
    board[3][7] = 'x'
    board[4][1] = '/'
    board[4][6] = 'x'
    board[4][7] = '+'
    board[4][12] = '/'
    board[5][2] = '-'
    board[5][11] = '-'
    board[6][3] = 'x'
    board[6][4] = '+'
    board[6][9] = 'x'
    board[6][10] = '+'
    board[7][3] = '+'
    board[7][4] = 'x'
    board[7][9] = '+'
    board[7][10] = 'x'
    board[8][2] = '-'
    board[8][11] = '-'
    board[9][1] = '/'
    board[9][6] = '+'
    board[9][7] = 'x'
    board[9][12] = '/'
    board[10][6] = 'x'
    board[10][7] = '+'
    board[11][5] = '-'
    board[11][8] = '-'
    board[12][4] = '/'
    board[12][9] = '/'
    # default numbers
    board[6][6] = 1
    board[6][7] = 2
    board[7][6] = 3
    board[7][7] = 4
    return board

# numbers mapped to letters and vice versa
numToLetterDict = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G', 8: 'H', 9: 'I', 10: 'J', 11: 'K', 12: 'L', 13: 'M', 14: 'N'}
letterToNumDict = {v: k for k, v in numToLetterDict.items()}

# return the operations that can be done with the numbers a and b to get the result
def test_operation(a,b,result):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return set()
    operations = set()
    if a+b == result:
        operations.add('+')
    if a-b == result:
        operations.add('-')
    if b-a == result:
        operations.add('-')
    if a*b == result:
        operations.add('x')
    if b != 0:
        if a/b == result:
            operations.add('/')
    if a != 0:
        if b/a == result:
            operations.add('/')
    return operations

# returns if the cell can have 2 neighbors in the direction of i and the neighbor is not itself
def eligible_neighbor(i,x):
    if i == 0:
        return False
    if x+2 > 13 and i > 0:
        return False
    if x-2 < 0 and i < 0:
        return False
    return True

# returns the neighbors of the cell (2 left, 2 right, 2 up, 2 down if eligible)
def get_neighbors(x,y):
    neighbors = []
    for i in range(-2,3):
        if not eligible_neighbor(i,x):
            continue
        neighbors.append((x+i,y))
    for i in range(-2,3):
        if not eligible_neighbor(i,y):
            continue
        neighbors.append((x,y+i))
    return neighbors

# returns the number of operations succeded by the cell
def number_of_succeded_operations(virtualBoard, x, y, result):
    neighbors = get_neighbors(x,y)
    opsNumber = 0
    
    for n in range(0,len(neighbors),2):
        x1 = neighbors[n][0]
        y1 = neighbors[n][1]
        x2 = neighbors[n+1][0]
        y2 = neighbors[n+1][1]

        a = virtualBoard[x1][y1]
        b = virtualBoard[x2][y2]

        operations = test_operation(a,b,result)

        if virtualBoard[x][y] in ['/','+','-','x']:
            if virtualBoard[x][y] in operations:
                opsNumber += 1
        elif len(operations) > 0:
            opsNumber += 1

    return opsNumber

# returns all possible operations that can be done with the numbers in the neighbors of the cell
def all_possible_operations(virtualBoard, x, y, result):
    neighbors = get_neighbors(x,y)
    operations = set()
    for n in range(0,len(neighbors),2):
        x1 = neighbors[n][0]
        y1 = neighbors[n][1]
        x2 = neighbors[n+1][0]
        y2 = neighbors[n+1][1]

        a = virtualBoard[x1][y1]
        b = virtualBoard[x2][y2]

        operations = operations.union(test_operation(a,b,result))
    return operations

# returns the score of a cell
def calculate_score(virtualBoard:list[list], piece:str|int|float, x:int, y:int):
    roundScore = int(piece)
    roundScore = roundScore * number_of_succeded_operations(virtualBoard, x, y, roundScore)
    if virtualBoard[x][y] == 'b 2':
        roundScore = 2 * roundScore
    elif virtualBoard[x][y] == 'b 3':
        roundScore = 3 * roundScore
    return roundScore
