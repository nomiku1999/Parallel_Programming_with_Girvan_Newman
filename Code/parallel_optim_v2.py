from numba import cuda, jit, prange
import numpy as np
import math
import sys
gName = sys.argv[1]

INF = -123456789
BLOCK_SIZE = 32
f = open(gName, "r")
numVertices = int(f.readline())
g_adj = []

# print(vertices)
for i in range(numVertices):
    g_adj.append(list(map(int, f.readline().split())))
f.close()

neighborsPerVertex = 0
for i in range(numVertices):
    neighborsPerVertex = max(neighborsPerVertex, len(g_adj[i]))

numEdges = neighborsPerVertex * numVertices

edgeArray = np.empty(numEdges, dtype=int)
edgeArray.fill(INF)
for i in range(numVertices):
    for j in range(len(g_adj[i])):
        edgeArray[i * neighborsPerVertex + j] = g_adj[i][j]

@cuda.jit
def levelAndParent_Kernel(parent, visited, que, newque, edgeArray, numVertices, neighborsPerVertex):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        if que[tIdx] == True:
            startEdges, endEdges = tIdx * \
                neighborsPerVertex, (tIdx + 1) * neighborsPerVertex

            for edge in range(startEdges, endEdges, 1):
                visitVertice = edgeArray[edge]
                # not an edge
                if visitVertice == INF:
                    break
                # only visit vertex diffent level
                if que[visitVertice] == False:
                    # visit parent
                    if visited[visitVertice] == True:
                        parent[tIdx] += parent[visitVertice]
                    # visit child
                    else:
                        newque[visitVertice] = True

@cuda.jit
def putInQueue_Kernel(level, visited, que, newque, numVertices, currentLevel):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        if que[tIdx] == True:
            level[tIdx] = currentLevel
            que[tIdx] = False
            visited[tIdx] = True

        if newque[tIdx] == True:
            que[tIdx] = True
            newque[tIdx] = False

@cuda.jit
def countBetweenness_Kernel(bet, point, parent, level, visited, que, edgeArray, numVertices, neighborsPerVertex, currentLevel):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        if que[tIdx] == True:
            startEdges, endEdges = tIdx * \
                neighborsPerVertex, (tIdx + 1) * neighborsPerVertex

            for edge in range(startEdges, endEdges, 1):
                visitVertice = edgeArray[edge]
                # not an edge
                if visitVertice == INF:
                    break
                # only visit vertex diffent level
                if que[visitVertice] == False:
                    # visit parent
                    if visited[visitVertice] == False and level[visitVertice] == currentLevel - 1:
                        updatePoint = (
                            point[tIdx] / parent[tIdx]) * parent[visitVertice]
                        cuda.atomic.add(bet, edge, updatePoint)
                        cuda.atomic.add(point, visitVertice, updatePoint)

@cuda.jit
def markVisited_Kernel(que, visited, numVertices):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        if que[tIdx] == True:
            que[tIdx] = False
            visited[tIdx] = True

@cuda.jit
def putLowerLevel_Kernel(que, level, numVertices, currentLevel):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        if level[tIdx] == currentLevel:
            que[tIdx] = True

@cuda.jit
def restoreToDefault_kernel(level, parent, visited, que, newque, startVertice, numVertices):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        level[tIdx] = INF
        parent[tIdx] = 0.0
        visited[tIdx] = False
        que[tIdx] = False
        newque[tIdx] = False
        if (tIdx == startVertice):
            level[tIdx] = 0
            parent[tIdx] = 1
            que[tIdx] = True

@cuda.jit
def restoreVisitedAndQue_kernel(visited, que, numVertices):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        visited[tIdx] = False
        que[tIdx] = False

@cuda.jit
def fillPointArrWithOnes_kernel(point, numVertices):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        point[tIdx] = 1

@cuda.jit
def setQueInMaxLevel_kernel(que, level, numVertices, maxLevel):
    tIdx = cuda.grid(1)
    if tIdx < numVertices:
        if level[tIdx] == maxLevel[0]:
            que[tIdx] = True

@cuda.jit
def findMaxVal_kernel(level, maxLevel):
    tIdx = cuda.grid(1)
    cuda.atomic.max(maxLevel, 0, level[tIdx])

bet = np.zeros(numEdges, dtype=float)
d_bet = cuda.to_device(bet)
d_edgeArray = cuda.to_device(edgeArray)

level = np.empty(numVertices, dtype=int)
parent = np.zeros(numVertices, dtype=float)
visited = np.zeros(numVertices, dtype=bool)
que = np.zeros(numVertices, dtype=bool)
newque = np.zeros(numVertices, dtype=bool)
point = np.empty(numVertices, dtype=float)

d_level = cuda.to_device(level)
d_parent = cuda.to_device(parent)
d_visited = cuda.to_device(visited)
d_que = cuda.to_device(que)
d_newque = cuda.to_device(newque)
d_point = cuda.to_device(point)

GRID_SIZE = int(math.ceil(numVertices / BLOCK_SIZE))

def bfs(start):
    # write kernel to restart level, parent, visited, que, newque to default
    # before using 5 bfs kernel below
    global d_level
    global d_parent
    global d_visited
    global d_que
    global d_newque
    global d_point

    restoreToDefault_kernel[GRID_SIZE, BLOCK_SIZE](
        d_level, d_parent, d_visited, d_que, d_newque, start, numVertices)

    currentLevel1 = 0
    while currentLevel1 < 25:  # TODO: replace 15 with the condition - reach all the node
        levelAndParent_Kernel[GRID_SIZE, BLOCK_SIZE](
            d_parent, d_visited, d_que, d_newque, d_edgeArray, numVertices, neighborsPerVertex)

        putInQueue_Kernel[GRID_SIZE, BLOCK_SIZE](
            d_level, d_visited, d_que, d_newque, numVertices, currentLevel1)

        currentLevel1 += 1

    restoreVisitedAndQue_kernel[GRID_SIZE, BLOCK_SIZE](
        d_visited, d_que, numVertices)

    maxLevel = np.zeros(1, dtype=int)
    d_maxLevel = cuda.to_device(maxLevel)
    findMaxVal_kernel[GRID_SIZE, BLOCK_SIZE](d_level, d_maxLevel)

    setQueInMaxLevel_kernel[GRID_SIZE, BLOCK_SIZE](
        d_que, d_level, numVertices, d_maxLevel)

    d_maxLevel.copy_to_host(maxLevel)
    currentLevel2 = maxLevel[0]

    fillPointArrWithOnes_kernel[GRID_SIZE, BLOCK_SIZE](d_point, numVertices)

    while currentLevel2 >= 0:
        countBetweenness_Kernel[GRID_SIZE, BLOCK_SIZE](
            d_bet, d_point, d_parent, d_level, d_visited, d_que, d_edgeArray, numVertices, neighborsPerVertex, currentLevel2)
        currentLevel2 -= 1

        markVisited_Kernel[GRID_SIZE, BLOCK_SIZE](
            d_que, d_visited, numVertices)

        putLowerLevel_Kernel[GRID_SIZE, BLOCK_SIZE](
            d_que, d_level, numVertices, currentLevel2)

for i in range(numVertices):
    bfs(i)

d_bet.copy_to_host(bet)
d_edgeArray.copy_to_host(edgeArray)

resBet = []
for i in range(numVertices):
    for j in range(neighborsPerVertex):
        id = i * neighborsPerVertex + j
        if edgeArray[id] != INF:
            if(bet[id] != 0):
                if (edgeArray[id] > i):
                    resBet.append(f"({i}, {edgeArray[id]}) {bet[id]:0.2f}")

f = open("resParallelv2_" + gName, "w")
for i in range(len(resBet)):
    f.write(resBet[i] + '\n')
f.close()
