from numba import cuda, jit, prange
import numpy as np
import math

INF = 123456789
BLOCK_SIZE = 32
f = open("graph.txt", "r")
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
def bfs_kernel1(parent, visited, que, newque, edgeArray, numVertices, neighborsPerVertex):
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
def bfs_kernel2(level, parent, visited, que, newque, edgeArray, numVertices, neighborsPerVertex, currentLevel):
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
def bfs_kernel3(bet, point, parent, level, visited, que, edgeArray, numVertices, neighborsPerVertex, currentLevel):
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
                        updatePoint = (point[tIdx] / parent[tIdx]) * parent[visitVertice]
                        cuda.atomic.add(bet, edge, updatePoint)
                        cuda.atomic.add(point, visitVertice, updatePoint)


@cuda.jit
def bfs_kernel4(que, visited, edgeArray, numVertices):
    tIdx = cuda.grid(1)

    if tIdx < numVertices:
        if que[tIdx] == True:
            que[tIdx] = False
            visited[tIdx] = True


@cuda.jit
def bfs_kernel5(que, level, edgeArray, numVertices, currentLevel):
    tIdx = cuda.grid(1)

    if tIdx < numVertices:
        if level[tIdx] == currentLevel:
            que[tIdx] = True

bet = np.zeros(numEdges, dtype=float)


def bfs(start):
    level = np.empty(numVertices, dtype=int); level.fill(INF)
    parent = np.zeros(numVertices, dtype=float)
    visited = np.zeros(numVertices, dtype=bool)
    que = np.zeros(numVertices, dtype=bool)
    newque = np.zeros(numVertices, dtype=bool)
    point = np.zeros(numVertices, dtype=float)
    point.fill(1)
    GRID_SIZE = int(math.ceil(numVertices / BLOCK_SIZE))
    que[start] = True
    parent[start] = 1
    level[start] = 0

    currentLevel1 = 0
    while currentLevel1 < 15:
        bfs_kernel1[GRID_SIZE, BLOCK_SIZE](
            parent, visited, que, newque, edgeArray, numVertices, neighborsPerVertex)
        cuda.synchronize()
        bfs_kernel2[GRID_SIZE, BLOCK_SIZE](
            level, parent, visited, que, newque, edgeArray, numVertices, neighborsPerVertex, currentLevel1)
        currentLevel1 += 1

    que = np.zeros(numVertices, dtype=bool)
    visited = np.zeros(numVertices, dtype=bool)
    point = np.empty(numVertices, dtype=float)
    point.fill(1)

    maxLevel = 0
    for i in range(numVertices):
        if level[i] != INF:
            maxLevel = max(maxLevel, level[i])

    for i in range(numVertices):
        if level[i] == maxLevel:
            que[i] = True

    currentLevel2 = maxLevel

    while currentLevel2 >= 0:
        bfs_kernel3[GRID_SIZE, BLOCK_SIZE](
            bet, point, parent, level, visited, que, edgeArray, numVertices, neighborsPerVertex, currentLevel2)
        cuda.synchronize()
        currentLevel2 -= 1
        bfs_kernel4[GRID_SIZE, BLOCK_SIZE](
                que, visited, edgeArray, numVertices)
        cuda.synchronize()
        bfs_kernel5[GRID_SIZE, BLOCK_SIZE](
                que, level, edgeArray, numVertices, currentLevel2)
        cuda.synchronize()

for i in range(numVertices):
    bfs(i)

resBet = []
for i in range(numVertices):
    for j in range(neighborsPerVertex):
        id = i * neighborsPerVertex + j
        if edgeArray[id] != INF:
            if(bet[id] != 0):
                resBet.append(f"({i}, {edgeArray[id]}) {bet[id]:0.2f}")
print(resBet[:50])
