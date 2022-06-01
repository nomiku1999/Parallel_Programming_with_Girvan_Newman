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
def bfs_kernel2(level, visited, que, newque, numVertices, currentLevel):
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
def bfs_kernel4(que, visited, numVertices):
    tIdx = cuda.grid(1)

    if tIdx < numVertices:
        if que[tIdx] == True:
            que[tIdx] = False
            visited[tIdx] = True


@cuda.jit
def bfs_kernel5(que, level, numVertices, currentLevel):
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
    
    GRID_SIZE = int(math.ceil(numVertices / BLOCK_SIZE))
    que[start] = True
    parent[start] = 1
    level[start] = 0
    currentLevel1 = 0

    d_parent = cuda.to_device(parent)
    d_level = cuda.to_device(level)
    d_visited = cuda.to_device(visited)
    d_que = cuda.to_device(que)
    d_newque = cuda.to_device(newque)
    d_edgeArray = cuda.to_device(edgeArray)
    while currentLevel1 < 15:
        bfs_kernel1[GRID_SIZE, BLOCK_SIZE](
            d_parent, d_visited, d_que, d_newque, d_edgeArray, numVertices, neighborsPerVertex)
        cuda.synchronize()

        bfs_kernel2[GRID_SIZE, BLOCK_SIZE](
            d_level, d_visited, d_que, d_newque, numVertices, currentLevel1)
        cuda.synchronize()

        currentLevel1 += 1

    d_level.copy_to_host(level)
    d_visited.copy_to_host(visited)
    d_que.copy_to_host(que)
    d_newque.copy_to_host(newque)

    que.fill(False)
    visited.fill(False)
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

    d_bet = cuda.to_device(bet)
    d_point = cuda.to_device(point)
    d_level = cuda.to_device(level)
    d_visited = cuda.to_device(visited)
    d_que = cuda.to_device(que)

    while currentLevel2 >= 0:
        bfs_kernel3[GRID_SIZE, BLOCK_SIZE](
            d_bet, d_point, d_parent, d_level, d_visited, d_que, d_edgeArray, numVertices, neighborsPerVertex, currentLevel2)
        cuda.synchronize()
        currentLevel2 -= 1

        bfs_kernel4[GRID_SIZE, BLOCK_SIZE](
                d_que, d_visited, numVertices)
        cuda.synchronize()


        bfs_kernel5[GRID_SIZE, BLOCK_SIZE](
                d_que, d_level, numVertices, currentLevel2)
        cuda.synchronize()

    d_bet.copy_to_host(bet)
    d_point.copy_to_host(point)
    d_parent.copy_to_host(parent)
    d_level.copy_to_host(level)
    d_visited.copy_to_host(visited)
    d_que.copy_to_host(que)
    d_edgeArray.copy_to_host(edgeArray)

for i in range(numVertices):
    bfs(i)

resBet = []
for i in range(numVertices):
    for j in range(neighborsPerVertex):
        id = i * neighborsPerVertex + j
        if edgeArray[id] != INF:
            if(bet[id] != 0):
                resBet.append(f"({i}, {edgeArray[id]}) {bet[id]:0.2f}")
print(resBet)
