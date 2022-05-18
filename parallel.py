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
                        parent[tIdx] += 1
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


level = np.zeros(numVertices, dtype=int)
parent = np.zeros(numVertices, dtype=int)
visited = np.zeros(numVertices, dtype=bool)
que = np.zeros(numVertices, dtype=bool)
newque = np.zeros(numVertices, dtype=bool)
GRID_SIZE = int(math.ceil(numVertices / BLOCK_SIZE))
que[0] = True

currentLevel = 1
while currentLevel < 10:
    bfs_kernel1[GRID_SIZE, BLOCK_SIZE](
        parent, visited, que, newque, edgeArray, numVertices, neighborsPerVertex)
    cuda.synchronize()
    bfs_kernel2[GRID_SIZE, BLOCK_SIZE](
        level, parent, visited, que, newque, edgeArray, numVertices, neighborsPerVertex, currentLevel)
    cuda.synchronize()
    currentLevel += 1
    print(np.count_nonzero(visited))
