import numba
import numpy as np

f = open("graph.txt", "r")
vertices = int(f.readline())
g_pyObj = []
# print(vertices)
for i in range(vertices):
    g_pyObj.append(list(map(int, f.readline().split())))
f.close()

sizeOfG = np.zeros(vertices, dtype=int)
g = np.zeros((vertices, vertices), dtype=int)
for i in range(vertices):
    for j in range(len(g_pyObj[i])):
        g[i][j] = g_pyObj[i][j]
    sizeOfG[i] = len(g_pyObj[i])

bet = np.zeros((vertices, vertices), dtype=float)

#@numba.jit(nopython=True)
def bfs(x : int, g : np.array, bet : np.array):
    visited = np.zeros(vertices, dtype=int)
    level = np.empty(vertices, dtype=int)
    parents = np.zeros(vertices, dtype=float)
    point = np.empty(vertices, dtype=float)
    point.fill(1)
    level.fill(12345)
    q = np.zeros(vertices, dtype=int) # put/ get/ empty
    l, r = 0, 0 # left right is begin and end of queue
    q[r] = x; r += 1 # put operator
    level[x], parents[x] = 0, 1; maxlv = 0; visited[x] = 1
    while l < r:
        u = q[l]
        l += 1       # pop operator
        for vindex in range(sizeOfG[u]):
            v = g[u][vindex]
            if level[v] + 1 == level[u]: # meet your parent vertices
                parents[u] += 1
                continue
            if not visited[v]: # meet new vertices
                visited[v] = 1
                level[v] = level[u] + 1
                if maxlv < level[v]:
                    maxlv = level[v]
                q[r] = v; r += 1 # put operator

    # lv = [[] for i in range(maxlv + 1)]
    lv = np.empty((maxlv + 1, vertices), dtype=int)
    sizeOfLv = np.zeros(maxlv + 1, dtype=int)
    for i in range(vertices):
        newPosition = sizeOfLv[level[i]]
        lv[level[i]][newPosition] = i
        sizeOfLv[level[i]] += 1

    # print(lv)
    # print(sizeOfLv)
    # print(parents)

    for leafLevel in range(maxlv, 0, -1):
        for leafIndex in range(sizeOfLv[leafLevel]): #lv[lvleaf]:
            # leaf vertices if lv[leaf]
            leaf = lv[leafLevel][leafIndex]

            for connectedNodeIndex in range(sizeOfG[leaf]):
                node = g[leaf][connectedNodeIndex]                
                if level[node] + 1 == leafLevel:
                    if parents[leaf] == 0: continue
                    # print("parent, leaf: ", node, leaf)
                    # print("point parent", point[leaf], parents[leaf])
                    bet[node][leaf] += point[leaf] / parents[leaf]
                    bet[leaf][node] += point[leaf] / parents[leaf]
                    point[node] += point[leaf] / parents[leaf]
                    # print("Bet", bet[leaf][node], bet[node][leaf])
                    # print("point", point[node])

normalizationFactor = vertices ** 2 - vertices + 1 # C=(n-1)^{2}-(n-1)
resBet = []

def betweenness():
    # bfs(0, g, bet)
    for i in range(vertices):
        bfs(i, g, bet)
    for i in range(vertices):
        for j in range(vertices):
            if j < i: continue
            if bet[i][j]: 
                bet[i][j] *= (2 / (vertices * (vertices - 1)))
                bet[i][j] /= 2
                # print(f"({i}, {j}) {bet[i][j]:0.2f}")
                resBet.append(f"({i}, {j}) {bet[i][j]:0.2f}")

betweenness()
# bet
resBet