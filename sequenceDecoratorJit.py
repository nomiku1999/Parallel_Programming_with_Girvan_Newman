import numba
import numpy as np

INF = 123456789

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


def bfs(start, g, bet):
    visited = np.zeros(vertices, dtype=int)
    level = np.empty(vertices, dtype=int)
    level.fill(INF)
    parents = np.zeros(vertices, dtype=float)
    point = np.empty(vertices, dtype=float)
    point.fill(1)
    q = np.zeros(vertices, dtype=int)

    bfs1(q, g, level, parents, visited, start)

    lv = np.empty((15, vertices), dtype=int)
    sizeOfLv = np.zeros(15, dtype=int)
    bfs2(g, lv, sizeOfLv, level, parents, point, bet)


@numba.jit(nopython=True)
def bfs1(q, g, level, parents, visited, start):
    l, r = 0, 0  # left right is begin and end of queue
    q[r] = start
    r += 1  # put operator

    level[start], parents[start] = 0, 1
    visited[start] = 1
    while l < r:
        u = q[l]
        l += 1       # pop operator
        for vindex in range(sizeOfG[u]):
            v = g[u][vindex]
            if level[v] + 1 == level[u]:  # meet your parent vertices
                parents[u] += parents[v]
                continue
            if not visited[v]:  # meet new vertices
                visited[v] = 1
                level[v] = level[u] + 1
                q[r] = v
                r += 1  # put operator


@numba.jit(nopython=True)
def bfs2(g, lv, sizeOfLv, level, parents, point, bet):

    maxlv = 0
    for i in range(vertices):
        if level[i] != INF:
            newPosition = sizeOfLv[level[i]]
            lv[level[i]][newPosition] = i
            sizeOfLv[level[i]] += 1
            if maxlv < level[i]:
                maxlv = level[i]

    for leafLevel in range(maxlv, 0, -1):
        for leafIndex in range(sizeOfLv[leafLevel]):  # lv[lvleaf]:
            # leaf vertices if lv[leaf]
            leaf = lv[leafLevel][leafIndex]

            for connectedNodeIndex in range(sizeOfG[leaf]):
                node = g[leaf][connectedNodeIndex]
                if level[node] + 1 == leafLevel:
                    if parents[leaf] == 0:
                        continue
                    # print("parent, leaf: ", node, leaf)
                    # print("point parent", point[leaf], parents[leaf])
                    gainPoint = point[leaf] / parents[leaf] * parents[node]
                    bet[node][leaf] += gainPoint
                    bet[leaf][node] += gainPoint
                    point[node] += gainPoint
                    # print("Bet", bet[leaf][node], bet[node][leaf])
                    # print("point", point[node])


resBet = []


def betweenness():
    # bfs(0, g, bet)
    for i in range(vertices):
        bfs(i, g, bet)
    for i in range(vertices):
        for j in range(vertices):
            if j < i:
                continue
            if bet[i][j]:
                bet[i][j] /= 2
                resBet.append(f"({i}, {j}) {bet[i][j]:0.2f}")


betweenness()
# bet
print(resBet)
