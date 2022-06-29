import numba
import numpy as np
import sys
gName = sys.argv[1]

INF = 123456789

f = open(gName, "r")
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

    visited = np.zeros(vertices, dtype=int)
    # lv = np.empty((vertices + 1, vertices), dtype=int)
    # sizeOfLv = np.zeros(vertices + 1, dtype=int)
    bfs2(q, g, level, parents, point, bet, visited)


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
def bfs2(q, g, level, parents, point, bet, visited):
    maxlv = 0
    for i in range(vertices):
        if level[i] != INF:
            if maxlv < level[i]:
                maxlv = level[i]
    cnt = 0
    while cnt < 15 and maxlv > 0:
        l, r = 0, 0
        for i in range(vertices):
            if level[i] == maxlv and not visited[i]:
                q[r] = i
                visited[i] = 1
                r += 1

        while l < r:
            leaf = q[l]
            l += 1       # pop operator
            for vindex in range(sizeOfG[leaf]):
                parent = g[leaf][vindex]
                if level[parent] + 1 == level[leaf]:  # meet your parent vertices
                    gainPoint = (point[leaf] / parents[leaf]) * parents[parent]
                    bet[parent][leaf] += gainPoint
                    bet[leaf][parent] += gainPoint
                    point[parent] += gainPoint

                    # updatePoint = (point[tIdx] / parent[tIdx]) * parent[visitVertice]
                    # cuda.atomic.add(bet, edge, updatePoint)
                    # cuda.atomic.add(point, visitVertice, updatePoint)
        maxlv -= 1


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
                bet[j][i] = bet[i][j]
                resBet.append(f"({i}, {j}) {bet[i][j]:0.2f}")


betweenness()
# bet
f = open("resSeqJit_" + gName, "w")
for i in range(len(resBet)):
    f.write(resBet[i] + '\n')
f.close()
