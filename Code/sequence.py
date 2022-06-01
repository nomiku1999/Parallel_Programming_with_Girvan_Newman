from queue import Queue

f = open("graph.txt", "r")
vertices = int(f.readline())
g = []
# print(vertices)
for i in range(vertices):
    g.append(list(map(int, f.readline().split())))
f.close()

# noEdges = 0
# for i in range(len(g)):
#     noEdges += len(g[i])
# print(noEdges / 2)
bet = [[0 for i in range(vertices)] for i in range(vertices)]

def bfs(x : int):
    visited = [False for i in range(vertices)] # marked visited
    level = [123456789 for i in range(vertices)] # level of each vertices
    parents = [0 for i in range(vertices)] # number of parent lv = lv-1
    point = [1 for i in range(vertices)] # point of each vertices
    q = Queue() # put/ get/ empty
    q.put(x)
    level[x], parents[x] = 0, 1; maxlv = 0; visited[x] = True
    while not q.empty():
        u = q.get()
        for v in g[u]:
            if level[v] + 1 == level[u]: # meet your parent vertices
                parents[u] += parents[v]
                continue
            if not visited[v]: # meet new vertices
                visited[v] = True
                level[v] = level[u] + 1
                maxlv = max(maxlv, level[v])
                q.put(v)

    lv = [[] for i in range(maxlv + 1)]
    for i in range(vertices):
        idx = int(level[i])
        if idx != 123456789:
            lv[idx].append(i)
    
    for lvleaf in range(maxlv, 0, -1):
        for leaf in lv[lvleaf]:
            for parent in g[leaf]:                
                if level[parent] + 1 == lvleaf:
                    bet[parent][leaf] += point[leaf] / parents[leaf] * parents[parent]
                    bet[leaf][parent] += point[leaf] / parents[leaf] * parents[parent]
                    point[parent] += point[leaf] / parents[leaf] * parents[parent]

resBet = []

def betweenness():
    for i in range(vertices):
        bfs(i)
        
    for i in range(vertices):
        for j in range(vertices):
            if j < i: continue
            if bet[i][j]: 
                # bet[i][j] *= (2 / (vertices * (vertices - 1)))
                bet[i][j] /= 2
                bet[j][i] = bet[i][j]
                # print(f"({i}, {j}) {bet[i][j]:0.2f}")
                resBet.append(f"({i}, {j}) {bet[i][j]:0.2f}")

betweenness()
# bet
print(resBet)