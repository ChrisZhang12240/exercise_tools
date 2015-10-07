import random

class UndirectedGraphNode:
    def __init__(self, val):
        self.val = val
        self.adj = []

    def __cmp__(self, other):
        if other == None:
            return -1

        if self.val < other.val:
            return -1
        elif self.val == other.val:
            return 0
        return 1

    def addNeighbor(self, node):
        if node not in self.adj:
            self.adj.append(node)





class UndirectedGraph:
    def __init__(self):
        self.vCollection = {}

    def addEdge(self, v, w):
        if not self.vCollection.has_key(v):
            self.vCollection[v] = UndirectedGraphNode(v)
        if not self.vCollection.has_key(w):
            self.vCollection[w] = UndirectedGraphNode(w)

        nodeV = self.vCollection[v]
        nodeW = self.vCollection[w]
        nodeV.addNeighbor(nodeW)
        nodeW.addNeighbor(nodeV)


    def getNode(self, idx):
        if self.vCollection.has_key(idx):
            return self.vCollection[idx]
        return None



def strToGraph(s):
    lst = s.split('#')
    g = UndirectedGraph()

    for item in lst:
        lst = [int(val) for val in item.split(',')]
        v = lst.pop(0)
        for n in lst:
            g.addEdge(v, n)

    return g

def grapthicToStr(g):
    ret = []
    for node in sorted(g.vCollection.values()):
        lst = ["%d" % node.val]
        for neighbor in sorted(node.adj):
            lst.append("%d" % neighbor.val)
        ret.append(",".join(lst))

    return "#".join(ret)

def nodeToStr(node):
    marked = {}
    queue = [node]
    ret = []
    while queue:
        item = queue.pop(0)
        temp = [item.val]
        if marked.has_key(item.val):
            continue
        marked[item.val] = 1

        for neighbor in item.adj:
            if not marked.has_key(neighbor.val):
                queue.append(neighbor)
            temp.append(neighbor.val)
        ret.append(temp)

    ret = sorted(ret)
    for i in xrange(len(ret)):
        item = ret.pop(0)
        ret.append(','.join(map(str, item)))

    return "#".join(ret)


def equal(node1, node2):
    def dfs(node, ret, marked):
        if marked.has_key(node.val):
            return

        marked[node.val] = True
        ret.append(node)
        for neighbor in sorted(node.adj):
            if not marked.has_key(neighbor.val):
                dfs(neighbor, ret, marked)

        return ret

    collection1, collection2 = [], []
    dfs(node1, collection1, {})
    dfs(node2, collection2, {})

    if len(collection1) != len(collection2):
        return False

    while collection1:
        node1 = collection1.pop()
        node2 = collection2.pop()
        if node1.val != node2.val:
            return False

        if sorted(node1.adj) != sorted(node2.adj):
            return False


    return True

def generateRandomGraph(cnt):
    g = UndirectedGraph()

    inGraph = []
    notIn = [x for x in xrange(cnt)]

    v = notIn.pop(0)
    while notIn:
        w = notIn.pop(random.randint(0, len(notIn)-1))
        if random.randint(0, 3) == 1:
            notIn.append(w)

        g.addEdge(v, w)
        inGraph.append(v)
        inGraph.append(w)
        v = inGraph[random.randint(0, len(inGraph)-1)]

    return g











def test():
    def valid1():
        g = generateRandomGraph(50 + random.randint(-15, 15))
        s = ""
        temp = set()
        for i in xrange(25):
            if not i % 2:
                s = grapthicToStr(g)
                temp.add(s)
            else:
                g = strToGraph(s)

        temp = list(temp)
        if len(temp) != 1 or nodeToStr(g.getNode(1)) != temp[0]:
            raise RuntimeError("test graphic failed!!")


    def valid2():
        g = generateRandomGraph(50 + random.randint(-15, 15))
        s = grapthicToStr(g)
        g2 = strToGraph(s)

        if not equal(g.getNode(0), g2.getNode(0)):
            raise RuntimeError("test graphic failed!!")

    for i in xrange(50):
        valid1()
        valid2()

    print "test graphic done!"


#test()