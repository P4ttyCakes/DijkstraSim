
class Node:
    def __init__(self, x, y):
        self.edges_out = []
        self.next = None
        self.edge_into = None
        self.weight = None
        self.x = x
        self.y = y

class Edge:
    def __init__(self, weight, start, end):
        self.weight = weight
        self.start = start
        self.end = end

def dequeue(queue):
    if not queue:
        return None
    return queue.pop(0)

def remove(queue, node):
    if node in queue:
        queue.remove(node)

def insert(queue, node):
    if not queue:
        queue.append(node)
        return
    for i in range(len(queue)):
        if node.weight <= queue[i].weight:
            queue.insert(i, node)
            return
    queue.append(node)

def djikstras(queue):
    while queue:
        n = dequeue(queue)
        for edge in n.edges_out:
            k = edge.end
            if k.weight is None:
                k.weight = n.weight + edge.weight
                k.edge_into = edge
                insert(queue, k)
            elif k.weight > n.weight + edge.weight:
                k.weight = n.weight + edge.weight
                k.edge_into = edge
                remove(queue, k)
                insert(queue, k)
