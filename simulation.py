import random
import matplotlib.pyplot as plt

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


def dequeue(queue):
    if not queue:
        return None
    return queue.pop(0)


# Dijkstra's algorithm implementation
def dijkstras(queue):
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


# Simple visualization function
def visualize_path(nodes, a, b):
    plt.figure(figsize=(10, 10))
    plt.xlim(-1, a + 1)
    plt.ylim(-1, b + 1)
    plt.grid(True)
    plt.title("Dijkstra's Algorithm Verification")

    # Draw edges
    for i in range(a + 1):
        for j in range(b + 1):
            for edge in nodes[i][j].edges_out:
                plt.plot([i, edge.end.x], [j, edge.end.y], 'b-', alpha=0.3)

    # Draw shortest path
    path = []
    current = nodes[a][b]
    while current.edge_into is not None:
        parent = current.edge_into.start
        path.append((parent, current))
        current = parent

    for start_node, end_node in path:
        plt.plot([start_node.x, end_node.x], [start_node.y, end_node.y], 'r-', linewidth=2)

    # Mark start and end
    plt.scatter([0], [0], color='green', s=100, label='Start')
    plt.scatter([a], [b], color='red', s=100, label='End')

    plt.legend()
    plt.show()


# Main code
def main():
    # Grid dimensions
    a = 10
    b = 10

    # Create nodes
    nodes = []
    for i in range(a + 1):
        row = []
        for j in range(b + 1):
            row.append(Node(i, j))
        nodes.append(row)

    # Create edges with random weights
    for j in range(b + 1):
        for i in range(a):
            w = random.randint(1, 3)
            nodes[i][j].edges_out.append(Edge(w, nodes[i][j], nodes[i + 1][j]))
            w = random.randint(1, 3)
            nodes[i + 1][j].edges_out.append(Edge(w, nodes[i + 1][j], nodes[i][j]))

    for i in range(a + 1):
        for j in range(b):
            w = random.randint(1, 3)
            nodes[i][j].edges_out.append(Edge(w, nodes[i][j], nodes[i][j + 1]))
            w = random.randint(1, 3)
            nodes[i][j + 1].edges_out.append(Edge(w, nodes[i][j + 1], nodes[i][j]))

    # Initialize start node
    s = nodes[0][0]
    s.weight = 0

    # Initialize queue
    queue = []
    insert(queue, s)

    # Run Dijkstra's algorithm
    dijkstras(queue)

    # Print result and verify
    if nodes[a][b].weight is not None:
        print(f"Path found! Distance from (0,0) to ({a},{b}): {nodes[a][b].weight}")

        # Print path
        path = []
        current = nodes[a][b]
        while current.edge_into is not None:
            parent = current.edge_into.start
            path.append((parent.x, parent.y))
            current = parent

        path.reverse()
        print("Path:", end=" ")
        print(f"(0,0) →", end=" ")
        for x, y in path:
            print(f"({x},{y}) →", end=" ")
        print(f"({a},{b})")

        # Visualize
        visualize_path(nodes, a, b)
    else:
        print("No path found to the target!")


# Run the code
if __name__ == "__main__":
    main()