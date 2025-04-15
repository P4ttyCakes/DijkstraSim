import streamlit as st
import random
from simulation import Node, Edge, insert, djikstras
from streamlit_agraph import agraph, Node as ANode, Edge as AEdge, Config

# Sidebar controls
st.sidebar.title("Graph Settings")
a = st.sidebar.slider("Grid Width (a)", 2, 10, 5)
b = st.sidebar.slider("Grid Height (b)", 2, 10, 5)
min_w = st.sidebar.slider("Min Edge Weight", 1, 10, 1)
max_w = st.sidebar.slider("Max Edge Weight", min_w, 20, 5)

# Main layout and dynamic width
col = st.columns([1])
container_width = st.session_state.get("container_width", 800)
container_height = 150 * (b + 1)

# Button to start algorithm
run_dijkstra = st.button("Start Dijkstra's Algorithm")

# Create nodes function
def create_nodes(a, b):
    nodes = []
    for i in range(a + 1):
        row = []
        for j in range(b + 1):
            row.append(Node(i, j))
        nodes.append(row)
    return nodes

# Create edges function
def create_edges(nodes, a, b, min_w, max_w):
    for j in range(b + 1):
        for i in range(a):
            w1 = random.randint(min_w, max_w)
            nodes[i][j].edges_out.append(Edge(w1, nodes[i][j], nodes[i + 1][j]))
            w2 = random.randint(min_w, max_w)
            nodes[i + 1][j].edges_out.append(Edge(w2, nodes[i + 1][j], nodes[i][j]))

    for i in range(a + 1):
        for j in range(b):
            w3 = random.randint(min_w, max_w)
            nodes[i][j].edges_out.append(Edge(w3, nodes[i][j], nodes[i][j + 1]))
            w4 = random.randint(min_w, max_w)
            nodes[i][j + 1].edges_out.append(Edge(w4, nodes[i][j + 1], nodes[i][j]))

    return nodes

# Prepare graph for AGraph
agraph_nodes = []
agraph_edges = []
node_ids = {}

# Helper set to track added edge pairs
edge_pairs = set()

nodes = create_nodes(a, b)  # Create nodes
nodes = create_edges(nodes, a, b, min_w, max_w)  # Create edges

for i in range(a + 1):
    for j in range(b + 1):
        n = nodes[i][j]
        node_id = f"{i},{j}"
        node_ids[n] = node_id
        color = "green" if (i == 0 and j == 0) else "red" if (i == a and j == b) else "#97C2FC"
        agraph_nodes.append(
            ANode(
                id=node_id,
                label=node_id,
                size=15,
                color=color,
                x=i * 150,
                y=j * 150
            )
        )

        for edge in n.edges_out:
            source_id = node_id
            target_id = f"{edge.end.x},{edge.end.y}"
            reverse_pair = (target_id, source_id)
            is_reverse = reverse_pair in edge_pairs
            edge_pairs.add((source_id, target_id))

            agraph_edges.append(
                AEdge(
                    source=source_id,
                    target=target_id,
                    label=" " * 7 + str(edge.weight) if is_reverse else str(edge.weight),
                    color="gray",
                    font={"color": "red" if is_reverse else "green"},
                    strokeWidth=1,
                )
            )

# Run Dijkstra's algorithm if button clicked
if run_dijkstra:
    start = nodes[0][0]
    start.weight = 0
    queue = []
    insert(queue, start)
    djikstras(queue)

    # Highlight the shortest path
    current = nodes[a][b]
    while current.edge_into:
        e = current.edge_into
        agraph_edges.append(
            AEdge(
                source=node_ids[e.start],
                target=node_ids[e.end],
                color="yellow",
                strokeWidth=12
            )
        )
        current = e.start

    if nodes[a][b].weight is not None:
        st.success(f"Shortest path total weight: {nodes[a][b].weight}")
    else:
        st.error("No path found.")

# Apply CSS to create a visible border
st.markdown("""
<style>
.main .block-container div[data-testid="stVerticalBlock"] > div:has(canvas),
.stAgraph,
iframe,
canvas,
div:has(> canvas) {
    border: 3px solid #808080 !important;
    border-radius: 8px !important;
    padding: 5px !important;
    margin: 10px 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Create the config with locked positions
config = Config(
    width=container_width,
    height=container_height,
    directed=True,
    physics=False,
    hierarchical=False,
    staticGraph=True,
    nodeHighlightBehavior=False,
    staticImage=True
)

agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

# Add a key explaining font colors
st.markdown("""
### 🔑 Edge Label Key
- 🟩 <span style='color:green;'>Green label</span>: Forward edge  
- 🟥 <span style='color:red;'>Red label</span>: Reverse edge  
""", unsafe_allow_html=True)
