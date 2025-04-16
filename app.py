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

# Initialize session state for selected node
if 'selected_node' not in st.session_state:
    st.session_state.selected_node = None


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

        # Set node color based on selection state or special positions
        if st.session_state.selected_node == node_id:
            color = "#FFA500"  # Orange for selected node
        elif (i == 0 and j == 0):
            color = "green"  # Start node
        elif (i == a and j == b):
            color = "red"  # End node
        else:
            color = "#97C2FC"  # Default color

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
                strokeWidth=20
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

# Create the config with interactive options
config = Config(
    width=container_width,
    height=container_height,
    directed=True,
    physics=False,
    hierarchical=False,
    staticGraph=False,  # Changed to False to allow interaction
    nodeHighlightBehavior=True,  # Enable highlighting on hover
    staticImage=False  # Changed to False to allow interaction
)

# Create buttons for each node
st.write("### Select a Node")
cols = st.columns(min(5, a + 1))  # Create column layout for buttons
node_buttons = {}

for i in range(a + 1):
    for j in range(b + 1):
        node_id = f"{i},{j}"
        col_idx = i % len(cols)  # Distribute buttons across columns

        # Add some visual indication if this is start or end node
        button_label = node_id
        if i == 0 and j == 0:
            button_label = f"🟢 {node_id}"
        elif i == a and j == b:
            button_label = f"🔴 {node_id}"

        # Add selection indicator
        if st.session_state.selected_node == node_id:
            button_label = f"✓ {button_label}"

        with cols[col_idx]:
            if st.button(button_label, key=f"btn_{node_id}"):
                st.session_state.selected_node = node_id
                st.rerun()

# Display the graph
selected_node = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

# If a node was selected in the graph, update our selection
if selected_node:
    st.session_state.selected_node = selected_node
    st.rerun()

# Display selected node information
if st.session_state.selected_node:
    st.write(f"### Selected Node: {st.session_state.selected_node}")

    # Parse coordinates from node id
    try:
        x, y = map(int, st.session_state.selected_node.split(','))

        # Show node details
        st.write(f"Position: ({x}, {y})")

        # Show connected edges
        st.write("#### Connected Edges:")
        edge_info = []
        for i in range(a + 1):
            for j in range(b + 1):
                if i == x and j == y:
                    for edge in nodes[i][j].edges_out:
                        edge_info.append(f"To ({edge.end.x}, {edge.end.y}) - Weight: {edge.weight}")

        if edge_info:
            for info in edge_info:
                st.write(f"- {info}")
        else:
            st.write("No connected edges found.")

        # Add a button to clear selection
        if st.button("Clear Selection"):
            st.session_state.selected_node = None
            st.rerun()
    except ValueError:
        st.error("Invalid node format")

# Add a key explaining font colors and node selection
st.markdown("""
### 🔑 Edge Label Key
- 🟩 <span style='color:green;'>Green label</span>: Forward edge  
- 🟥 <span style='color:red;'>Red label</span>: Reverse edge  

### Node Selection
- Click on any node in the graph or use the buttons above to select a node
- Selected nodes will appear in orange in the graph
""", unsafe_allow_html=True)