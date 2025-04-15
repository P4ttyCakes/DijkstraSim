import streamlit as st
import matplotlib.pyplot as plt
import random
import numpy as np
import pandas as pd
import networkx as nx
import altair as alt
import time
from PIL import Image


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


def dijkstras(queue, nodes, a, b, visualize=False):
    """Run Dijkstra's algorithm with optional step-by-step visualization"""
    visited_nodes = []
    step_history = []

    while queue:
        current_node = dequeue(queue)
        visited_nodes.append((current_node.x, current_node.y))

        # Save the current state for visualization
        if visualize:
            # Find the current path to end (if exists)
            current_path = []
            if nodes[a][b].weight is not None:
                temp = nodes[a][b]
                while temp.edge_into is not None:
                    parent = temp.edge_into.start
                    current_path.append(((parent.x, parent.y), (temp.x, temp.y)))
                    temp = parent
                current_path.reverse()

            step_history.append({
                'visited': visited_nodes.copy(),
                'queue': [(n.x, n.y, n.weight) for n in queue],
                'path': current_path.copy(),
                'current': (current_node.x, current_node.y),
                'end_weight': nodes[a][b].weight
            })

        for edge in current_node.edges_out:
            neighbor = edge.end
            distance_through_current = current_node.weight + edge.weight

            if neighbor.weight is None:
                neighbor.weight = distance_through_current
                neighbor.edge_into = edge
                insert(queue, neighbor)
            elif neighbor.weight > distance_through_current:
                neighbor.weight = distance_through_current
                neighbor.edge_into = edge
                remove(queue, neighbor)
                insert(queue, neighbor)

    return step_history


def create_graph(a, b, min_weight, max_weight):
    """Create a grid graph with random edge weights"""
    # Create nodes
    nodes = []
    for i in range(a + 1):
        row = []
        for j in range(b + 1):
            row.append(Node(i, j))
        nodes.append(row)

    # Create horizontal edges
    for j in range(b + 1):
        for i in range(a):
            w = random.randint(min_weight, max_weight)
            nodes[i][j].edges_out.append(Edge(w, nodes[i][j], nodes[i + 1][j]))
            w = random.randint(min_weight, max_weight)
            nodes[i + 1][j].edges_out.append(Edge(w, nodes[i + 1][j], nodes[i][j]))

    # Create vertical edges
    for i in range(a + 1):
        for j in range(b):
            w = random.randint(min_weight, max_weight)
            nodes[i][j].edges_out.append(Edge(w, nodes[i][j], nodes[i][j + 1]))
            w = random.randint(min_weight, max_weight)
            nodes[i][j + 1].edges_out.append(Edge(w, nodes[i][j + 1], nodes[i][j]))

    # Create edge list for visualization
    edge_list = []
    for i in range(a + 1):
        for j in range(b + 1):
            for edge in nodes[i][j].edges_out:
                edge_list.append({
                    'start_x': i,
                    'start_y': j,
                    'end_x': edge.end.x,
                    'end_y': edge.end.y,
                    'weight': edge.weight
                })

    return nodes, edge_list


def get_path(nodes, a, b):
    """Extract the shortest path from source to target"""
    path = []
    current = nodes[a][b]
    while current.edge_into is not None:
        parent = current.edge_into.start
        path.append((parent, current))
        current = parent
    path.reverse()  # Reverse to get start-to-end order
    return path


def create_interactive_visualization(a, b, edge_list, step_data=None, final_path=None, current_step=0):
    """Create an interactive visualization of the graph and algorithm progress"""
    # Create NetworkX graph
    G = nx.Graph()

    # Add nodes
    for i in range(a + 1):
        for j in range(b + 1):
            G.add_node(f"{i},{j}", pos=(i, j))

    # Add edges with weights
    for edge in edge_list:
        G.add_edge(
            f"{edge['start_x']},{edge['start_y']}",
            f"{edge['end_x']},{edge['end_y']}",
            weight=edge['weight']
        )

    # Get positions
    pos = nx.get_node_attributes(G, 'pos')

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw edges with weights based on their value
    for edge in edge_list:
        start = (edge['start_x'], edge['start_y'])
        end = (edge['end_x'], edge['end_y'])
        weight = edge['weight']

        # Normalize weight for color intensity
        color_intensity = 0.2 + 0.6 * (weight - 1) / 9  # Scale from 0.2 to 0.8

        # Draw the edge
        ax.plot([start[0], end[0]], [start[1], end[1]],
                color=f'0.5', alpha=color_intensity, linewidth=1.5)

        # Add weight label at the middle of the edge
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y, str(weight), fontsize=8,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1),
                ha='center', va='center')

    # Draw all nodes
    for i in range(a + 1):
        for j in range(b + 1):
            ax.scatter(i, j, color='lightgray', s=100, edgecolor='black', zorder=5)

    # Highlight start and end nodes
    ax.scatter(0, 0, color='limegreen', s=150, edgecolor='black', zorder=10)
    ax.scatter(a, b, color='red', s=150, edgecolor='black', zorder=10)

    # If we have step data, visualize current state
    if step_data and current_step < len(step_data):
        current_data = step_data[current_step]

        # Highlight visited nodes
        visited_x = [x for x, y in current_data['visited']]
        visited_y = [y for x, y in current_data['visited']]
        ax.scatter(visited_x, visited_y, color='lightskyblue', s=120, alpha=0.7, edgecolor='black', zorder=6)

        # Highlight current node being processed
        if current_data['current']:
            ax.scatter(current_data['current'][0], current_data['current'][1],
                       color='blue', s=150, edgecolor='black', zorder=8)

        # Highlight current path
        for (sx, sy), (ex, ey) in current_data['path']:
            ax.plot([sx, ex], [sy, ey], color='blue', linewidth=3, zorder=7)

        # Add distance information
        if current_data['end_weight'] is not None:
            ax.set_title(f"Current shortest path distance: {current_data['end_weight']}", fontsize=14)
        else:
            ax.set_title("Finding shortest path...", fontsize=14)

    # If we have final path data and this is the last step, show it
    elif final_path and (step_data is None or current_step >= len(step_data)):
        for start_node, end_node in final_path:
            ax.plot([start_node.x, end_node.x], [start_node.y, end_node.y],
                    color='blue', linewidth=3, zorder=10)

        # Extract and plot all nodes in the path
        path_nodes = [final_path[0][0]]  # Start with the first node
        for _, end_node in final_path:
            path_nodes.append(end_node)

        path_x = [node.x for node in path_nodes]
        path_y = [node.y for node in path_nodes]
        ax.scatter(path_x, path_y, color='blue', s=120, edgecolor='black', zorder=11)

        # Add final distance
        if final_path:
            final_node = final_path[-1][1]
            ax.set_title(f"Final shortest path distance: {final_node.weight}", fontsize=14)

    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_xlim(-0.5, a + 0.5)
    ax.set_ylim(-0.5, b + 0.5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')

    plt.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Interactive Dijkstra's Algorithm"
    )


    # Create two columns for the interface
    left_col, right_col = st.columns([1, 3])

    with left_col:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.subheader("Grid Settings")

        a = st.slider("Grid Width", 3, 15, 8)
        b = st.slider("Grid Height", 3, 15, 8)

        min_weight, max_weight = st.slider("Edge Weight Range", 1, 10, (1, 5))

        # Algorithm visualization options
        st.subheader("Visualization")
        show_steps = st.checkbox("Show step-by-step execution", value=True)

        if show_steps:
            animation_speed = st.select_slider(
                "Animation Speed",
                options=["Slow", "Medium", "Fast"],
                value="Medium"
            )
            speed_mapping = {"Slow": .5, "Medium": 0.05, "Fast": 0.003}
            delay = speed_mapping[animation_speed]

        # Generate button with loading animation
        generate_button = st.button("Generate New Graph", type="primary")
        run_button = st.button("Run Dijkstra's Algorithm", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    # Session state to store graph data
    if 'nodes' not in st.session_state or generate_button:
        st.session_state.nodes, st.session_state.edge_list = create_graph(a, b, min_weight, max_weight)
        st.session_state.final_path = None
        st.session_state.step_history = None
        st.session_state.has_run = False

    # Visualization area
    with right_col:
        # If algorithm hasn't been run yet or after regenerating the graph
        if not st.session_state.has_run or generate_button:
            fig = create_interactive_visualization(a, b, st.session_state.edge_list)
            st.pyplot(fig)

            # Display grid information
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Grid Size", f"{a + 1}×{b + 1} nodes")
            with col2:
                edge_count = len(st.session_state.edge_list)
                st.metric("Edges", f"{edge_count}")
            with col3:
                avg_weight = sum(edge['weight'] for edge in st.session_state.edge_list) / edge_count
                st.metric("Avg. Weight", f"{avg_weight:.2f}")

        # Run the algorithm
        if run_button:
            # Initialize start node
            s = st.session_state.nodes[0][0]
            s.weight = 0

            # Initialize queue
            queue = []
            insert(queue, s)

            # Run Dijkstra's with visualization
            with st.spinner("Running Dijkstra's algorithm..."):
                st.session_state.step_history = dijkstras(queue, st.session_state.nodes, a, b, visualize=show_steps)
                st.session_state.final_path = get_path(st.session_state.nodes, a, b)
                st.session_state.has_run = True

            # Show step-by-step execution if requested
            if show_steps:
                step_placeholder = st.empty()
                progress_bar = st.progress(0)

                for i, step in enumerate(st.session_state.step_history):
                    progress = i / len(st.session_state.step_history)
                    progress_bar.progress(progress)

                    fig = create_interactive_visualization(
                        a, b, st.session_state.edge_list,
                        st.session_state.step_history,
                        st.session_state.final_path, i
                    )
                    step_placeholder.pyplot(fig)
                    time.sleep(delay)

                # Show final result
                progress_bar.progress(1.0)
                fig = create_interactive_visualization(
                    a, b, st.session_state.edge_list,
                    None, st.session_state.final_path
                )
                step_placeholder.pyplot(fig)

                # Success message with animation
                st.success(f"✅ Found shortest path with distance: {st.session_state.nodes[a][b].weight}")

                # Path details
                if st.session_state.final_path:
                    path_data = []
                    for i, (start, end) in enumerate(st.session_state.final_path):
                        edge_weight = end.weight - start.weight
                        path_data.append({
                            'Step': i + 1,
                            'From': f"({start.x},{start.y})",
                            'To': f"({end.x},{end.y})",
                            'Edge Weight': edge_weight,
                            'Total Distance': end.weight
                        })

                    path_df = pd.DataFrame(path_data)
                    st.dataframe(path_df, use_container_width=True)

            else:
                # Just show final result
                fig = create_interactive_visualization(
                    a, b, st.session_state.edge_list,
                    None, st.session_state.final_path
                )
                st.pyplot(fig)
                st.success(f"✅ Found shortest path with distance: {st.session_state.nodes[a][b].weight}")


if __name__ == "__main__":
    main()