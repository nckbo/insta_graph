import matplotlib.pyplot as plt
from pyvis.network import Network
import networkx as nx

from typing import *
import random
import webbrowser
import colorsys
import math
import os



def generate_distinct_colors(n, seed=None):
    """
    Generate a list of distinct colors using the golden ratio conjugate.

    Parameters:
    - n (int): The number of distinct colors to generate.
    - seed (int, optional): Seed for random number generation. If provided, the generated colors will be consistent for the same seed value.

    Returns:
    - list: A list of distinct colors in hexadecimal format.
    """
    # Set the seed for reproducibility
    if seed is not None:
        random.seed(seed)

    colors = []
    golden_ratio_conjugate = 0.618033988749895

    hue = random.random()  # Starting point (randomized)

    for i in range(n):
        hue += golden_ratio_conjugate
        hue %= 1  # Ensure hue stays between 0 and 1

        lightness = random.uniform(0.6, 0.9)  # Avoid very light or very dark colors
        saturation = random.uniform(0.6, 0.9)  # A bit of randomness to get grays and vibrant colors

        r, g, b = [int(x * 255) for x in colorsys.hls_to_rgb(hue, lightness, saturation)]

        color = "#{:02x}{:02x}{:02x}".format(r, g, b)
        colors.append(color)

    return colors

def generate_pyvis_graph(G: nx.Graph, positions: Dict, partition: Dict, output_filepath, cluster_name: str = 'cluster', menus=True, color_seed=42) -> nx.Graph:
    """
    Generates a Pyvis network graph from a NetworkX graph, applies a partition coloring based on the provided
    cluster mapping, and saves the visualized graph to an HTML file.

    :param G: The NetworkX graph to be visualized.
    :type G: nx.Graph
    :param positions: A dictionary with nodes as keys and (x, y) positions as values.
    :type positions: Dict
    :param partition: A dictionary mapping nodes to their respective cluster identifiers.
    :type partition: Dict
    :param output_filepath: The path to the file where the HTML output will be saved. If a file with the same name
                            exists, the function will not overwrite it and will prompt for action.
    :type output_filepath: str
    :param cluster_name: The attribute name to be used for the cluster information on each node. Defaults to 'cluster'.
    :type cluster_name: str
    :param menus: If True, selection and filter menus will be enabled in the Pyvis graph. Defaults to True.
    :type menus: bool
    :return: The original NetworkX graph with cluster and color attributes assigned to its nodes.
    :rtype: nx.Graph

    This function also creates a color mapping for each distinct cluster and applies it to the nodes.
    The positions provided in the `positions` dictionary are used to set the x, y coordinates of the nodes
    in the Pyvis network graph. If the `output_filepath` already exists, the function will issue a warning and
    not proceed with writing the file, to prevent overwriting existing files.
    """


    # Assign the partition as an attribute to each node
    for node, cluster in partition.items():
        G.nodes[node][cluster_name] = cluster

    # Create color dictionary
    distinct_clusters = set(v for k, v in partition.items())
    color_mapping = {k: v for k, v in zip(distinct_clusters, generate_distinct_colors(len(distinct_clusters), seed=color_seed))}

    # Map clusters to colors
    for node in G.nodes:
        G.nodes[node]['color'] = color_mapping[G.nodes[node][cluster_name]]

    # Set x, y in each node given the returned positions
    for node in G.nodes:
        G.nodes[node]['x'] = positions[node][0]
        G.nodes[node]['y'] = positions[node][1]

    if os.path.exists(output_filepath):
        print("There's already a file here. Rename, move, or delete the file before proceeding")
    else:
        nt_menu = Network(select_menu=menus, filter_menu=menus, notebook=False, bgcolor='#222222', font_color= '#FFFFFF', cdn_resources='in_line')
        nt_menu.from_nx(G)
        nt_menu.toggle_physics(False) # Ensure physics are off so nodes use custom positions
        nt_menu.write_html(output_filepath)

    return G

def visualize_color_key(color_key):
    """
    Visualize a color key using swatches.

    Parameters:
    - color_key (dict): A dictionary where keys represent groups and values are their corresponding colors.

    Returns:
    - None: This function only produces a visualization.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    # Iterate through the color key to create swatches
    for i, (group, color) in enumerate(color_key.items()):
        ax.fill_between([0, 10], i, i + 1, color=color)
        ax.text(11, i + 0.5, f"Group {group}: {color}", va='center')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, len(color_key))
    ax.axis('off')
    plt.show()


def rotate_coordinates(coord_dict, degrees):
    # Convert degrees to radians
    radians = math.radians(degrees)

    # Cosine and sine of the rotation angle
    cos_angle = math.cos(radians)
    sin_angle = math.sin(radians)

    # Dictionary to store rotated coordinates
    rotated_coords = {}

    for label, (x, y) in coord_dict.items():
        # Apply the rotation matrix to each coordinate
        x_rotated = x * cos_angle - y * sin_angle
        y_rotated = x * sin_angle + y * cos_angle

        # Update the dictionary with rotated coordinates
        rotated_coords[label] = (x_rotated, y_rotated)

    return rotated_coords



