import networkx as nx
from sklearn.cluster import SpectralClustering
import numpy as np

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering
from networkx.algorithms.components import connected_components

import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

def bisect_community(G, community):
    """
    Bisects a community in the graph, returning the largest connected component
    and the union of all other components if the community is disconnected.
    If the community is connected, it applies Spectral Clustering to divide it
    into two communities.

    Parameters:
    - G (nx.Graph): The input graph.
    - community (set): The set of nodes representing the community to be bisected.

    Returns:
    - tuple(set, set): A tuple containing two sets of nodes, representing the two communities
      obtained after bisecting the input community. If the community is disconnected, the
      first set is the largest connected component, and the second set is the union of all
      other components.

    Note:
    This function constructs a subgraph from the provided community. If the subgraph is
    disconnected, the function returns the largest connected component and the other components
    as separate communities. If the subgraph is connected, it applies Spectral Clustering to
    divide it into two communities.
    """
    subgraph = G.subgraph(community).copy()
    connected_subgraphs = list(nx.connected_components(subgraph))

    # Check if the subgraph is disconnected
    if len(connected_subgraphs) > 1:
        # Find the largest connected component
        largest_component = max(connected_subgraphs, key=len)
        # Create a set of all other nodes excluding the largest connected component
        other_components = set(community) - largest_component
        return largest_component, other_components
    else:
        # Subgraph is connected; proceed with spectral clustering
        nodes = list(subgraph.nodes())
        adjacency_matrix = np.asarray(nx.to_numpy_matrix(subgraph, nodelist=nodes))

        # Use Spectral Clustering to bisect the community
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', random_state=0)
        labels = sc.fit_predict(adjacency_matrix)

        community1 = {nodes[i] for i in range(len(nodes)) if labels[i] == 0}
        community2 = {nodes[i] for i in range(len(nodes)) if labels[i] == 1}

        return community1, community2

def recursive_spectral_bisection(G, community_size_threshold = 8):
    """
    Constructs a partition hierarchy of a graph using iterative spectral bisection.

    Parameters:
    - G (nx.Graph): The input graph.
    - community_size_threshold (int): The size threshold beyond which a community should be bisected.

    Returns:
    - list(tuple): A list of tuples, where each tuple represents a partition (i.e., a set of communities)
      of the graph. The first tuple contains one community with all nodes, and each subsequent tuple
      in the list represents a refined partition obtained by bisecting the largest community of the
      previous partition.

    Note:
    The function iteratively bisects the largest community of the current partition until all communities
    are below the given threshold in size. The result is a hierarchical decomposition of the graph
    where neighboring levels in the hierarchy are related by community bisections.
    """
    hierarchy = [(set(G.nodes()),)]  # Start with all nodes in one community

    while max(len(community) for community in hierarchy[-1]) > community_size_threshold:
        # Convert the last hierarchy level to a list for manipulation
        partition = list(hierarchy[-1])

        # Find the largest community in the latest partition
        largest_community = max(partition, key=len)
        partition.remove(largest_community)

        # Bisect the largest community
        community1, community2 = bisect_community(G, largest_community)

        # Add the bisected communities to the current partition
        partition.extend([community1, community2])

        # Add the new partition as a tuple to the hierarchy
        hierarchy.append(tuple(partition))

    return hierarchy

#%%

