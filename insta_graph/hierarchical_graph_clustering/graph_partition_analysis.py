import networkx as nx
import pandas as pd
from tqdm import tqdm

from typing import *
import statistics

def get_break_info(partitions: List[Tuple[Set]]) -> List[Tuple[int, int]]:
    """
    Determines the size of the original and the resulting smaller clusters after each split.

    :param partitions: A list of tuples, where each tuple contains sets representing clusters.
    :return: A list of tuples where each tuple holds the size of the original cluster and the size of the smallest resulting cluster after a split.

    Each tuple in the returned list corresponds to a split operation on a cluster, capturing the decrease in size from the original cluster to the smallest cluster that results from the split.
    """
    info = []

    # Convert the tuple of sets to a set of frozensets, so we can perform set operations
    converted_partitions = [set(map(frozenset, partition)) for partition in partitions]

    for i, partition in enumerate(converted_partitions):
        if i == 0:
            original_set = set().union(*converted_partitions[0])
            smaller_set_len = min(len(resulting_set) for resulting_set in converted_partitions[0] if resulting_set.issubset(original_set))
            info.append((len(original_set), smaller_set_len))
        else:
            # find the sets that were broken up
            original_sets = converted_partitions[i-1] - converted_partitions[i]

            # find the resulting sets
            resulting_sets = converted_partitions[i] - converted_partitions[i-1]

            # for each original set, find the corresponding resulting sets
            for original_set in original_sets:
                smaller_set_len = min(len(resulting_set) for resulting_set in resulting_sets if resulting_set.issubset(original_set))
                info.append((len(original_set), smaller_set_len))

    return info

def gini_coefficient_for_numbers(numbers):
    # Ensure the list is not empty to avoid division by zero
    if not numbers:
        return None

    # Sort the numbers
    numbers.sort()

    # Calculate the cumulative sum of the numbers
    cumulative_sums = [sum(numbers[:i+1]) for i in range(len(numbers))]
    total_sum = cumulative_sums[-1]  # Total sum is the last item in cumulative_sums

    # Calculate the Gini coefficient using the corrected approach
    # Sum the differences between the cumulative sums and the equal distribution line
    gini_sum = sum([(i + 1) - cumulative_sums[i] / total_sum * len(numbers) for i in range(len(numbers))])

    # Gini coefficient is the ratio of the gini_sum to the total number of elements, normalized by the number of elements
    gini_index = gini_sum / len(numbers) / len(numbers)
    return gini_index

def get_hierarchical_cluster_analysis_df(G, partitions):
    """
    Generates a DataFrame containing performance metrics for partitions of a graph.

    :param G: The graph from which partitions are derived.
    :param partitions: A list of tuples, where each tuple contains sets representing the partitions of the graph at each iteration.
    :return: A pandas DataFrame with columns for the number of clusters, modularity, performance, and cluster size statistics at each partitioning step.

    The DataFrame provides insights into how the partitioning impacts various performance metrics, including modularity and the distribution of cluster sizes.
    """

    # Variable initialization
    n_clusters = []
    avg_cluster_size = []
    max_cluster_size = []
    q1_cluster_size = []
    median_cluster_size = []
    q3_cluster_size = []
    min_cluster_size = []
    modularity = []
    coverage = []
    performance = []
    len_gini_coefs = []

    original_cluster_size, break_size = [info[0] for info in get_break_info(partitions)], [info[1] for info in get_break_info(partitions)]

    for i, partition in enumerate(tqdm(partitions)):
        n_clusters.append(i + 2)
        modularity.append(nx.community.modularity(G, communities=partition))
        coverage.append(nx.community.quality.partition_quality(G, partition)[0])
        performance.append(nx.community.quality.partition_quality(G, partition)[1])
        clusters = []

        # Cluster Size Statistics
        cluster_sizes = [len(cluster) for cluster in partition]
        quantiles = statistics.quantiles(cluster_sizes, n=4)
        q1_cluster_size.append(quantiles[0])
        median_cluster_size.append(quantiles[1])
        q3_cluster_size.append(quantiles[2])
        min_cluster_size.append(min(cluster_sizes))
        max_cluster_size.append(max(cluster_sizes))
        len_gini_coefs.append(gini_coefficient_for_numbers(cluster_sizes))

    df = pd.DataFrame({
        'n_clusters': n_clusters,
        'modularity': modularity,
        'coverage': coverage,
        'performance': performance,
        'q1_cluster_size': q1_cluster_size,
        'median_cluster_size': median_cluster_size,
        'q3_cluster_size': q3_cluster_size,
        'min_cluster_size': min_cluster_size,
        'max_cluster_size': max_cluster_size,
        'original_cluster_size' : original_cluster_size,
        'break_size' : break_size,
        'len_gini_coefficient' : len_gini_coefs
    })

    df['change_in_modularity'] = df['modularity'] - df['modularity'].shift(1)
    df['change_in_performance'] = df['performance'] - df['performance'].shift(1)
    df['change_in_coverage'] = df['coverage'] - df['coverage'].shift(1)
    return df

import pandas as pd
import networkx as nx
from collections import defaultdict

def get_cluster_analysis_df(graph, cluster_name):
    """
    Analyzes clusters in a graph and returns a DataFrame with various metrics.

    :param graph: A networkx graph object.
    :param cluster_name: The name of the attribute in the nodes that contains the cluster identifier.
    :return: A pandas DataFrame with columns for the cluster identifier, number of nodes, number of intra-cluster edges,
             number of inter-cluster edges, density, and the labels of the top three most connected nodes in each cluster.
    """
    # Initialize the DataFrame to store results
    cluster_df = pd.DataFrame()

    # Initialize a dictionary to count intra- and inter-cluster edges
    edge_count = defaultdict(lambda: {'intra': 0, 'inter': 0})

    # Calculate intra- and inter-cluster edges
    for node1, node2 in graph.edges():
        cluster1 = graph.nodes[node1][cluster_name]
        cluster2 = graph.nodes[node2][cluster_name]
        if cluster1 == cluster2:
            edge_count[cluster1]['intra'] += 1
        else:
            edge_count[cluster1]['inter'] += 1
            edge_count[cluster2]['inter'] += 1

    # Calculate the degree of each node and sort within each cluster
    cluster_nodes = defaultdict(list)
    for node, data in graph.nodes(data=True):
        cluster = data[cluster_name]
        cluster_nodes[cluster].append((node, graph.degree[node]))

    # Sort nodes in each cluster by degree and get top 3
    top_nodes = {cluster: sorted(node_list, key=lambda x: x[1], reverse=True)[:3] for cluster, node_list in cluster_nodes.items()}

    # Create DataFrame entries for each cluster
    rows = []
    for cluster, nodes in cluster_nodes.items():
        top_3_labels = ', '.join([str(node[0]) for node in top_nodes[cluster]])
        num_nodes = len(nodes)
        intra_edges = edge_count[cluster]['intra']
        # Density calculation for undirected graph without self-loops
        density = (2 * intra_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
        rows.append({
            cluster_name.lower(): cluster,
            'num_nodes': num_nodes,
            'intra_edges': intra_edges,
            'inter_edges': edge_count[cluster]['inter'],
            'top_3_connected_nodes': top_3_labels,
            'density': density
        })

    # Concatenate all rows to the DataFrame
    cluster_df = pd.concat([cluster_df, pd.DataFrame(rows)], ignore_index=True)

    return cluster_df
