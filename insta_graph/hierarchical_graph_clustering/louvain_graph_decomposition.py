import community as community_louvain
import networkx as nx

from itertools import combinations
from typing import *


def generate_partitions_list_from_dict(partitions_dict) -> List[Set]:
    """
    Convert a dictionary-based representation of partitions into a list of sets.

    Given a dictionary where keys are node labels and values are cluster assignments,
    this function returns a list of sets. Each set consists of node labels belonging
    to the same cluster.

    Parameters:
    ----------
    partitions_dict : dict
        A dictionary where keys are node labels and values are integers representing
        cluster assignments. Typically, this is the output format of algorithms like
        community_louvain.best_partition().

    Returns:
    -------
    List[Set]
        A list where each element is a set of node labels. Each set corresponds to
        the nodes in a single partition. This is the format we need to generate our linkage matrix.

    Examples:
    --------
    >>> partitions_dict = {'A': 0, 'B': 0, 'C': 1, 'D': 1, 'E': 2}
    >>> generate_partitions_list_from_dict(partitions_dict)
    [{'A', 'B'}, {'C', 'D'}, {'E'}]
    """

    dict_list = {}
    for k, v in partitions_dict.items():
        try:
            dict_list[v].append(k)
        except KeyError:
            dict_list[v] = [k]
    return [set(v) for k, v in dict_list.items()]

def bisecting_louvain_decomposition(G, community_size_threshold=20) -> List[Tuple[Set]]:
    """
    Decompose a graph using bisecting strategy based on the Louvain community detection algorithm.

    This function aims to identify meaningful communities in the graph using the Louvain method, but
    it employs a bisecting strategy. When a community is larger than the specified threshold,
    the function attempts to bisect it using the Louvain algorithm, and this process is repeated
    iteratively until all communities are below the threshold or no further improvement in modularity
    can be achieved.

    Parameters:
    ----------
    G : NetworkX graph
        The graph to decompose into communities. Nodes should be unique, and edges represent connections
        between nodes.

    community_size_threshold : int, optional (default=20)
        The maximum allowed size for a community. Communities larger than this threshold will be attempted
        to be split using the Louvain method.

    Returns:
    -------
    list of tuples of sets
        A list representing the hierarchy of communities detected. Each element in the list is a tuple
        representing the state of the graph's communities at that iteration. Each tuple contains sets,
        where each set is a community of nodes. The last item in the list represents the final partitioning
        of the graph's nodes.

    Notes:
    -----
    - The function employs a caching strategy to avoid recomputing modularity for subgraphs already processed.
    - The function uses the `community_louvain.best_partition()` method to perform the Louvain community detection.
    - A potential weakness of this method is that it can produce unbalanced hierarchies where some branches
      are much deeper than others.

    Examples:
    --------
    >>> G = nx.erdos_renyi_graph(100, 0.1)
    >>> hierarchy = bisecting_louvain_decomposition(G, community_size_threshold=10)
    >>> for level in hierarchy:
    ...     print(level)
    """
    def hash_subgraph(H):
        return hash(tuple(sorted(H.nodes())))

    def highest_modularity_group_to_split(G, groupings):
        """
        Find the community in G with the highest modularity for potential splitting.

        :param G: NetworkX graph
        :param groupings: tuple of sets of node communities
        :return: group with highest modularity and its modularity value
        """
        best_group_to_split = None
        best_modularity = -100
        for group in groupings:
            modularity = nx.community.modularity(G, [group, list(set(node for node in G.nodes) - set(group))])
            if modularity > best_modularity:
                best_group_to_split = group
                best_modularity = modularity
        return best_group_to_split, best_modularity


    louvain_hierarchy = []
    modularity_cache = {}

    # Load in initial unified group
    if len(G.nodes) < community_size_threshold:
        return list(tuple(frozenset(G.nodes)))
    else:
        groupings = [set(G.nodes)]
        louvain_hierarchy.append(groupings)

        prev_n_groups = 0

        while max([len(group) for group in groupings]) > community_size_threshold:
            best_modularity = -100
            best_split = None
            best_split_other_half = None
            group_to_remove = None

            for group in groupings:
                if (len(group) < community_size_threshold):
                    continue

                H = nx.subgraph(G, group) # Generate a subgraph

                # Check cache before computing
                subgraph_hash = hash_subgraph(H)
                if subgraph_hash in modularity_cache:
                    split, split_modularity = modularity_cache[subgraph_hash]
                else:
                    # Find the best node to split and the modularity
                    subgroupings = generate_partitions_list_from_dict(community_louvain.best_partition(H))
                    if len(subgroupings) <= 1:
                        continue

                    split, split_modularity = highest_modularity_group_to_split(H, subgroupings)
                    modularity_cache[subgraph_hash] = (split, split_modularity)

                if split_modularity > best_modularity:
                    best_modularity = split_modularity
                    best_split = split
                    best_split_other_half = set(H.nodes) - split
                    group_to_remove = set(H.nodes)

            # If there's no best split, return
            if not best_split:
                return louvain_hierarchy

            # Generate new groupings based on split
            groupings = [g for g in groupings if g != group_to_remove]
            groupings.append(best_split)
            groupings.append(best_split_other_half)

            louvain_hierarchy.append(tuple(groupings))

            if len(groupings) <= prev_n_groups:
                # print(len(groupings))
                break
            else:
                prev_n_groups = len(groupings)

    return louvain_hierarchy

def mutually_exclusive_pairs(lst: List[Any]) -> List[Tuple[List[Any], List[Any]]]:
    """
    Returns all unique mutually exclusive pairs of groups of elements from the provided list.

    :param lst: Input list containing any type of elements.
    :type lst: List[Any]
    :return: A list of tuple pairs, where each tuple consists of two mutually exclusive lists. Combined, these two lists contain all elements of the original list.
    :rtype: List[Tuple[List[Any], List[Any]]]

    Example:
    >>> mutually_exclusive_pairs([1, 2, 3])
    [([1], [2, 3]), ([2], [1, 3]), ([3], [1, 2])]
    """


    n = len(lst)
    result = set()

    for i in range(1, n // 2 + 1):  # Limiting to half of the list size
        for combo in combinations(lst, i):
            remaining = [x for x in lst if x not in combo]

            # Create pairs ensuring smaller group or lexicographically smaller one is first
            if len(combo) < len(remaining) or (len(combo) == len(remaining) and sorted(combo) < sorted(remaining)):
                result.add((tuple(sorted(combo)), tuple(sorted(remaining))))
            else:
                result.add((tuple(sorted(remaining)), tuple(sorted(combo))))

    return [(list(pair[0]), list(pair[1])) for pair in sorted(result)]

def flatten_list_of_list_of_fsets(list_of_lists: Union[List[List[FrozenSet]], Tuple[List[FrozenSet]]]) -> List[Set]:
    """
    Converts an iterable of lists of frozensets into a list of sets.

    :param list_of_lists: An iterable containing lists of frozensets
    :return: A list of sets
    """

    if not isinstance(list_of_lists, (list, tuple)):
        raise TypeError(f"Expected list_of_lists to be a list or tuple, but got {type(list_of_lists)}\tinputted list_of_lists: {list_of_lists}")

    if not all(isinstance(sublist, list) and all(isinstance(item, frozenset) for item in sublist) for sublist in list_of_lists):
        raise TypeError(f"Expected each element of list_of_lists to be a list of frozensets \tinputted list_of_lists: {list_of_lists}")

    return [set().union(*fset_list) for fset_list in list_of_lists]

def get_split(G: nx.Graph, groups_to_consider: List[FrozenSet], other_groups: List[FrozenSet]=None, partition_size_threshold = 10) -> Tuple[List[FrozenSet], List[FrozenSet]]:
    """
    A helper function for `order_splits` that identifies the optimal way to split
    the given groups in order to maximize the modularity of the graph.

    Parameters:
    - G (nx.Graph): The NetworkX graph to be split.
    - groups_to_consider (List[FrozenSet]): The groups that are candidates for splitting.
    - other_groups (List[FrozenSet], optional): Other groups that are not being considered for splitting but are
      part of the graph. Defaults to None.
    - partition_size_threshold: Maximum number of groups to consider before creating a supergraph and performing a spectral bisection,
      This is to help prevent long computation times

    Returns:
    - Tuple[List[FrozenSet], List[FrozenSet]]: The optimal grouping of partitions after splitting.

    Note:
    The function employs a super-graph approach if there are more than partition_size_threshold groups to consider, otherwise
    it evaluates all possible mutually exclusive pairs. The result aims to maximize the modularity of the graph.
    """
    # region Input type checking
    assert isinstance(groups_to_consider, list), "groups_to_consider must be a list"
    for group in groups_to_consider:
        assert isinstance(group, FrozenSet), f"Each group in groups_to_consider must be a Frozenset | Type: {type(group)} Group: {group}"

    if other_groups:
        assert isinstance(other_groups, list), "other_groups must be a list"
        for group in other_groups:
            assert isinstance(group, FrozenSet), "Each group in other_groups must be a Frozenset"
    # endregion

    best_modularity = -100
    best_pair = None

    if len(groups_to_consider) > partition_size_threshold:
        # Construct super-graph
        super_G = nx.Graph()
        for group in groups_to_consider:
            super_G.add_node(group)
            for node in group:
                for neighbor in G.neighbors(node):
                    neighbor_group = next((g for g in groups_to_consider if neighbor in g), None)
                    if neighbor_group and neighbor_group != group:
                        if super_G.has_edge(group, neighbor_group):
                            super_G[group][neighbor_group]['weight'] += 1
                        else:
                            super_G.add_edge(group, neighbor_group, weight=1)
        # Apply a bisection method on super_G, for simplicity, I am using the spectral clustering approach
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters=2, affinity='precomputed').fit(nx.to_numpy_array(super_G))
        labels = clustering.labels_

        best_pair = ([groups_to_consider[i] for i in range(len(labels)) if labels[i] == 0],
                     [groups_to_consider[i] for i in range(len(labels)) if labels[i] == 1])
        return best_pair
    else:
        for pair in mutually_exclusive_pairs(groups_to_consider):  # Expected to return pairs of groups to consider
            full_partition = flatten_list_of_list_of_fsets(pair)

            # Incorporate other groups if provided
            if other_groups:
                full_partition.extend([set(fs) for fs in other_groups]) #Convert to set before extending

            modularity = nx.community.modularity(G, full_partition)
            if modularity > best_modularity:
                best_modularity = modularity
                best_pair = pair

        return best_pair


def initialize_and_split(G: nx.Graph, partitions: List[FrozenSet], partition_size_threshold) -> Tuple:
    """
    Initializes necessary variables and performs the first split on the graph based on the given partitions.

    This function is a helper for the order_splits() function and prepares the ground for subsequent iterative splits.

    :param G: The networkx graph to be split.
    :type G: nx.Graph
    :param partitions: Initial partitions to be considered for the first split. Each partition is a frozenset of nodes.
    :type partitions: List[FrozenSet]

    :return: A tuple containing the first split of partitions and a list of these partitions ready for further splits.
    :rtype: Tuple

    Example:
        Given a graph `G` with nodes 1, 2, 3, and 4, and an initial partition [frozenset([1, 2]), frozenset([3, 4])],
        the function might return two partitions ({1, 2}, {3, 4}) and a list of these partitions for further processing.
    """

    # Initial split
    first_split = get_split(G, groups_to_consider=partitions, partition_size_threshold=partition_size_threshold)

    # Prepare for subsequent splits
    lists_to_split = [lst for lst in first_split]

    return first_split, lists_to_split

def iterative_splitting(G: nx.Graph, lists_to_split: List[List[FrozenSet]], partition_size_threshold) -> List[Tuple[Set]]:
    """
    Iteratively splits the given graph partitions until no more valid splits can be made. Each iteration finds
    the optimal split for the largest current partition, and the split is then added to the hierarchy.

    This function is designed as a helper for the order_splits() function.

    :param G: The networkx graph to be split.
    :type G: nx.Graph
    :param lists_to_split: A list containing the current partitions, with each partition being a list of frozen sets.
    :type lists_to_split: List[List[FrozenSet]]

    :return: A list of tuples, where each tuple represents the partitions at each iteration of the split. Each partition is a set.
    :rtype: List[Tuple[Set]]

    Raises:
        ValueError: If there are mismatches between the number of elements in the provided partitions and the graph's nodes.

    Example:
        Assume we have a graph `G` with nodes 1, 2, 3, and 4, and we want to iteratively split it based on some criteria.
        Starting with an initial partition [[frozenset([1, 2]), frozenset([3, 4])]], the function might return:
        [({1, 2, 3, 4}), ({1, 2}, {3, 4})] if the optimal split keeps nodes 1 and 2 together and nodes 3 and 4 together.
    """

    split_hierarchy_flat = []


    while True:
        # Calculate max group size
        max_group_size = max([len(lst) for lst in lists_to_split])

        # Break condition if the largest group size is 1
        if max_group_size == 1:
            break

        # Extract the current group
        current_group = max(lists_to_split, key=len)

        # Identify groups not being considered for the current split
        other_lists = [lst for lst in lists_to_split if lst != current_group]

        # Find the best split for the current group
        other_lists_flat_fs = [frozenset(s) for s in flatten_list_of_list_of_fsets(other_lists)]
        pair = get_split(G, groups_to_consider=current_group, other_groups=other_lists_flat_fs, partition_size_threshold=partition_size_threshold)
        # region pair error handling
        if not isinstance(pair, tuple) or not all(isinstance(lst, list) and isinstance(item, frozenset) for lst in pair for item in lst):
            raise ValueError("pair must be of type Tuple[List[FrozenSet]]!")
        # endregion

        # This should remove the other lists
        lists_to_split = [lst for lst in other_lists]
        lists_to_split.extend([lst for lst in pair])
        # region lists_to_split and other_list error handling
        if not all(isinstance(lst, list) and all(isinstance(item, frozenset) for item in lst) for lst in other_lists):
            raise ValueError("other_lists must be of type List[List[frozenset]]!")
        total_elements_in_other_lists = sum([len(fset) for fset in other_lists_flat_fs])
        total_elements_in_current_group = sum([len(fset) for fset in current_group])
        if total_elements_in_other_lists + total_elements_in_current_group != len(G.nodes):
            raise ValueError(f"Total elements in other_lists ({total_elements_in_other_lists}) and current_group ({total_elements_in_current_group}) ({total_elements_in_other_lists + total_elements_in_current_group}) do not match the number of nodes in G ({len(G.nodes)})")

        if not all(isinstance(lst, list) and all(isinstance(item, frozenset) for item in lst) for lst in lists_to_split):
            raise ValueError("lists_to_split must be of type List[List[frozenset]]!")
        # Check if the total number of elements in lists_to_split equals the number of nodes in G
        total_elements_in_lists_to_split = sum([len(item) for sublist in lists_to_split for item in sublist])
        if total_elements_in_lists_to_split != len(G.nodes):
            raise ValueError(f"Total elements in lists_to_split ({total_elements_in_lists_to_split}) do not match the number of nodes in G ({len(G.nodes)})")
        # endregion

        split_hierarchy_flat.append(tuple(flatten_list_of_list_of_fsets(lists_to_split)))
        if not all(isinstance(item, tuple) and all(isinstance(s, set) for s in item) for item in split_hierarchy_flat):
            raise ValueError("split_hierarchy_flat must be of type List[Tuple[Set]]!")

    return split_hierarchy_flat

def order_splits(G: nx.Graph, partitions: List[FrozenSet], partition_size_threshold) -> List[Tuple[List[Set]]]:
    """
    Orders the initial partitions of a graph based on their decomposition modularity.

    Given an initial set of partitions, this function determines the sequence in which the graph decomposes into these
    partitions in a way that prioritizes splits with the highest modularity. The function uses the `get_split` method
    to iteratively split the graph, optimizing the split sequence for modularity.

    :param G: The networkx graph whose initial partitions need to be ordered.
    :type G: nx.Graph
    :param partitions: Initial partitions of the graph. Each partition is a frozenset of nodes.
    :type partitions: List[FrozenSet]

    :return: A list of tuples, where each tuple represents the partitions at each split iteration. Each partition within a tuple is represented as a set.
    :rtype: List[Tuple[List[Set]]]

    Raises:
        ValueError: If input types are not as expected or if there's a mismatch between returned partition types and expected types.

    Example:
        Given a graph `G` and an initial set of partitions represented as [frozenset([1]), frozenset([2]), frozenset([3]), frozenset([4])],
        where each number represents a distinct partition, `order_splits` might determine that the first best split, based on modularity,
        groups partitions {1, 2} together and partitions {3, 4} together.
        The first iteration might yield the combinations ({1, 2}, {3, 4}). In the next iteration, if the combination of {1, 2} into
        {1} and {2} yields a higher modularity than combining {3, 4} into {3} and {4}, the combinations would be ({1}, {2}, {3, 4}).
        In the final iteration, each initial partition is isolated, yielding ({1}, {2}, {3}, {4}).

    Note:
        This function uses the `get_split` method for determining the optimal partition combinations at each iteration based on modularity.
    """


    # region Type checks for inputs
    if not isinstance(G, nx.Graph):
        raise ValueError("Input G must be of type nx.Graph!")
    if not all(isinstance(partition, frozenset) for partition in partitions):
        raise ValueError("Input partitions must be a list of frozensets!")
    # endregion
    first_split, lists_to_split = initialize_and_split(G, partitions, partition_size_threshold)


    split_hierarchy_flat = [tuple(flatten_list_of_list_of_fsets(first_split))]
    split_hierarchy_flat.extend(iterative_splitting(G, lists_to_split, partition_size_threshold))

    if not all(isinstance(item, tuple) and all(isinstance(s, set) for s in item) for item in split_hierarchy_flat):
        raise ValueError("split_hierarchy_flat must be of type List[Tuple[Set]]!")

    return split_hierarchy_flat

def balanced_louvain_decomposition(G, community_size_threshold=20, partition_size_threshold=10):
    """
    Decompose a graph to produce balanced and meaningful communities using an enhanced Louvain method.

    This function extends the Louvain bisecting decomposition approach by maintaining the original optimal
    Louvain partitions at every hierarchy level. Unlike the basic Louvain bisecting approach, this method ensures
    balanced hierarchies where neighboring branches are generally related to each other. It is especially effective
    at recognizing substructures within densely connected communities, which the classic Louvain method may overlook.

    The algorithm operates iteratively, always working with the largest community that surpasses the community size
    threshold and has the potential to be split further. This method uses `order_splits` to determine the optimal
    sequence for decomposing each community, considering modularity as the primary metric.

    :param G: The networkx graph to be decomposed.
    :type G: nx.Graph
    :param community_size_threshold: The maximum allowable size for communities. Communities larger than this will be
                                     considered for further decomposition.
    :type community_size_threshold: int

    :return: A list of tuples, where each tuple represents the communities at a particular decomposition step.
    :rtype: List[Tuple[Set]]

    :raises ValueError: If there's a mismatch between returned community types and expected types.

    Example:
        Given a graph `G` with certain structures and a community_size_threshold of 20, the function might return a
        hierarchy of communities where each level reflects the decomposition at a specific step.

    Note:
        This function relies heavily on the `order_splits` function to determine the optimal split at each decomposition
        step.
    """
    louvain_hierarchy: List[Tuple[Set]] = []

    # Load in initial unified group
    initial_louvain_partition = [frozenset(s) for s in generate_partitions_list_from_dict(community_louvain.best_partition(G))]
    if len(G.nodes) < community_size_threshold or (len(initial_louvain_partition) <= 1):
        print("Graph passed in cannot be divided further")
        return

    louvain_hierarchy.extend(order_splits(G, initial_louvain_partition, partition_size_threshold=partition_size_threshold))

    while True:
        # Check that are communities that can be partitioned by Louvain and that are larger than the size threshold
        eligible_communities = []
        for comm in louvain_hierarchy[-1]:
            H = nx.subgraph(G, [node for node in comm])
            if (len(comm) > community_size_threshold) and (len(generate_partitions_list_from_dict(community_louvain.best_partition(H, random_state=42))) > 1):
                eligible_communities.append(comm)

        if eligible_communities:
            community_to_split = max(eligible_communities, key=len)
            H = nx.subgraph(G, [node for node in community_to_split])

            partitions = [frozenset(s) for s in generate_partitions_list_from_dict(community_louvain.best_partition(H, random_state=42))]
            # region partitions type checking
            if not all(isinstance(part, frozenset) for part in partitions):
                raise TypeError("All items in 'partitions' must be of type 'frozenset'.")
            # endregion

            community_split_hierarchy = order_splits(H, partitions=partitions, partition_size_threshold=partition_size_threshold)
            # region community_split_hierarchy type checking
            if not all(isinstance(level, tuple) and all(isinstance(s, set) for s in level) for level in community_split_hierarchy):
                raise TypeError("Each item in 'community_split_hierarchy' must be a tuple of sets.")
            # endregion


            other_communities = [comm for comm in louvain_hierarchy[-1] if comm != community_to_split]
            # region other_communities type checking
            if not all(isinstance(comm, set) for comm in other_communities):
                raise TypeError("All items in 'other_communities' must be of type 'set'.")
            # endregion

            # Add other groups outside the subgraph to the splits
            splits_with_context = []
            for level in community_split_hierarchy:
                level_with_context = [s for s in level]
                # region level_with_context type checking
                if not all(isinstance(s, set) for s in level_with_context):
                    raise TypeError("All items to be extended in 'level_with_context' must be of type 'set'.")
                # endregion

                level_with_context.extend(other_communities)
                splits_with_context.append(tuple(level_with_context))
                # region splits_with_context type checking
                if not all(isinstance(item, tuple) and all(isinstance(sub_item, set) for sub_item in item) for item in splits_with_context):
                    raise TypeError(f"All items in 'splits_with_context' must be tuples of sets. {[sub_item for sub_item in [item for item in splits_with_context] if not isinstance(sub_item, set)][0]}")
                # endregion

            louvain_hierarchy.extend(splits_with_context)
        else:
            break
    return louvain_hierarchy