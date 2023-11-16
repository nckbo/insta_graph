import networkx as nx
import scipy
import networkx as nx
import numpy as np

def generate_linkage_matrix(hierarchy, G):
    """
    Geneate a linkage matrix Z from a hierarchy and a graph
    :param hierarchy: List of tuples of sets, where each tuple corresponds to one partition of one of the previous sets
    :param G: NetworkX graph corresponding to the partition
    :return: Linkage matrix Z
    """
    def count_leaf_sets(s):
        applicable_leaf_sets = []
        leaf_sets_copy = leaf_sets.copy()
        for item in s:
            for leaf in leaf_sets_copy:
                if item in leaf:
                    applicable_leaf_sets.append(leaf)
                    leaf_sets_copy.remove(leaf)
                    break  # break after the first match to move to the next item
        return len(applicable_leaf_sets)

    def flatten_and_count(s):
        count = 0
        for item in s:
            if isinstance(item, set):
                count += flatten_and_count(item)
            else:
                count += 1
        return count




    # Load partitions
    if len(hierarchy[0]) == 1:
        partitions = hierarchy
    elif len(hierarchy[0]) == 2:
        full_set = tuple(set(element for subset in hierarchy for s in subset for element in s))  # Add in fully merged set
        partitions = [(full_set,)]
        partitions.extend(hierarchy)
    else:
        print("Error: Pass in a list of partitions starting with either the first set or the first partition")

    # Compute the minimum and maximum modularity values over all partitions.
    min_modularity = min(nx.community.modularity(G, partition) for partition in partitions)
    max_modularity = max(nx.community.modularity(G, partition) for partition in partitions)


    # Variable initializations
    set_labels = {}
    Z = []
    count = 0
    prev_cluster_labels = []

    # Loop through last partition, where each set contains just one element
    leaf_sets = [frozenset(leaf) for leaf in hierarchy[-1]] # Get leaf sets
    for cluster in partitions[-1]:
        # Load set mappings into dictionary
        frozen_cluster = frozenset(cluster)
        set_labels[frozen_cluster] = count

        # Populate Previous node set labels for future use
        prev_cluster_labels.append(count)
        # print(count)
        # print(cluster)
        count += 1

    # Loop through partitions starting at second to last element
    for partition in partitions[-2::-1]:
        cluster_labels = []
        num_leaf_sets_in_new_cluster = -1
        num_elems_in_new_cluster = -1
        new_set_label = -1

        # Load the labels of the partition except for the new cluster
        for cluster in partition:
            # print(cluster)
            frozen_cluster = frozenset(cluster)
            try:
                cluster_labels.append(set_labels[frozen_cluster])
            except KeyError:
                # print(count)
                set_labels[frozen_cluster] = count
                count += 1
                new_set_label = set_labels[frozen_cluster]

                #Get the number of unique elements in a new cluster
                num_leaf_sets_in_new_cluster = count_leaf_sets(cluster)
                num_elems_in_new_cluster = flatten_and_count(cluster)

        # Subtract the labels of this partition from what we saw in last partition to find out which clusters were merged
        Z_row = list(set(prev_cluster_labels) - set(cluster_labels)) # Clusters that merged

        # Reset the previous cluster labels
        cluster_labels.append(new_set_label)
        prev_cluster_labels = cluster_labels

        # Load in min-max scaled modularity for distance
        modularity = nx.community.modularity(G, partition)
        if modularity:
            distance = float(num_elems_in_new_cluster)
        else:
            distance = float(len(G.nodes))
        Z_row.append(distance)

        # Append number of elements
        Z_row.append(num_leaf_sets_in_new_cluster)
        Z.append(Z_row)
        # print(Z_row)

    return Z, set_labels

# # Test the function
# G = nx.karate_club_graph()
# # Z_matrix = generate_linkage_matrix(G)
# Z_matrix = generate_linkage_matrix(G)
# # Z_matrix



class Node:
    def __init__(self, value):
        """
        Initialize a Node object.

        :param value: Value to be stored in the node.
        """
        self.value = value
        self.left = None
        self.right = None

    def isLeaf(self) -> bool:
        """
        Check if the current node is a leaf node (i.e., has no children).

        :return: True if the node is a leaf, otherwise False.
        """
        if (self.left == None) and (self.right == None):
            return True
        else:
            return False

    def print_tree_text(self):
        """
        Print the tree in a pre-order traversal manner.
        """
        if self:
            left_val = self.left.value if self.left else None
            right_val = self.right.value if self.right else None
            print(f"Node: {self.value}, \tLeft: {left_val}, \tRight: {right_val}")
            if self.left:
                self.left.print_tree_text()
            if self.right:
                self.right.print_tree_text()

def tree_to_linkage_order(node: Node) -> list:
    """
    Convert a binary tree to a list of tuples representing linkage orders.

    This function traverses the tree and constructs a list of tuples.
    Each tuple contains values of two nodes: a leaf node and another node
    from the subtree. This is useful for representing hierarchies or dendrograms.

    :param node: Root node of the tree.
    :return: List of tuples representing linkage orders.
    """
    mapping = []

    if node is None:
        return mapping

    if node.left.isLeaf():
        if node.right.isLeaf(): # Both leafs
            mapping.append((node.left.value, node.right.value))
            return mapping
        else: # left leaf, right subtree
            mapping.extend(tree_to_linkage_order(node.right))
            mapping.append((node.left.value, node.right.value))
            return mapping
    elif node.right.isLeaf(): #left subtree, right leaf
        mapping.extend(tree_to_linkage_order(node.left))
        mapping.append((node.left.value, node.right.value))
        return mapping
    else: #both subtrees
        mapping.extend(tree_to_linkage_order(node.left))
        mapping.extend(tree_to_linkage_order(node.right))
        mapping.append((node.left.value, node.right.value))
        return mapping


def map_linkage_pairs_to_labels(linkage_list, Z):
    """
    Map linkage pairs to corresponding labels in the linkage matrix.

    :param linkage_list: List of tuples containing linkage pairs.
    :param Z: Linkage matrix.
    :return: List of labels corresponding to linkage pairs.
    """
    labels = []

    # Add in label column
    for i, row in enumerate(Z):
        row.append(len(Z) + 1 + i)

    for pair in linkage_list:
        for row in Z:
            if ((row[0] == pair[0]) or (row[1] == pair[0])) and ((row[0] == pair[1]) or (row[1] == pair[1])):
                labels.append(row[4])
    return labels

def prune_to_levels(node: Node, max_depth: int, current_depth: int = 0):
    """
    Prune the binary tree to the specified depth level.

    :param node: Current node.
    :param max_depth: Maximum depth to prune to.
    :param current_depth: Current depth level (default to 0).
    """
    if not node or current_depth == (max_depth + 2):  # Adding + 2 to align with scipy's behavior
        return

    if current_depth == max_depth + 1:  # Adding + 1 to align with scipy's behavior
        node.left = None
        node.right = None
        return

    prune_to_levels(node.left, max_depth, current_depth + 1)
    prune_to_levels(node.right, max_depth, current_depth + 1)

def get_children(node_label, Z):
    """
    Retrieve children for a given node label from the linkage matrix.

    :param node_label: Label of the node.
    :param Z: Linkage matrix.
    :return: List of children nodes if present, else empty list.
    """
    for i, row in enumerate(Z):
        row.append(len(Z) + 1 + i)
    children = [[row[0], row[1]] for row in Z if (row[4] == node_label)]
    return children[0] if children else []

def get_leftmost_leaf_index(node_label, ordered_list, Z):
    """
    Get the index of the leftmost leaf node for a given node label.

    :param node_label: Label of the node.
    :param ordered_list: List of ordered nodes.
    :param Z: Linkage matrix.
    :return: Index of the leftmost leaf node.
    """
    if get_children(node_label, Z):
        return min([get_leftmost_leaf_index(label, ordered_list, Z) for label in get_children(node_label, Z)])
    else:
        return index_in_list(node_label, ordered_list)

def index_in_list(node_label, ordered_list):
    """
    Find the index of a given node label in an ordered list.

    :param node_label: Label of the node.
    :param ordered_list: List of ordered nodes.
    :return: Index of the node label in the list, if present.
    """
    for i, elem in enumerate(ordered_list):
        if elem == node_label:
            return i


def tree_from_Z(Z, ordered_list):
    """
    Construct a binary tree from a linkage matrix.

    :param Z: Linkage matrix.
    :param ordered_list: List of ordered nodes.
    :return: Root node of the constructed binary tree.
    """
    for i, row in enumerate(Z):
        row.append(len(Z) + 1 + i)
    root = Node(max([row[4] for row in Z]))

    def tree_from_Z_helper(Z, ordered_list, node):
        """
        Recursive helper function to construct the tree from a linkage matrix.

        :param Z: Linkage matrix.
        :param ordered_list: List of ordered nodes.
        :param node: Current node.
        :return: Current node after adding children.
        """
        if get_children(node.value, Z):
            child1 = get_children(node.value, Z)[0]
            child2 = get_children(node.value, Z)[1]
            if get_leftmost_leaf_index(child1, ordered_list, Z) < get_leftmost_leaf_index(child2, ordered_list, Z):
                node.left = Node(child1)
                node.right = Node(child2)
            else:
                node.right = Node(child1)
                node.left = Node(child2)

            tree_from_Z_helper(Z, ordered_list, node.left)
            tree_from_Z_helper(Z, ordered_list, node.right)
            return node

    return tree_from_Z_helper(Z, ordered_list, root)

def pruned_linkage_matrix_to_list(Z, p):
    """
    Convert a pruned linkage matrix to a list of linkage pairs.

    :param Z: Linkage matrix.
    :param p: Maximum depth for pruning.
    :return: List of linkage pairs after pruning.
    """
    import scipy.cluster.hierarchy

    R = scipy.cluster.hierarchy.dendrogram(Z=Z, no_plot=True)
    tree = tree_from_Z(Z, ordered_list=R['leaves'])
    prune_to_levels(tree, p)
    return map_linkage_pairs_to_labels(tree_to_linkage_order(tree), Z)


