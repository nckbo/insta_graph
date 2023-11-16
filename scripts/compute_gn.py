import networkx as nx
import pickle
import json
# from notebooks.constants import *

def load_ig_network(json_file, n_users=None, include_your_username: str = None):
    """
    Load Instagram following data from a JSON file and create a network graph.

    Parameters:
    - json_file (str): Path to the JSON file containing following data.
    - n_users (int, optional): Number of users to include in the graph. If not provided, all users from the file are included.

    Returns:
    - nx.Graph: Network graph representing the following relationships.
    """

    with open(json_file, 'r') as f:
        following_data = json.load(f)
    # print(following_data)

    # If n_users is provided, only take the first n users
    if n_users:
        following_data = following_data[:n_users]

    G = nx.Graph()
    if include_your_username:
        G.add_node(include_your_username, date_added=min([user['timestamp'] for user in following_data]))


    for user in following_data:
        if include_your_username:
            G.add_edge(include_your_username, user['username'])
        # Load in Mutual followers

        try:
            if (user['mutual_followers'] != ['Load Issue']) and (user['mutual_followers'] != []):
                # Add date added on instagram
                G.add_node(user['username'])
                G.nodes[user['username']]['date_added'] = user['timestamp']
                # print(len(G.nodes))
                for mf in user['mutual_followers']:

                    if mf['follow_status'] == 'Following':
                        # Create an edge between mutual follower and user
                        G.add_edge(mf['username'], user['username'])


        except KeyError:  # Handle cases where expected keys might not be present in the data
            continue

    return G

def main():
    print('starting')
    G = load_ig_network('../data/processed/following_data_processed_real_users.json', include_your_username='nboveri')
    comp = list(nx.algorithms.community.girvan_newman(G))

    with open('../data/processed/girvan_newman_hierarchy_real_users.pkl', 'wb') as file:
        pickle.dump(comp, file)
    print('done')

if __name__ == "__main__":
    main()
