import networkx as nx
import json
from typing import *

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

def get_mutual_followers(username, following_data):
    """
    Returns mutual followers for a specific username.

    :param username: The username to find mutual followers for.
    :param following_data: List of dicts with followers data.
    :return: List of mutual followers if present and not a load issue, else empty.
    """
    for user in following_data:
        if (user['username'] == username) and (user['mutual_followers'] != ['Load Issue']):
            return [mf for mf in user['mutual_followers'] if (mf['follow_status'] == 'Following')]
    return []

def get_mutual_following(username, following_data):
    """
    Retrieves a list of users that a given username is following.

    :param username: Username to check against mutual followers.
    :param following_data: List of dicts containing user following data.
    :return: List of dicts with username and name for each following user.
    """
    following = []
    for user in following_data:
        # Check if 'mutual_followers' key exists and has a valid list that isn't a load issue
        if ('mutual_followers' in user and user['mutual_followers'] and
                user['mutual_followers'] != ['Load Issue']):
            # Iterate through the mutual followers
            for mf in user['mutual_followers']:
                if mf['username'] == username:
                    following.append({
                        'username': mf['username'],
                        'name': mf['name']
                    })
    return following

def remove_users(users_to_remove: List[str], following_data: List[Dict[Any,Any]])  -> List[Dict[Any,Any]]:

    users = [user for user in following_data if user['username'] not in users_to_remove]
    for user in users:
        if ('mutual_followers' in user) and (user['mutual_followers'] != 'Load Issue'):
            user['mutual_followers'] = [mf for mf in user['mutual_followers'] if mf not in users_to_remove]
    return users

