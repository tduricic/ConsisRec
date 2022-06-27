import networkx as nx
from networkx.algorithms import bipartite
import community as community_louvain
import numpy as np
import random


def create_user_item_bipartite_graph(user_items_dict):
    users = set()
    items = set()
    edges = []

    for user_id in user_items_dict:
        for item_id in user_items_dict[user_id]:
            users.add('user_id_' + str(user_id))
            items.add('item_id_' + str(item_id))
            edges.append(('user_id_' + str(user_id), 'item_id_' + str(item_id)))

    B = nx.Graph()
    B.add_nodes_from(users, bipartite=0)
    B.add_nodes_from(items, bipartite=1)
    B.add_edges_from(edges)

    return B, users, items


def create_user_communities_interaction_dict(B, items, user_items_dict):
    projected_G = bipartite.projected_graph(B, items)
    item_community_dict_tmp = community_louvain.best_partition(projected_G)
    item_community_dict = {}
    for node_id in item_community_dict_tmp:
        item_id = int(node_id.replace('item_id_', ''))
        item_community_dict[item_id] = item_community_dict_tmp[node_id]
    community_lists = {}
    for key in item_community_dict:
        if item_community_dict[key] not in community_lists:
            community_lists[item_community_dict[key]] = []
            community_lists[item_community_dict[key]].append(key)
        else:
            community_lists[item_community_dict[key]].append(key)

    user_communities_interactions_dict = {}
    for userId in user_items_dict:
        if userId not in user_communities_interactions_dict:
            user_communities_interactions_dict[userId] = [0] * len(community_lists)
        for itemId in user_items_dict[userId]:
            user_communities_interactions_dict[userId][item_community_dict[itemId]] += 1

    return user_communities_interactions_dict, item_community_dict


def calculate_weighted_average_diversity(user_communities_interactions):
    user_community_vector = np.array(user_communities_interactions)
    user_diversity = np.sum(user_community_vector / np.max(user_community_vector)) / user_community_vector.shape[0]
    return user_diversity


# Compute entropy of label distribution
def entropy_label_distribution(communities):
    n_communities = len(communities)
    if n_communities <= 1:
        return 0
    value, counts = np.unique(communities, return_counts=True)
    probs = counts / np.float32(n_communities)
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0.0
    # Compute entropy
    ent = 0.0
    for p in probs:
        ent -= p * np.log(p)
    return ent


def calculate_item_diversities(user_items_dict, item_community_dict):
    user_recommendation_diversity_dict = {}
    for user_id in user_items_dict:
        item_communities = [item_community_dict[item_id] for item_id in user_items_dict[user_id]]
        item_diversity = entropy_label_distribution(item_communities)
        user_recommendation_diversity_dict[user_id] = item_diversity
    return user_recommendation_diversity_dict


def create_user_item_rating_dict_from_file(user_item_ratings_filepath):
    user_item_rating_dict = {}
    with open(user_item_ratings_filepath) as fr:
        lines = []
        for line in fr.readlines():
            tokens = line.split()
            user_id = int(tokens[0])
            item_id = int(tokens[1])
            rating = float(tokens[2])
            if user_id not in user_item_rating_dict:
                user_item_rating_dict[user_id] = {}
            user_item_rating_dict[user_id][item_id] = rating
    return user_item_rating_dict


def create_history_u_lists(user_item_ratings_dict):
    history_u_lists = {}
    for user_id in user_item_ratings_dict:
        history_u_lists[user_id] = list(user_item_ratings_dict[user_id].keys())
    return history_u_lists


def create_history_ur_lists(user_item_ratings_dict):
    history_ur_lists = {}
    for user_id in user_item_ratings_dict:
        history_ur_lists[user_id] = list(user_item_ratings_dict[user_id].values())
    return history_ur_lists


def create_history_v_lists(user_item_ratings_dict):
    history_v_lists = {}
    for user_id in user_item_ratings_dict:
        for item_id in user_item_ratings_dict[user_id]:
            if item_id not in history_v_lists:
                history_v_lists[item_id] = []
            history_v_lists[item_id].append(user_id)
    return history_v_lists


def create_history_vr_lists(user_item_ratings_dict):
    history_vr_lists = {}
    for user_id in user_item_ratings_dict:
        for item_id in user_item_ratings_dict[user_id]:
            if item_id not in history_vr_lists:
                history_vr_lists[item_id] = []
            history_vr_lists[item_id].append(user_item_ratings_dict[user_id][item_id])
    return history_vr_lists


def create_uvr(user_item_ratings_dict):
    uvr_list = []
    u_list = []
    v_list = []
    r_list = []
    for user_id in user_item_ratings_dict:
        for item_id in user_item_ratings_dict[user_id]:
            uvr_list.append((user_id, item_id, user_item_ratings_dict[user_id][item_id]))
    random.shuffle(uvr_list)
    for (u, v, r) in uvr_list:
        u_list.append(u)
        v_list.append(v)
        r_list.append(r)
    return u_list, v_list, r_list


def create_social_adj_lists(social_connections_filepath):
    social_adj_lists = {}
    with open(social_connections_filepath) as f:
        for line in f:
            tokens = line.split('\t')
            user_1 = int(tokens[0])
            user_2 = int(tokens[1])
            weight = float(tokens[2])

            if user_1 not in social_adj_lists:
                social_adj_lists[user_1] = set()
            if user_2 not in social_adj_lists:
                social_adj_lists[user_2] = set()

            social_adj_lists[user_1].add(user_2)
            social_adj_lists[user_2].add(user_1)

    return social_adj_lists


def create_ratings_list(ratings_list):
    return sorted(list(set(ratings_list)))


def build_item_adj_lists(history_v_lists):
    adj_lists = {}
    for key in history_v_lists.keys():
        adj_lists[key] = []
    for key in history_v_lists.keys():
        for key_temp in history_v_lists.keys():
            if key != key_temp and len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) != 0:
                if len(set(history_v_lists[key]) & set(history_v_lists[key_temp])) / len(set(history_v_lists[key]) | set(history_v_lists[key_temp])) > 0.5:
                    adj_lists[key].append(key_temp)
    return adj_lists


def preprocess_data_test(train_dict, test_dict):
    history_u_lists = create_history_u_lists(train_dict)
    history_ur_lists = create_history_ur_lists(train_dict)
    history_v_lists = create_history_v_lists(train_dict)
    history_vr_lists = create_history_vr_lists(train_dict)
    train_u, train_v, train_r = create_uvr(train_dict)
    test_u, test_v, test_r = create_uvr(test_dict)
    item_adj_lists = build_item_adj_lists(history_v_lists)
    ratings_list = create_ratings_list(train_r + test_r)

    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
           train_u, train_v, train_r, test_u, test_v, test_r, item_adj_lists, ratings_list


def preprocess_data_val(train_dict, test_dict, val_dict):
    history_u_lists = create_history_u_lists(train_dict)
    history_ur_lists = create_history_ur_lists(train_dict)
    history_v_lists = create_history_v_lists(train_dict)
    history_vr_lists = create_history_vr_lists(train_dict)
    train_u, train_v, train_r = create_uvr(train_dict)
    test_u, test_v, test_r = create_uvr(test_dict)
    val_u, val_v, val_r = create_uvr(val_dict)
    item_adj_lists = build_item_adj_lists(history_v_lists)
    ratings_list = create_ratings_list(train_r + test_r)

    return history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, \
           train_u, train_v, train_r, test_u, test_v, test_r, val_u, val_v, val_r, item_adj_lists, ratings_list
