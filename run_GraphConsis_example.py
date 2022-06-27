import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from Node_Encoders import Node_Encoder
from Node_Aggregators import Node_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import sys
from os import path
from tqdm import tqdm
from utils import utils
from GraphConsis import GraphConsis

def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / 100, best_rmse, best_mae))
            running_loss = 0.0
    return 0


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))
    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    expected_rmse = sqrt(mean_squared_error(tmp_pred, target))
    mae = mean_absolute_error(tmp_pred, target)
    return expected_rmse, mae

def train_and_store_model(model, optimizer, epochs, device, train_loader, test_loader, dataset_name):
    """
    ## toy dataset
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: user's connected neighborhoods
    ratings_list: rating value from 0.5 to 4.0 (8 opinion embeddings)
    """

    best_rmse = 9999.0
    best_mae = 9999.0
    endure_count = 0

    for epoch in range(1, epochs + 1):

       train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae)
       expected_rmse, mae = test(model, device, test_loader)
       # please add the validation set to tune the hyper-parameters based on your datasets.

       if not os.path.exists('./checkpoint/' + dataset_name):
           os.makedirs('./checkpoint/' + dataset_name)
    # early stopping (no validation set in toy dataset)
       if best_rmse > expected_rmse:
           best_rmse = expected_rmse
           best_mae = mae
           endure_count = 0
           # best_model = copy.deepcopy(model)
           torch.save({
               'epoch': epoch,
               'model_state_dict': model.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
           }, './checkpoint/' + dataset_name + '/model.pt')
           # torch.save(best_model.state_dict(), './checkpoint/' + dataset_name + '/model')
       else:
           endure_count += 1
       print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

       rmse_mae_dict = {
           'rmse' : expected_rmse,
           'mae' : mae
       }

       with open('./results/' + dataset_name + '/rmse_mae.pickle', 'wb') as handle:
           pickle.dump(rmse_mae_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

       if endure_count > 5:
           break


def get_top_k_recommendations(model, device, dataset_name, target_users, history_u_lists, history_v_lists, k, use_test_set_candidates, test_v):
    B, users, items = utils.create_user_item_bipartite_graph(history_u_lists)

    user_communities_interactions_dict_filepath = './results/' + dataset_name + '/user_communities_interactions_dict.pickle'
    item_community_dict_filepath = './results/' + dataset_name + '/item_community_dict.pickle'

    if path.exists('./results/' + dataset_name + '/user_communities_interactions_dict.pickle') and \
            path.exists('./results/' + dataset_name + '/item_community_dict.pickle'):
        user_communities_interactions_dict = pickle.load(user_communities_interactions_dict_filepath)
        item_community_dict = pickle.load(item_community_dict_filepath)
    else:
        user_communities_interactions_dict, item_community_dict = utils.create_user_communities_interaction_dict(B, items, history_u_lists)
        with open(user_communities_interactions_dict_filepath, 'wb') as handle:
            pickle.dump(user_communities_interactions_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(item_community_dict_filepath, 'wb') as handle:
            pickle.dump(item_community_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model.eval()
    all_items = list(set(history_v_lists.keys()))
    # {user_id:[item_id1, ..., item_idk]}
    results = {}
    with torch.no_grad():
        print('Generating recommendations...')
        for user_id in tqdm(target_users):
            if user_id not in history_u_lists:
                continue
            if use_test_set_candidates:
                candidate_items = [item_id for item_id in list(set(test_v)) if item_id not in history_u_lists[user_id]]
            else:
                candidate_items = [item_id for item_id in all_items if item_id not in history_u_lists[user_id]]
            u = torch.tensor(np.repeat(user_id, len(candidate_items))).to(device)
            v = torch.tensor(candidate_items).to(device)
            # multiply this with the mask of excluded recommendations derived from target_users_items
            val_output = model.forward(u, v).data.cpu().numpy()
            topk_prediction_indices = np.argpartition(val_output, -k)[-k:]
            topk_prediction_indices_sorted = list(np.flip(topk_prediction_indices[np.argsort(val_output[topk_prediction_indices])]))
            topk_item_ids = [candidate_items[i] for i in topk_prediction_indices_sorted]

            user_item_communities = [item_community_dict[item_id] for item_id in history_u_lists[user_id]]
            user_diversity = utils.entropy_label_distribution(user_item_communities)

            recommended_item_communities = [item_community_dict[item_id] for item_id in topk_item_ids]
            entropy_item_diversity = utils.entropy_label_distribution(recommended_item_communities)
            weighted_average_item_diversity = utils.calculate_weighted_average_diversity(user_communities_interactions_dict[user_id])

            results[user_id] = {
                'recommendations' : topk_item_ids,
                'user_diversity' : user_diversity,
                'entropy_item_diversity' : entropy_item_diversity,
                'weighted_average_item_diversity' : weighted_average_item_diversity,
            }

    return results


def evaluate_and_store_recommendations(model, device, dataset_name, train_u, test_u, history_u_lists, history_v_lists, k, use_test_set_candidates, test_v):
    # target_users = list(set(train_u + test_u))
    target_users = list(set(test_u))
    results = get_top_k_recommendations(model, device, dataset_name, target_users, history_u_lists, history_v_lists, k, use_test_set_candidates, test_v)

    return results


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Social Recommendation: GraphConsis model')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training')
    parser.add_argument('--percent', type=float, default=0.4, help='neighbor percent')
    parser.add_argument('--embed_dim', type=int, default=64, metavar='N', help='embedding size')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N', help='input batch size for testing')
    parser.add_argument('--epochs', type=int, default=20, metavar='N', help='number of epochs to train')
    parser.add_argument('--device', type=str, default='cuda', help='cpu or cuda')
    parser.add_argument('--gpu_id', type=str, default='2', metavar='N', help='gpu id')
    parser.add_argument('--dataset_name', type = str, default='ciao', help='dataset name')
    parser.add_argument('--k', type=int, default=10, metavar='N', help='number of recommendations to generate per user')
    parser.add_argument('--load_model', type=bool, default=True, help='Load from checkpoint or not')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--use_test_set_candidates', type=bool, default=True, help='if this is True, then the candidate items come only from the test set')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device(args.device)

    train_filepath = './data/' + args.dataset_name + '/train.tsv'
    test_filepath = './data/' + args.dataset_name + '/test.tsv'
    # val_filepath = './data/' + args.dataset_name + '/val.tsv'
    social_connections_filepath = './data/' + args.dataset_name + '/filtered_social_connections.tsv'
    train_dict = utils.create_user_item_rating_dict_from_file(train_filepath)
    test_dict = utils.create_user_item_rating_dict_from_file(test_filepath)
    # val_dict = utils.create_user_item_rating_dict_from_file(val_filepath)
    social_adj_lists = utils.create_social_adj_lists(social_connections_filepath)

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, \
    item_adj_lists, ratings_list = utils.preprocess_data_test(train_dict, test_dict)

    # history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, \
    # val_u, val_v, val_r, item_adj_lists, ratings_list = utils.preprocess_data_val(train_dict, test_dict, val_dict)



    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    # validset = torch.utils.data.TensorDataset(torch.LongTensor(valid_u), torch.LongTensor(valid_v),
    #                                           torch.FloatTensor(valid_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # valid_loader = torch.utils.data.DataLoader(validset, batch_size=args.test_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True)
    # num_users = history_u_lists.__len__()
    # num_items = history_v_lists.__len__()
    num_users = max(set(train_u + test_u)) + 1
    num_items = max(set(train_v + test_v)) + 1
    num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, args.embed_dim).to(device)
    v2e = nn.Embedding(num_items, args.embed_dim).to(device)
    r2e = nn.Embedding(num_ratings + 1, args.embed_dim).to(device)
    #node_feature
    node_agg = Node_Aggregator(v2e, r2e, u2e, args.embed_dim, r2e.num_embeddings - 1, cuda=device)
    node_enc = Node_Encoder(u2e, v2e, args.embed_dim, history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, social_adj_lists, item_adj_lists, node_agg, percent=args.percent,  cuda=device)

    # model
    model = GraphConsis(node_enc, r2e).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)

    # load from checkpoint
    if args.load_model is True:
        checkpoint = torch.load('./checkpoint/' + args.dataset_name + '/model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        train_and_store_model(model, optimizer, args.epochs, device, train_loader, test_loader, args.dataset_name)

    model.eval()

    results = evaluate_and_store_recommendations(model, device, args.dataset_name, train_u, test_u, history_u_lists, history_v_lists, args.k, args.use_test_set_candidates, test_v)

    with open('./results/' + args.dataset_name + '/recommendations.pickle', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    unique_recommended_items = []
    for user_id in results:
        unique_recommended_items += results[user_id]['recommendations']
    unique_recommended_items = set(unique_recommended_items)

    users_items_stats = {
        'num_users' : num_users,
        'num_items' : num_items,
        'num_recommended_items' : len(unique_recommended_items),
        'item_coverage' : round(len(unique_recommended_items)/num_items, 2)
    }

    with open('./results/' + args.dataset_name + '/users_items_stats.pickle', 'wb') as handle:
        pickle.dump(users_items_stats, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    main()
