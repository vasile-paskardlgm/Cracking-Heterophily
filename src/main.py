import argparse
import torch.nn.functional as F
import torch
import random
from torch import tensor
from network import training_GCN
import numpy as np
from dataset import load_nc_dataset
from os import path
from utils import eval_acc, memory_free, fix_seed, PREPROCESSPATH
import torch.nn as nn
# import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Coarsening GNN Training')
    parser.add_argument('--dataset', type=str, default='twitch-gamer', help='datasets.') # twitch-gamer, wiki, pokec
    # parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--early_stopping', type=int, default=50)
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--gpu', action='store_true')

    parser.add_argument('--hidden', type=int, default=125)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=5e-4)

    args = parser.parse_args()
    # path = "params/"
    # if not os.path.isdir(path):
    #     os.mkdir(path)

    device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'

    if not path.exists(f'{PREPROCESSPATH}{args.dataset}_{int(args.coarsening_ratio * 100)}percent.pt'):
        raise ValueError('Please first generate the coarsened data.')
    
    OMEGA = torch.load(f'{PREPROCESSPATH}omega_{args.dataset}.pt').to(device)
    
    coarsened_graph = torch.load(f'{PREPROCESSPATH}{args.dataset}_{int(args.coarsening_ratio * 100)}percent.pt')
    edge_index_prime, x_prime, partition = coarsened_graph[0], coarsened_graph[1], coarsened_graph[2]

    edge_index_prime = edge_index_prime.to(device)
    x_prime = x_prime.to(device)
    partition = partition.to(device)

    dataset = load_nc_dataset(args.dataset)

    fix_seed()
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
    valid_idx = split_idx['valid'].to(device)
    test_idx = split_idx['test'].to(device)

    label = dataset.label.to(device)
    N = dataset.graph['num_nodes']
    C = dataset.graph['num_classes']
    F = x_prime.shape[1]

    edge_index = dataset.graph['edge_index'].to(device)
    x = dataset.graph['node_feat'].to(device)

    del dataset, split_idx, coarsened_graph

    # GNN training via CTH method
    model = training_GCN(F, C, args.hidden, args.num_layers, args.dropout).to(device)
    optimizer = torch.optim.Adam([
        {
        'params': model.gcn.parameters(), 
        'weight_decay': args.wd, 'lr': args.lr
    },
        {
        'params': model.compensation.parameters(),
        'weight_decay': 5e-4, 'lr': 0.01
    }
    ])
    criterion = nn.NLLLoss()

    # training
    best_valid_acc = best_test_acc = 0
    stop = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_prime, edge_index_prime, OMEGA, partition)
        loss = criterion(out[train_idx], label.squeeze(1)[train_idx])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out = model(x, edge_index, OMEGA, partition) # valid and test on original graph

            train_acc = eval_acc(label[train_idx], out[train_idx])
            valid_acc = eval_acc(label[valid_idx], out[valid_idx])
            test_acc = eval_acc(label[test_idx], out[test_idx])

        if epoch % 10 == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Train: {100 * train_acc:.2f}%, '
                  f'Valid: {100 * valid_acc:.2f}%, '
                  f'Test: {100 * test_acc:.2f}%')
        
        if valid_acc > best_valid_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc
            stop = 0
        
        stop += 1

        if (stop > args.early_stopping) and (args.early_stopping > 0):
            break

    print(f'Final Testing Performance:{100*best_test_acc:.2f}%')