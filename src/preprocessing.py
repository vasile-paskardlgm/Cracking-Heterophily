import numpy as np
import torch
import argparse

import torch.nn.functional as F
from os import path
import scipy.sparse as sp
from torch_geometric.utils import to_undirected

from dataset import load_nc_dataset
from utils import sys_normalized_adjacency, sparse_mx_to_torch_sparse_tensor,\
    similarity_coarsening, memory_free, deepwalk_encoding, PREPROCESSPATH


if __name__ == '__main__':

    ### Parse args ###
    parser = argparse.ArgumentParser(description='Pre-Processing Pipeline')
    parser.add_argument('--dataset', type=str, default='twitch-gamer', help='datasets.') # twitch-gamer, wiki, pokec
    parser.add_argument('--M', type=int, default=5, help='propagation steps.')
    parser.add_argument('--coarsening_ratio', type=float, default=0.1)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'

    if not path.exists(f'{PREPROCESSPATH}{args.dataset}_{int(args.coarsening_ratio * 100)}percent.pt'):

        if not path.exists(f'{PREPROCESSPATH}omega_{args.dataset}.pt'):

            #Loading dataset
            dataset = load_nc_dataset(args.dataset)
            # print("Loading completed!")

            edge_index = dataset.graph['edge_index']
            N = dataset.graph['num_nodes']
            x = dataset.graph['node_feat']
            num_features = x.shape[1]
            # print("Node:", N, "Edge:", edge_index.shape)

            # print('Making the graph undirected')
            edge_index = to_undirected(edge_index)

            #Load edges and create adjacency
            row,col = edge_index
            row=row.numpy()
            col=col.numpy()
            adj_mat=sp.csr_matrix((np.ones(row.shape[0]),(row,col)),shape=(N,N))


            print("Get scaled laplacian matrix.")
            adj_mat = sys_normalized_adjacency(adj_mat)
            adj_mat = -1.0*adj_mat #\hat{L}=2/(L*lambda_max)-I, lambda_max=2, \hat{L}=L-I=-P
            adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat).to(device)

            #T_0(\hat{L})X
            T_0_feat = x.to(device)
            
            # del for free
            del dataset, row, col, x

            #T_1(\hat{L})X
            T_1_feat = torch.spmm(adj_mat,T_0_feat)

            # compute T_k(\hat{L})X
            print("Begining of iteration!")
            for i in range(1,args.M):
                #T_k(\hat{L})X
                T_2_feat = torch.spmm(adj_mat,T_1_feat)
                T_2_feat = 2*T_2_feat-T_0_feat
                T_0_feat, T_1_feat =T_1_feat, T_2_feat

            prop_feat = F.normalize(T_2_feat, p=1, dim=1).cpu()

            # del for free
            del adj_mat, T_0_feat, T_1_feat, T_2_feat
            memory_free()

            dw_encode = deepwalk_encoding(edge_index, num_features, device)

            OMEGA = torch.concat([prop_feat, dw_encode], dim=1)

            # del for free
            del prop_feat, dw_encode, edge_index
            memory_free()

            torch.save(OMEGA, f'{PREPROCESSPATH}omega_{args.dataset}.pt')
            print("Augmented feature omega for "+args.dataset+" has been successfully generated")

        else:
            OMEGA = torch.load(f'{PREPROCESSPATH}omega_{args.dataset}.pt')
            print("Augmented feature omega for "+args.dataset+" has been loaded")

        coarsened_data = similarity_coarsening(dataset, OMEGA, args.coarsening_ratio, device)
        torch.save(coarsened_data, f'{PREPROCESSPATH}{args.dataset}_{int(args.coarsening_ratio * 100)}percent.pt')
    
    else:
        print("The coarsened data for "+args.dataset+" has already existed. "
              +"Please move to next stage.")

