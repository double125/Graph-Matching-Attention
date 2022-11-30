import torch
import numpy as np
import datetime

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from GM_module.gconv import Siamese_Gconv
from GM_module.affinity_layer import Affinity


class GraphMatch(Module):
    """
    This module is used to match two graph(question graph and visual graph)
    """
    def __init__(self, gnn_layer,in_dim_g1, in_dim_g2, out_dim, K1, K2, neighbourhood_size, dropout):
        super(GraphMatch, self).__init__()
        self.gnn_layer = gnn_layer
        self.dropout = nn.Dropout(p=dropout)
        self.in_dim_g1 = in_dim_g1
        self.in_dim_g2 = in_dim_g2
        self.out_dim = out_dim

        self.fc1 = nn.Linear(in_dim_g1,out_dim)
        self.fc2 = nn.Linear(in_dim_g2,out_dim)

        for i in range(self.gnn_layer):
            self.add_module('GE_{}'.format(i), Siamese_Gconv(self.out_dim, self.out_dim))
            self.add_module('SIM_{}'.format(i), Affinity(self.out_dim))
            self.add_module('GA_{}'.format(i), Siamese_Gconv(self.out_dim, self.out_dim))
            self.add_module('GMA_{}'.format(i), Affinity(self.out_dim))
            self.add_module('CG_{}'.format(i), nn.Linear(self.out_dim * 2, self.out_dim))

    def forward(self, vg_nodes, vg_edges, vg_mask, qg_nodes, qg_edges, qg_mask, mask_SA=None):

        vg_nodes = self.fc1(vg_nodes)
        qg_nodes = self.fc2(qg_nodes)

        for i in range(self.gnn_layer):
            GE = getattr(self, 'GE_{}'.format(i))
            SIM = getattr(self, 'SIM_{}'.format(i))
            GA = getattr(self, 'GA_{}'.format(i))
            GMA = getattr(self, 'GMA_{}'.format(i))
            cross_graph = getattr(self, 'CG_{}'.format(i))

            # Graph Encoder layer
            vg_nodes, qg_nodes = GE([vg_edges, vg_nodes, vg_mask], [qg_edges, qg_nodes, qg_mask])
            vg_A = SIM(vg_nodes, vg_nodes)
            qg_A = SIM(qg_nodes, qg_nodes)
            vg_nodes, qg_nodes = GA([vg_A, vg_nodes, vg_mask], [qg_A, qg_nodes, qg_mask])
            # Graph match Attention layer
            emb1 = vg_nodes
            emb2 = qg_nodes
            s = GMA(vg_nodes, qg_nodes).type(torch.cuda.FloatTensor)
            ## soft2
            if vg_mask is not None:
                s1 = s.masked_fill(qg_mask, -1e9)
            if qg_mask is not None:
                s2 = s.transpose(1,2).masked_fill(vg_mask, -1e9)
            #print('s1:', s1.shape, s1)
            #print('s2:', s2.shape, s2)
            s1 = F.softmax(s1,dim=-1)
            s2 = F.softmax(s2,dim=-1)
            vg_nodes = cross_graph(torch.cat((emb1, torch.bmm(s1, emb2)), dim=-1))
            qg_nodes = cross_graph(torch.cat((emb2, torch.bmm(s2, emb1)), dim=-1))
        return vg_nodes, qg_nodes

class GraphLearner(Module):
    def __init__(self, in_feature_dim, combined_feature_dim, dropout=0.1):
        super(GraphLearner, self).__init__()

        '''
        ## Variables:
        - in_feature_dim: dimensionality of input features
        - combined_feature_dim: dimensionality of the joint hidden embedding
        - K: number of graph nodes/objects on the image
        '''

        # Parameters
        self.in_dim = in_feature_dim
        self.combined_dim = combined_feature_dim

        # Embedding layers
        self.edge_layer_1 = nn.Linear(in_feature_dim,
                                      combined_feature_dim)
        self.edge_layer_2 = nn.Linear(in_feature_dim,
                                      combined_feature_dim)

        # Regularisation
        self.dropout = nn.Dropout(p=dropout)
        self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)

    def forward(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        # layer 1: Query
        h1 = self.edge_layer_1(graph_nodes)
        h1 = self.dropout(F.relu(h1))

        # layer 2: Key
        h2 = self.edge_layer_2(graph_nodes)
        h2 = self.dropout(F.relu(h2))

        # outer product
        adjacency_matrix = torch.matmul(h1, h2.transpose(1, 2))
        #adjacency_matrix = torch.sigmoid(adjacency_matrix)
        return adjacency_matrix

