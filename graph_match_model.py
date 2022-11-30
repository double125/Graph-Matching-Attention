from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import numpy as np
from time import *
import datetime
from collections import Counter

from layers import GraphLearner
from layers import GraphMatch as GM
from GM_module.affinity_layer import Affinity 
from GM_module.voting_layer import Voting

class Model(nn.Module):

    def __init__(self,
                 q_vocab_size,
                 K_vg,
                 K_qg,
                 vg_nodes_dim,
                 qg_nodes_dim,
                 emb_dim,
                 feat_dim,
                 hid_dim,
                 out_dim,
                 pretrained_wemb_q,
                 dropout,
                 question_emb,
                 neighbourhood_size=4
                 ):

        '''
        ## Variables:
        - vocab_size: dimensionality of input vocabulary
        - *gnode_dim :dimensionality of input graph nodes
        - emb_dim: question embedding size
        - feat_dim: dimensionality of input image features
        - out_dim: dimensionality of the output
        - pretrained_wemb_* : question or text dict
        - dropout: dropout probability
        - n_kernels : number of Gaussian kernels for convolutions
        - bias: whether to add a bias to Gaussian kernels
        '''

        super(Model, self).__init__()

        # Set parameters
        self.q_vocab_size = q_vocab_size
        self.K_vg = K_vg
        self.K_qg = K_qg
        self.vg_nodes_dim = vg_nodes_dim
        self.qg_nodes_dim = qg_nodes_dim
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.neighbourhood_size = neighbourhood_size

        self.question_emb = question_emb

        # initialize question word embedding layer weight
        self.wembed_q = nn.Embedding(q_vocab_size, emb_dim)
        self.wembed_q.weight.data.copy_(torch.from_numpy(pretrained_wemb_q))

        # question encoding
        self.q_lstm = nn.GRU(input_size=emb_dim, hidden_size=hid_dim, bidirectional=True)
        # Graph Match for visual_graph and question graph 
        #self.gm_v_q = GM(gnn_layer=1, in_dim_g1=self.vg_nodes_dim+2048, in_dim_g2=self.qg_nodes_dim+2048, out_dim=2048, K1=self.K_vg, K2=self.K_qg, neighbourhood_size=self.neighbourhood_size, dropout=dropout)
        self.gm_v_q = GM(gnn_layer=2, in_dim_g1=self.vg_nodes_dim+2048, in_dim_g2=self.qg_nodes_dim+2048, out_dim=2048, K1=self.K_vg, K2=self.K_qg, neighbourhood_size=self.neighbourhood_size, dropout=dropout)

        # dropout layers
        self.dropout = nn.Dropout(p=0.5)

        # baseline
        self.dropout_v = nn.Dropout(p=0.25)
        self.dropout_w = nn.Dropout(p=0.3)
        self.dropout_q = nn.Dropout(p=0.25)

        # output classifier
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim * 2, out_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(out_dim, out_dim))

    def forward(self, question, vg_nodes, vg_edges, qg_nodes, qg_edges, qglen, qlen):
        '''
        ## Inputs:
        - question (batch_size, max_qlen): input tokenised question
        - image (batch_size, K, feat_dim): input image features
        - vg_boxes (batchsize, K, 4)
        # question graph  
        - qg_nodes(batch_size, K, 10): input question graph
        - qglen (batch_size,node_dim): words number of each question node
        ## text graph
        - tg_nodes (batchsize, 14, 10, 5) :input text graph
        - tg_boxes (batchsize, 14, 4)
        - tg_boxes_norm (batchsize, 14, 4)

        - tgn_len (batchsize, 14, 10)
        - tgn_scores (batchsize, 14,10,5)
        - tgn_num (batchsize,1): number of text graph's nodes

        - K (int): number of image features/objects in the image
        - qlen (batch_size): vector describing the length (in words) of each input question
        ## Returns:
        - logits (batch_size, out_dim)
        - matrix: graph match matrix
        '''
        K_vg = vg_nodes.shape[1]
        K_qg = qg_nodes.shape[1]

        # Make mask
        #lang_feat_mask = self.make_mask(question.unsqueeze(2))
        qg_mask = self.make_mask(qg_nodes) # (B, 1, 14)
        vg_mask = self.make_mask(vg_nodes) # (B, 1, 100)

        # apply dropout to image features
        vg_nodes = self.dropout_v(vg_nodes).type(torch.cuda.FloatTensor)   #[B, K_vg, 2052]

        #####################################################
        # Compute question Embedding
        emb_q = self.wembed_q(question) #(batchsize, 14, 300)
        emb_q = self.dropout_w(emb_q)
        packed_q = pack_padded_sequence(emb_q, qlen, batch_first=True, enforce_sorted=False)  # questions have variable lengths
        self.q_lstm.flatten_parameters()
        _, hid_q = self.q_lstm(packed_q)
        hid_q = self.dropout_q(hid_q)
        qenc = torch.cat((hid_q[0], hid_q[1]), dim=-1).unsqueeze(1) #(B, 1, 2048)
        qenc_repeat_vg = qenc.repeat(1, K_vg, 1)   #(B, K, 2048)
        qenc_repeat_qg = qenc.repeat(1, K_qg, 1)   #(B, K, 2048)

        #######################################################
        # Compute question graph encoding
        qg_nodes = qg_nodes.view(-1,10)  # [B * 14, 10]
        qglen = qglen.view(-1)           # [B * 14]
        sort_emb_qg = self.wembed_q(qg_nodes) # [B * 14, 10, 300]
        sort_emb_qg = self.dropout_w(sort_emb_qg)
        packed_qg_node = pack_padded_sequence(sort_emb_qg, qglen, batch_first=True,enforce_sorted=False)
        _, hid_qg = self.q_lstm(packed_qg_node) # [ 1, B*14, 1024]
        hid_qg = self.dropout_q(hid_qg)
        qgenc_nodes = torch.cat((hid_qg[0], hid_qg[1]), dim=-1).unsqueeze(1)
        qgenc_nodes = qgenc_nodes.view(-1,K_qg,2048) # [B, 14, 2048]

        mask_SVQ = None

        ################# Graph Match Module ###########

        vg_nodes = torch.cat((vg_nodes, qenc_repeat_vg), dim=-1)
        qg_nodes = torch.cat((qgenc_nodes, qenc_repeat_qg), dim=-1)
        vg_nodes, qg_nodes = self.gm_v_q(vg_nodes, vg_edges, vg_mask, qg_nodes, qg_edges, qg_mask, mask_SVQ)

        # maxpooling /choose the max nodes features
        #all_nodes = torch.cat((vg_nodes, qg_nodes), dim=1)
        #final_feat, _ = torch.max(qg_nodes, dim=1)  #[B,1024]
        #final_feat, _ = torch.max(vg_nodes, dim=1)  #[B,1024]
        final_feat, _ = torch.max(qg_nodes, dim=1)  #[B,1024]

        h = F.relu(qenc).squeeze(1) * final_feat

################# output layer:classifier #####################

        # Output classifier
        hidden_1 = self.out_1(h)     #[B,3001]
        hidden_1 = F.relu(hidden_1)  #[B,3001]
        hidden_1 = self.dropout(hidden_1)  #[B,3001]
        logits = self.out_2(hidden_1)

        return logits

    # Masking
    def make_mask(self, feature):
        return (torch.sum(
              torch.abs(feature),
              dim=-1
              ) == 0).unsqueeze(1)
