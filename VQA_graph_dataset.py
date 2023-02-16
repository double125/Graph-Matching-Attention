from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np
import zarr
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import dataloader
from utils import normalize_adj

try:
    import cPickle as pickle
except:
    import pickle as pickle


class VQA_Graph_Dataset(Dataset):

    def __init__(self, data_dir, emb_dim=300, train=True):

        # Set parameters
        self.data_dir = data_dir  # directory where the data is stored
        self.emb_dim = emb_dim    # question embedding dimension
        self.train = train        # train (True) or eval (False) mode
        self.seqlen = 14          # maximum question sequence length

        # Load training question dictionary
        q_dict = pickle.load(
            open(os.path.join(data_dir, 'question_graph/train_data/vqa_trainval_q_dict.p'), 'rb'))

        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        # Load training answer dictionary
        a_dict = pickle.load(
            open(os.path.join(data_dir, 'question_graph/train_data/vqa_trainval_a_dict.p'), 'rb'))

        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1


        # Load image features and bounding boxes
        self.i_feat = zarr.open(os.path.join(
            data_dir, 'vg_100/trainval.zarr'), mode='r')
        self.bbox = zarr.open(os.path.join(
            data_dir, 'vg_100/trainval_boxes.zarr'), mode='r')
        self.sizes = pd.read_csv(os.path.join(
            data_dir, 'vg_100/trainval_image_size.csv'))
        self.visual_graph = pickle.load(
                open(os.path.join(data_dir, 'visual_graph/vg_100/trainval_visual_graph_1_03.pkl'), 'rb'))

        # Load questions
        if train:
            self.vqa = json.load(
                open(os.path.join(data_dir, 'question_graph/train_data/vqa_train_q_graph.json')))
        else:
            self.vqa = json.load(
                open(os.path.join(data_dir, 'question_graph/train_data/vqa_val_q_graph.json')))

        self.n_questions = len(self.vqa)

        print('Loading done')
        self.feat_dim = self.i_feat[list(self.i_feat.keys())[
            0]].shape[1] + 4  # + bbox
        self.init_pretrained_wemb(emb_dim)
        print('init_pretrained_wemb over!')

    def init_pretrained_wemb(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            #values = line.split()
            values = line.strip().split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype=np.float32)
            except:
                continue
            embeddings_index[word] = coefs
        f.close()

        # question embedding dict
        count = 0
        embedding_mat_q = np.zeros((self.q_words, emb_dim), dtype=np.float32)
        for word, i in self.q_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat_q[i] = embedding_v
            else:
                count += 1
        print('Unkonwn question words: ',count)
        self.pretrained_wemb_q = embedding_mat_q

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        qlen = len(self.vqa[idx]['question_toked'])
        q = [0] * 100
        for i, w in enumerate(self.vqa[idx]['question_toked']):
            try:
                q[i] = self.q_wtoi[w.lower()]
            except:
                q[i] = 0    # validation questions may contain unseen word

        # soft label answers
        a = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.vqa[idx]['answers_w_scores']:
            try:
                a[self.a_wtoi[w]] = c
            except:
                continue

        # number of votes for each answer
        n_votes = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.vqa[idx]['answers']:
            try:
                n_votes[self.a_wtoi[w]] = c
            except:
                continue

        # id of the question
        qid = self.vqa[idx]['question_id']

        # image sample
        iid = self.vqa[idx]['image_id']
        img = self.i_feat[str(iid)]
        vg_boxes = np.asarray(self.bbox[str(iid)])
        vg_edges = np.asarray(self.visual_graph[str(iid)])
        imsize = self.sizes[str(iid)]

        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # number of image objects
        K = img.shape[0]

        # question graph of question(rely on the nlp Dependency parser result )
        #question_token = self.vqa[idx]['question_toked']
        question_graph_nodes = self.vqa[idx]['question_parser_graph_nodes']

        qglen = np.ones(self.seqlen).astype(np.int)
        qg_nodes = np.zeros([self.seqlen, 10]).astype(np.int)

        for i,node_id in enumerate(question_graph_nodes):
            if i>= self.seqlen:
                break
            qglen[int(node_id)] = min(len(question_graph_nodes[node_id]), 10)
            for j, w in enumerate(question_graph_nodes[node_id]):
                if j>=10:
                    break
                try:
                    qg_nodes[int(node_id)][j] = self.q_wtoi[w.lower()]
                except:
                    qg_nodes[int(node_id)][j] = 0

        qg_edges = np.asarray(self.vqa[idx]['question_A_Matrix'])
        edges_num = qg_edges.shape[0]
        if edges_num >= 14:
            qg_edges = qg_edges[:14,:14]
        else:
            qg_edges_pad1 = np.zeros([edges_num, 14-edges_num])
            qg_edges_pad2 = np.zeros([14-edges_num, 14])
            qg_edges = np.concatenate([qg_edges, qg_edges_pad1], axis=1)
            qg_edges = np.concatenate([qg_edges, qg_edges_pad2], axis=0)

        # scale bounding boxes by image dimensions
        for i in range(K):
            bbox = vg_boxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            vg_boxes[i] = bbox

        qg_edges = normalize_adj(qg_edges,'DAD')
        vg_edges = normalize_adj(vg_edges,'DAD')

        # question graph nodes not sorted
        qg_nodes = np.asarray(qg_nodes)
        qg_edges = np.asarray(qg_edges).astype(np.float)
        qglen = np.asarray(qglen)

        # format variables
        q = np.asarray(q)
        a = np.asarray(a).reshape(-1)
        n_votes = np.asarray(n_votes).reshape(-1)
        qid = np.asarray(qid).reshape(-1)

        # padding vg feat
        vg_pad = np.zeros([100-K, self.feat_dim])
        i = np.concatenate([img, vg_boxes], axis=1)
        vg_nodes = np.concatenate([i, vg_pad], axis=0)
        vg_edges = np.asarray(vg_edges).astype(np.float)

        return q, a, n_votes, qid, vg_nodes, vg_edges, qg_nodes, qg_edges, qglen, qlen

class VQA_Graph_Dataset_Test(Dataset):

    def __init__(self, data_dir, emb_dim=300, train=True):
        self.data_dir = data_dir
        self.emb_dim = emb_dim
        self.train = train
        self.seqlen = 14    # hard set based on paper

        q_dict = pickle.load(
            open(os.path.join(data_dir, 'question_graph/trainval_data/vqa_all_aug_q_dict.p'), 'rb'))

        self.q_itow = q_dict['itow']
        self.q_wtoi = q_dict['wtoi']
        self.q_words = len(self.q_itow) + 1

        a_dict = pickle.load(
            open(os.path.join(data_dir, 'question_graph/train_data/vqa_trainval_a_dict.p'), 'rb'))
        self.a_itow = a_dict['itow']
        self.a_wtoi = a_dict['wtoi']
        self.n_answers = len(self.a_itow) + 1

        if train:
            # Data augment with Visual Genome
            self.vqa = json.load(open(os.path.join(data_dir, 'question_graph/trainval_data/vqa_trainval_q_graph.json'))) + json.load(open(os.path.join(data_dir, 'question_graph/trainval_data/vg_aug_train_q_graph.json'))) + json.load(open(os.path.join(data_dir, 'question_graph/trainval_data/vg_aug_val_q_graph.json')))

            # No data augment with VG
            #self.vqa = json.load(open(os.path.join(data_dir, 'question_graph/trainval_data/vqa_trainval_final_3129_q_graph.json')))

            self.i_feat = zarr.open(os.path.join(
                data_dir, 'vg_100/trainval.zarr'), mode='r')
            self.bbox = zarr.open(os.path.join(
                data_dir, 'vg_100/trainval_boxes.zarr'), mode='r')
            self.sizes = pd.read_csv(os.path.join(
                data_dir, 'vg_100/trainval_image_size.csv'))
            self.visual_graph = pickle.load(
                    open(os.path.join(data_dir, 'visual_graph/vg_100/trainval_visual_graph_1_03.pkl'), 'rb'))

        else:
            self.vqa = json.load(
                open(os.path.join(data_dir, 'question_graph/trainval_data/vqa_test_q_graph.json')))
                #open(os.path.join(data_dir, 'vqa_test_toked.json')))
            self.i_feat = zarr.open(os.path.join(
                data_dir, 'vg_100/test.zarr'), mode='r')
            self.bbox = zarr.open(os.path.join(
                data_dir, 'vg_100/test_boxes.zarr'), mode='r')
            self.sizes = pd.read_csv(os.path.join(
                data_dir, 'vg_100/test_image_size.csv'))
            self.visual_graph = pickle.load(
                    open(os.path.join(data_dir, 'visual_graph/vg_100/test_visual_graph_1_03.pkl'), 'rb'))

        self.n_questions = len(self.vqa)
        print("question loader over!")

        self.feat_dim = self.i_feat[list(self.i_feat.keys())[
            0]].shape[1] + 4  # + bbox
        self.init_pretrained_wemb(emb_dim)
        print('init_pretrained_wemb over!')
        print('Loading done')


    def init_pretrained_wemb(self, emb_dim):
        """
            From blog.keras.io
            Initialises words embeddings with pre-trained GLOVE embeddings
        """
        embeddings_index = {}
        f = open(os.path.join(self.data_dir, 'glove.6B.') +
                 str(emb_dim) + 'd.txt')
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype=np.float32)
            except:
                continue
            embeddings_index[word] = coefs
        f.close()

        # question embedding dict
        count = 0
        embedding_mat_q = np.zeros((self.q_words, emb_dim), dtype=np.float32)
        for word, i in self.q_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat_q[i] = embedding_v
            else:
                count += 1
        print('Unkonwn question words: ',count)
        self.pretrained_wemb_q = embedding_mat_q

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        qlen = len(self.vqa[idx]['question_toked'])
        q = [0] * 100
        for i, w in enumerate(self.vqa[idx]['question_toked']):
            try:
                q[i] = self.q_wtoi[w.lower()]
            except:
                q[i] = 0    # validation questions may contain unseen word

        # soft label answers
        if self.train:
            a = np.zeros(self.n_answers, dtype=np.float32)
            for w, c in self.vqa[idx]['answers_w_scores']:
                try:
                    a[self.a_wtoi[w]] = c
                except:
                    continue
            a = np.asarray(a).reshape(-1)
        else:
            # return 0's for unknown test set answers
            a = 0

        # votes
        if self.train:
            n_votes = np.zeros(self.n_answers, dtype=np.float32)
            for w, c in self.vqa[idx]['answers']:
                try:
                    n_votes[self.a_wtoi[w]] = c
                except:
                    continue
            n_votes = np.asarray(n_votes).reshape(-1)
        else:
            # return 0's for unknown test set answers
            n_votes = 0

        # id of the question
        qid = self.vqa[idx]['question_id']

        # image sample
        iid = self.vqa[idx]['image_id']
        img = self.i_feat[str(iid)]
        vg_boxes = np.asarray(self.bbox[str(iid)])
        vg_edges = np.asarray(self.visual_graph[str(iid)])
        imsize = self.sizes[str(iid)]

        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError


        # number of image objects
        #K = 36
        K = img.shape[0]

        # question graph of question(rely on the nlp Dependency parser result )
        #question_token = self.vqa[idx]['question_toked']
        question_graph_nodes = self.vqa[idx]['question_parser_graph_nodes']

        qglen = np.ones(self.seqlen).astype(np.int)
        qg_nodes = np.zeros([self.seqlen, 10]).astype(np.int)

        for i,node_id in enumerate(question_graph_nodes):
            if i>= self.seqlen:
                break
            qglen[int(node_id)] = min(len(question_graph_nodes[node_id]), 10)
            for j, w in enumerate(question_graph_nodes[node_id]):
                if j>=10:
                    break
                try:
                    qg_nodes[int(node_id)][j] = self.q_wtoi[w.lower()]
                except:
                    qg_nodes[int(node_id)][j] = 0

        qg_edges = np.asarray(self.vqa[idx]['question_A_Matrix'])
        edges_num = qg_edges.shape[0]
        if edges_num >= 14:
            qg_edges = qg_edges[:14,:14]
        else:
            qg_edges_pad1 = np.zeros([edges_num, 14-edges_num])
            qg_edges_pad2 = np.zeros([14-edges_num, 14])
            qg_edges = np.concatenate([qg_edges, qg_edges_pad1], axis=1)
            qg_edges = np.concatenate([qg_edges, qg_edges_pad2], axis=0)

        qg_edges = normalize_adj(qg_edges,'DAD')
        vg_edges = normalize_adj(vg_edges,'DAD')

        # scale bounding boxes by image dimensions
        for i in range(K):
            bbox = vg_boxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            vg_boxes[i] = bbox

        qg_nodes = np.asarray(qg_nodes)
        qg_edges = np.asarray(qg_edges).astype(np.float)
        qglen = np.asarray(qglen)

        # format variables
        q = np.asarray(q)
        a = np.asarray(a).reshape(-1)
        n_votes = np.asarray(n_votes).reshape(-1)
        qid = np.asarray(qid).reshape(-1)

        # padding vg feat
        vg_pad = np.zeros([100-K, self.feat_dim])
        i = np.concatenate([img, vg_boxes], axis=1)
        vg_nodes = np.concatenate([i, vg_pad], axis=0)
        vg_edges = np.asarray(vg_edges).astype(np.float)

        return q, a, n_votes, qid, vg_nodes, vg_edges, qg_nodes, qg_edges, qglen, qlen
