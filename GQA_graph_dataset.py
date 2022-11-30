from __future__ import absolute_import, division, print_function

import os
import json
import numpy as np
import zarr
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import dataloader

try:
    import cPickle as pickle
except:
    import pickle as pickle

class GQA_Graph_Dataset(Dataset):

    def __init__(self, data_dir, emb_dim=300, data_balanced=True, train=True):

        # Set parameters
        self.data_dir = data_dir  # directory where the data is stored
        self.data_balanced = data_balanced #dataset balanced or not
        self.emb_dim = emb_dim    # question embedding dimension
        self.train = train        # train (True) or eval (False) mode
        self.seqlen = 14          # maximum question sequence length
        self.tglen = 14

        if data_balanced:
             # Load training question dictionary
            q_dict = pickle.load(
                open(os.path.join(data_dir, 'question_graph/balanced_train_q_dict.p'), 'rb'))
            self.q_itow = q_dict['itow']
            self.q_wtoi = q_dict['wtoi']
            self.q_words = len(self.q_itow) + 1

            # Load training answer dictionary
            a_dict = pickle.load(
                open(os.path.join(data_dir, 'question_graph/balanced_train_a_dict.p'), 'rb'))
            self.a_itow = a_dict['itow']
            self.a_wtoi = a_dict['wtoi']
            self.n_answers = len(self.a_itow) + 1
            print('balanced train_q/a_dict.p load over!')
        else:
            # Load training question dictionary
            q_dict = pickle.load(
                open(os.path.join(data_dir, 'question_graph/all_train_q_dict.p'), 'rb'))
            self.q_itow = q_dict['itow']
            self.q_wtoi = q_dict['wtoi']
            self.q_words = len(self.q_itow) + 1

            # Load training answer dictionary
            a_dict = pickle.load(
                open(os.path.join(data_dir, 'question_graph/all_train_a_dict.p'), 'rb'))
            self.a_itow = a_dict['itow']
            self.a_wtoi = a_dict['wtoi']
            self.n_answers = len(self.a_itow) + 1
            print('all train_q/a_dict.p load over!')

       # Load text_graphs word dictionary
        graph_dict = pickle.load(
            open(os.path.join(data_dir, 'gqa_text_graph_dict.p'), 'rb'))
        self.t_itow = graph_dict['itow']
        self.t_wtoi = graph_dict['wtoi']
        self.t_words = len(self.t_itow) + 1
        print('gqa_text_graph_dict.p load over!')

        # Load image features and bounding boxes
        print("     feature loading")
        self.i_feat = zarr.open(os.path.join(
            data_dir, 'objects.zarr'), mode='r')
        print("     boxes loading")
        self.bbox = zarr.open(os.path.join(
            data_dir, 'objects_boxes.zarr'), mode='r')
        print("     image_size loading")
        self.sizes = pd.read_csv(os.path.join(
            data_dir, 'objects_image_size.csv'))
        print('image features and bounding boxes load over!')

        # Load questions
        if data_balanced:
            if train:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_balanced_train_q_graph.json')))
            else:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_balanced_val_q_graph.json')))
        else:
            if train:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_all_train_q_graph.json')))
            else:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_all_val_q_graph.json')))

        self.n_questions = len(self.gqa)
        print("question loader over!")

        # Load text_graph
        self.text_graph = json.load(
                open(os.path.join(data_dir, 'gqa_text_graph_toked.json')))
        print("text_graph loader over!")

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
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype=np.float32)
            except:
                continue
            embeddings_index[word] = coefs
        f.close()

        ## init_question_embedding
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

        ## init_textgraph_embedding
        count = 0
        embedding_mat_t = np.zeros((self.t_words, emb_dim), dtype=np.float32)
        for word, i in self.t_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat_t[i] = embedding_v
            else:
                count += 1
        print('Unkonwn text graph words: ',count)
        self.pretrained_wemb_t = embedding_mat_t

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        qlen = len(self.gqa[idx]['question_toked'])
        q = [0] * 100
        for i, w in enumerate(self.gqa[idx]['question_toked']):
            try:
                q[i] = self.q_wtoi[w]
            except:
                q[i] = 0    # validation questions may contain unseen word

        # soft label answers
        a = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.gqa[idx]['answers_w_scores']:
            try:
                a[self.a_wtoi[w]] = c
            except:
                continue

        # number of votes for each answer
        n_votes = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.gqa[idx]['answers']:
            try:
                n_votes[self.a_wtoi[w]] = c
            except:
                continue

        # id of the question
        qid = int(self.gqa[idx]['question_id'])

        # image sample
        iid = self.gqa[idx]['image_id']
        img = self.i_feat[str(iid)]
        bboxes = np.asarray(self.bbox[str(iid)])
        imsize = self.sizes[str(iid)]

        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # number of image objects
        K = 100

        # question graph of question(rely on the nlp Dependency parser result )
        question_graph_nodes = self.gqa[idx]['question_parser_graph_nodes']
        qglen = np.ones(self.seqlen).astype(np.int)
        qg_nodes = np.zeros([self.seqlen, 10]).astype(np.int)

        for i,node_id in enumerate(question_graph_nodes):
            if i >= self.seqlen:
                break
            qglen[i] = min(len(question_graph_nodes[node_id]), 10)
            for j, w in enumerate(question_graph_nodes[node_id]):
                if j>= 10:
                    break
                try:
                    qg_nodes[i][j] = self.q_wtoi[w.lower()]
                except:
                    qg_nodes[i][j] = 0

        # text_graph of image
        tg_nodes = np.zeros([self.tglen,10,5]).astype(np.int)   # [14,10,5]
        tg_nodes_len = np.ones([self.tglen,10]).astype(np.int)  # [14,10]
        tg_boxes = np.zeros([self.tglen,4])                 # [14,4]
        tg_nodes_num = 14

        #for i,node_id in enumerate(self.text_graph[str(iid)]):
        #    #print(self.text_graph[iid][node_id]['node_pos'])
        #    if i>=k:
        #        break
        #    tg_boxes[i] = self.text_graph[iid][node_id]['node_pos']
        #    tglen[i] = len(self.text_graph[iid][node_id]['node_toked'])
        #    for j, w in enumerate(self.text_graph[iid][node_id]['node_toked']):
        #        if j>= 10:
        #            break
        #        try:
        #            tg_nodes[i][j] = self.g_wtoi[w]
        #        except:
        #            tg_nodes[i][j] = 0

        # scale bounding boxes by image dimensions
        for i in range(K):
            bbox = bboxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            bboxes[i] = bbox
        for i in range(self.tglen):
            bbox = tg_boxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            tg_boxes[i] = bbox

        # sort by the len(nodes),node which has more words placed first
        # text graph nodes sorted
        tg_nodes = np.asarray(tg_nodes)
        tg_boxes = np.asarray(tg_boxes)
        tg_nodes_len =np.asarray(tg_nodes_len)
        tg_nodes_num = np.asarray(tg_nodes_num)

        # question graph nodes sorted
        qg_nodes = np.asarray(qg_nodes)
        qglen = np.asarray(qglen)

        # format variables
        q = np.asarray(q)
        a = np.asarray(a).reshape(-1)
        n_votes = np.asarray(n_votes).reshape(-1)
        qid = np.asarray(qid).reshape(-1)
        i = np.concatenate([img, bboxes], axis=1)
        #tg = np.concatenate([tg_nodes, tg_boxes], axis=1)
        K = np.asarray(K).reshape(1)

        return q, a, n_votes, qid, i, qg_nodes, qglen, tg_nodes, tg_nodes_len, tg_nodes_num, K, qlen

class GQA_Graph_Dataset_Test(Dataset):

    def __init__(self, data_dir, emb_dim=300, data_balanced=True, train=True):
        self.data_dir = data_dir
        self.data_balanced = data_balanced #dataset balanced or not
        self.emb_dim = emb_dim
        self.train = train
        self.seqlen = 14 # hard set based on paper
        self.tglen = 14

        if data_balanced:
            # Load trainning question dictionary
            q_dict = pickle.load(
                    open(os.path.join(data_dir, 'question_graph/balanced_train_q_dict.p'),'rb'))
            self.q_itow = q_dict['itow']
            self.q_wtoi = q_dict['wtoi']
            self.q_words = len(self.q_itow) + 1

            # Load training answer dictionary
            a_dict = pickle.load(
                    open(os.path.join(data_dir, 'question_graph/balanced_train_a_dict.p'),'rb'))
            self.a_itow = a_dict['itow']
            self.a_wtoi = a_dict['wtoi']
            self.n_answers = len(self.a_itow) + 1
            print('balance train_q/a_dict.p load over!')
        else:
            # Load training question dictionary
            q_dict = pickle.load(
                open(os.path.join(data_dir, 'question_graph/all_train_q_dict.p'), 'rb'))
            self.q_itow = q_dict['itow']
            self.q_wtoi = q_dict['wtoi']
            self.q_words = len(self.q_itow) + 1

            # Load training answer dictionary
            a_dict = pickle.load(
                open(os.path.join(data_dir, 'question_graph/all_train_a_dict.p'), 'rb'))
            self.a_itow = a_dict['itow']
            self.a_wtoi = a_dict['wtoi']
            self.n_answers = len(self.a_itow) + 1
            print('all train_q/a_dict.p load over!')

        # Load text_graphs word dictionary
        graph_dict = pickle.load(
            open(os.path.join(data_dir, 'gqa_text_graph_dict.p'), 'rb'))
        self.t_itow = graph_dict['itow']
        self.t_wtoi = graph_dict['wtoi']
        self.t_words = len(self.t_itow) + 1
        print('gqa_text_graph_dict.p load over!')

        # Load image features and bounding boxes
        print("     feature loading")
        self.i_feat = zarr.open(os.path.join(
            data_dir, 'objects.zarr'), mode='r')
        print("     boxes loading")
        self.bbox = zarr.open(os.path.join(
            data_dir, 'objects_boxes.zarr'), mode='r')
        print("     image_size loading")
        self.sizes = pd.read_csv(os.path.join(
            data_dir, 'objects_image_size.csv'))
        print('image features and bounding boxes load over!')

        # Load questions
        if data_balanced:
            if train:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_balanced_train_q_graph.json'))) + \
                json.load(open(os.path.join(data_dir, 'question_graph/gqa_balanced_val_q_graph.json')))
            else:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_balanced_testdev_q_graph.json')))
                print("load gqa_balanced_testdev_q_graph.json")

        else:
            if train:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_all_train_q_graph.json')) + \
                            json.load(open(os.path.join(data_dir, 'question_graph_3/gqa_all_val_q_graph.json'))))
            else:
                self.gqa = json.load(
                    open(os.path.join(data_dir, 'question_graph/gqa_all_testdev_q_graph.json')))
                print("load gqa_all_testdev_q_graph.json")

        self.n_questions = len(self.gqa)
        print("question loader over!")

        # Load text_graph
        self.text_graph = json.load(
                 open(os.path.join(data_dir, 'gqa_text_graph_toked.json')))
        print("text_graph loader over!")

        self.feat_dim = self.i_feat[list(self.i_feat.keys())[
            0]].shape[1] + 4  # + bbox
        print('load feat_dim over!')
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

        ## init_question_embedding
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

        ## init_textgraph_embedding
        count = 0
        embedding_mat_t = np.zeros((self.t_words, emb_dim), dtype=np.float32)
        for word, i in self.t_wtoi.items():
            embedding_v = embeddings_index.get(word)
            if embedding_v is not None:
                embedding_mat_t[i] = embedding_v
            else:
                count += 1
        print('Unkonwn text graph words: ',count)
        self.pretrained_wemb_t = embedding_mat_t

    def __len__(self):
        return self.n_questions

    def __getitem__(self, idx):

        # question sample
        qlen = len(self.gqa[idx]['question_toked'])
        q = [0] * 100
        for i, w in enumerate(self.gqa[idx]['question_toked']):
            try:
                q[i] = self.q_wtoi[w]
            except:
                q[i] = 0    # validation questions may contain unseen word

        # soft label answers
        a = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.gqa[idx]['answers_w_scores']:
            try:
                a[self.a_wtoi[w]] = c
            except:
                continue
        a = np.asarray(a).reshape(-1)

        # votes
        n_votes = np.zeros(self.n_answers, dtype=np.float32)
        for w, c in self.gqa[idx]['answers']:
            try:
                n_votes[self.a_wtoi[w]] = c
            except:
                continue
        n_votes = np.asarray(n_votes).reshape(-1)

        # id of the question
        qid = int(self.gqa[idx]['question_id'])

        # image sample
        iid = self.gqa[idx]['image_id'] 
        img = self.i_feat[str(iid)]
        bboxes = np.asarray(self.bbox[str(iid)])
        imsize = self.sizes[str(iid)]

        if np.logical_not(np.isfinite(img)).sum() > 0:
            raise ValueError

        # k sample number of image objects
        K = 100

        # question graph of question(rely on the nlp Dependency parser result )
        question_graph_nodes = self.gqa[idx]['question_parser_graph_nodes']
        qglen = np.ones(self.seqlen).astype(np.int)
        qg_nodes = np.zeros([self.seqlen, 10]).astype(np.int)

        for i,node_id in enumerate(question_graph_nodes):
            if i >= self.seqlen:
                break
            qglen[i] = min(len(question_graph_nodes[node_id]), 10)
            for j, w in enumerate(question_graph_nodes[node_id]):
                if j >= 10:
                    break
                try:
                    qg_nodes[i][j] = self.q_wtoi[w.lower()]
                except:
                    qg_nodes[i][j] = 0

        # text_graph of image
        tg_nodes = np.zeros([self.tglen,10,5]).astype(np.int)   # [14,10,5]]
        tg_nodes_len = np.ones([self.tglen,10]).astype(np.int)  # [14,10]]
        tg_boxes = np.zeros([self.tglen,4])                 # [14,4]]
        tg_nodes_num = 14

        #for i,node_id in enumerate(self.text_graph[iid]):
        #    if i>=k:
        #        break
        #    tg_boxes[i] = self.text_graph[iid][node_id]['node_pos']
        #    tglen[i] = len(self.text_graph[iid][node_id]['node_toked'])
        #    for j, w in enumerate(self.text_graph[iid][node_id]['node_toked']):
        #        if j>=10:
        #            break
        #        try:
        #            tg_nodes[i][j] = self.g_wtoi[w]
        #        except:
        #            tg_nodes[i][j] = 0

        ## scale bounding boxes by image dimensions
        for i in range(K):
            bbox = bboxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            bboxes[i] = bbox

        for i in range(self.seqlen):
            bbox = tg_boxes[i]
            bbox[0] /= imsize[0]
            bbox[1] /= imsize[1]
            bbox[2] /= imsize[0]
            bbox[3] /= imsize[1]
            tg_boxes[i] = bbox

        # sort by the len(nodes),node which has more words placed first
        # text graph nodes sorted
        tg_nodes = np.asarray(tg_nodes)
        tg_boxes = np.asarray(tg_boxes)
        tg_nodes_len =np.asarray(tg_nodes_len)
        tg_nodes_num = np.asarray(tg_nodes_num)

        # question graph nodes sorted
        qg_nodes = np.asarray(qg_nodes)
        qglen = np.asarray(qglen)

        # format variables
        q = np.asarray(q)
        a = np.asarray(a).reshape(-1)
        n_votes = np.asarray(n_votes).reshape(-1)
        qid = np.asarray(qid).reshape(-1)
        i = np.concatenate([img, bboxes], axis=1)
        #tg = np.concatenate([tg_nodes, tg_boxes], axis=1)
        K = np.asarray(K).reshape(1)

        return q, a, n_votes, qid, i, qg_nodes, qglen, tg_nodes, tg_nodes_len, tg_nodes_num, K, qlen
