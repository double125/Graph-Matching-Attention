#    Copyright 2018 AimBrain Ltd.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import torch
from torch.autograd import Variable
from torch.utils.data import dataloader
from torch.utils.data import Dataset
import numpy as np


def normalize_adj(A, type="AD"):
    if type == "DAD":
        # d is  Degree of nodes A=A+I
        # L = D^-1/2 A D^-1/2
        A = A + np.eye(A.shape[0])  # A=A+I
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif type == "AD":
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        A = A + np.eye(A.shape[0])  # A=A+I
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        D = np.diag(D)
        G = D - A
    return G

def collate_fn(batch):
    # put question lengths in descending order so that we can use packed sequences later
    batch.sort(key=lambda x: x[-1], reverse=True)
    return dataloader.default_collate(batch)

def batch_to_cuda(batch, volatile=False):
    # moves dataset batch on GPU
    with torch.no_grad():
        q = Variable(batch[0], requires_grad=False).cuda()
        a = Variable(batch[1], requires_grad=False).cuda()
        n_votes = Variable(batch[2],  requires_grad=False).cuda()

        # visual graph
        vg_nodes = Variable(batch[4],  requires_grad=False).cuda()
        vg_edges = Variable(batch[5],  requires_grad=False).cuda()

        # question graph
        qg_nodes = Variable(batch[6], requires_grad=False).cuda()
        qg_edges = Variable(batch[7], requires_grad=False).cuda()

        qglen = Variable(batch[8], requires_grad=False).cuda()
        qlen = Variable(batch[-1], requires_grad=False).cuda()

        return q, a, n_votes, vg_nodes, vg_edges, qg_nodes, qg_edges, qglen, qlen


def save(model, optimizer, ep, epoch_loss, epoch_acc, dir, name):
    # saves model and optimizer state

    tbs = {
        'epoch': ep + 1,
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
        }
    torch.save(tbs, os.path.join(dir, name + '.pth.tar'))


def total_vqa_score(output_batch, n_votes_batch):
    # computes the total vqa score as assessed by the challenge

    vqa_score = 0
    #_, oix = output_batch.data.max(1)
    _, oix = output_batch.data.topk(3)
    for i, pred in enumerate(oix):
        count0 = n_votes_batch[i,pred[0]]
        vqa_score += min(count0.cpu().item()/3, 1)
    return vqa_score

def total_vqa_score_2(oix_batch, n_votes_batch):
    # computes the total vqa score as assessed by the challenge

    vqa_score = 0
    for i, pred in enumerate(oix_batch):
        count0 = n_votes_batch[i,pred]
        vqa_score += min(count0.cpu().item()/3, 1)
    return vqa_score
