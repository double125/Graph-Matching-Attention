from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import json
import argparse
from tqdm import tqdm
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import ExponentialLR

from graph_match_model import Model
from GQA_graph_dataset import *
from VQA_graph_dataset import *
from utils import *
from collections import Counter

def train(args):

    """
        Train a VQA or GQA  model using the training set
    """
    # set random seed
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires cuda')

    # Load the VQA training set
    print('Loading data')
    if args.data_type == 'VQA':
        dataset = VQA_Graph_Dataset(args.data_dir, args.emb)
        print('VQA train dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize,
                                    pin_memory=True,
                                    shuffle=True,
                                    num_workers=8,
                                    collate_fn=collate_fn)
        print('VQA train dataloader create over!')
        # Load the VQA validation set
        dataset_test = VQA_Graph_Dataset(args.data_dir, args.emb, train=False)
        print('VQA val dataset create over!')

        loader_test_val = DataLoader(dataset_test,
                                    batch_size=args.bsize,
                                    pin_memory=True,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=collate_fn)
        print('VQA val dataloader create over!')

        # Print data and model parameters
        print('Parameters:\n\t'
            'q_vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
            '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words,
                                                        args.emb,
                                                        dataset.feat_dim,
                                                        args.hid,
                                                        dataset.n_answers))
 

    if args.data_type == 'GQA':
        dataset = GQA_Graph_Dataset(args.data_dir, args.emb, args.data_balanced)
        print('GQA train dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize, 
                                    pin_memory=True,
                                    shuffle=True, 
                                    num_workers=6, 
                                    collate_fn=collate_fn)
        print('GQA train dataloader create over!')
        # Load the GQA validation set
        dataset_test = GQA_Graph_Dataset(args.data_dir, args.emb, args.data_balanced, train=False)
        print('GQA val dataset create over!')

        loader_test_val = DataLoader(dataset_test,
                                    batch_size=args.bsize,
                                    pin_memory=True,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=collate_fn) 
        print('GQA val dataloader create over!')

        # Print data and model parameters
        print('Parameters:\n\t'
                'q_vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
                '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words,
                                                          args.emb,
                                                          dataset.feat_dim,
                                                          args.hid,
                                                          dataset.n_answers))
    n_batches = len(dataset)//args.bsize
    print('n_batches:',n_batches)
    print('Initializing model')

    model = Model(q_vocab_size=dataset.q_words,
                    K_vg = 100,
                    K_qg = 14,
                    vg_nodes_dim=2052,
                    qg_nodes_dim=2048,
                    emb_dim=args.emb,
                    feat_dim=dataset.feat_dim,
                    hid_dim=args.hid,
                    out_dim=dataset.n_answers,
                    pretrained_wemb_q=dataset.pretrained_wemb_q,
                    dropout=args.dropout,
                    question_emb=args.question_emb,
                    )

    #criterion = nn.MultiLabelSoftMarginLoss()
    criterion = nn.BCEWithLogitsLoss()

    # Move it to multi GPU
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    criterion = criterion.cuda()

    # Define the optimiser
    lr_default = args.base_lr
    num_epochs = args.ep
    lr_decay_epochs = range(args.lr_decay_start, num_epochs,
                            args.lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,
                            1.5 * lr_default, 2.0 * lr_default]

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                model.parameters()),
                                lr=lr_default,
                                betas=(0.9, 0.999),
                                eps=1e-8,
                                weight_decay=args.weight_decay)

    # Continue training from saved model
    start_ep = 0
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
        ckpt = torch.load(args.model_path)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Train iterations
    print('Start training.')
    best_eval_score = 0
    best_eval_ep = 0
    last_eval_score, eval_score = 0, 0
    for ep in range(start_ep, start_ep+args.ep):

        #scheduler.step()
        ep_loss = 0.0
        ep_correct = 0.0
        ave_loss = 0.0
        ave_correct = 0.0
        losses = []

        if ep < len(gradual_warmup_steps):
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = gradual_warmup_steps[ep]
                print('gradual warmup lr: %.4f' %optimizer.param_groups[-1]['lr'])
        elif (ep in lr_decay_epochs or eval_score < last_eval_score and args.lr_decay_based_on_val):
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] *= args.lr_decay_rate
                print('decreased lr: %.4f' % optimizer.param_groups[-1]['lr'])
        else:
            print('lr: %.4f' % optimizer.param_groups[-1]['lr'])

        for step, next_batch in tqdm(enumerate(loader)):
            model.train()
            optimizer.zero_grad()
            # Move batch to cuda

            q_batch, a_batch, vote_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch = \
                batch_to_cuda(next_batch)

            # forward pass
            output = model(q_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch)

            loss = criterion(output, a_batch) * dataset.n_answers

            # Compute batch accuracy based on vqa evaluation
            correct = total_vqa_score(output, vote_batch)
            ep_correct += correct
            # pytorch version 0.3
            ep_loss += loss.item()
            ave_correct += correct
            ave_loss += loss.item()
            losses.append(loss.cpu().item())

            # This is a 40 step average
            if step % 40 == 0 and step != 0:
                print('  Epoch %02d(%03d/%03d), ave loss: %.7f, ave accuracy: %.2f%%' %
                      (ep+1, step, n_batches, ave_loss/40,
                       ave_correct * 100 / (args.bsize*40)))

                ave_correct = 0
                ave_loss = 0

            # Compute gradient and do optimisation step
            loss.backward()
            optimizer.step()
        # evaluation ep model

        # save model and compute accuracy for epoch
        epoch_loss = ep_loss / n_batches
        epoch_acc = ep_correct * 100 / (n_batches * args.bsize)

        if ep+1 >= 30:
            save(model, optimizer, ep, epoch_loss, epoch_acc,
                 dir=args.save_dir, name=args.name+'_'+str(ep+1))

            correct_eval = 0
            model.train(False)
            for _, test_batch in enumerate(loader_test_val):
                # move batch to cuda
                q_batch, a_batch, vote_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch = \
                    batch_to_cuda(test_batch, volatile=True)

                # get predictions
                output = model(q_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch)

                correct_eval += total_vqa_score(output, vote_batch)

            # compute and print average accuracy
            model.train(True)
            eval_score = correct_eval / dataset_test.n_questions*100

            last_eval_score = eval_score
            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                best_eval_ep = ep +1

        print('Epoch %02d done, average loss: %.3f, average accuracy: %.2f, average val_all_acc: %.2f%%' % (ep+1, epoch_loss, epoch_acc, eval_score))
        if ep+1 == args.ep:
            print('best_eval_ep: %02d, best_eval_score: %.2f' % (best_eval_ep, best_eval_score))

def eval_model(args):

    """
        Computes the VQA and GQA accuracy over the validation set
        using a pre-trained model
    """

    # Check that the model path is accurate
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
    else:
        raise SystemExit('Need to provide model path.')

    # Set random seed
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires cuda')

    # Load the VQA training set
    print('Loading data')
    if args.data_type == 'VQA':
        dataset = VQA_Graph_Dataset(args.data_dir, args.emb, train=False)
        print('VQA val dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize,
                                    pin_memory=True,
                                    shuffle=True, 
                                    num_workers=8, 
                                    collate_fn=collate_fn)
        print('VQA val dataloader create over!')

        # Print data and model parameters
        print('Parameters:\n\t'
            'q_vocab size:%d \n\tembedding dim: %d\n\tfeature dim: %d'
            '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words,
                                                        args.emb,
                                                        dataset.feat_dim,
                                                        args.hid,
                                                        dataset.n_answers))

    if args.data_type == 'GQA':
        dataset = GQA_Graph_Dataset(args.data_dir, args.emb, 
                                    args.data_balanced, train=False)
        print('GQA val dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize, 
                                    pin_memory=True,
                                    shuffle=True, 
                                    num_workers=16, 
                                    collate_fn=collate_fn)
        print('GQA val dataloader create over!')

        # Print data and model parameters
        print('Parameters:\n\t'
                'q_vocab size: %d\n\tg_vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
                '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words,
                                                          dataset.t_words,
                                                          args.emb,
                                                          dataset.feat_dim,
                                                          args.hid,
                                                          dataset.n_answers))
    n_batches = len(dataset)//args.bsize
    print('n_batches:',n_batches)

    print('Initializing model')
    if args.data_type == 'VQA':
        K_vg = 100
        K_qg = 14
    if args.data_type == 'GQA':
        K_vg = 100
        K_qg = 14

    model = Model(q_vocab_size=dataset.q_words,
                    K_vg = K_vg,
                    K_qg = K_qg,
                    vg_nodes_dim=2052,
                    qg_nodes_dim=2048,
                    emb_dim=args.emb,
                    feat_dim=dataset.feat_dim,
                    hid_dim=args.hid,
                    out_dim=dataset.n_answers,
                    pretrained_wemb_q=dataset.pretrained_wemb_q,
                    dropout=args.dropout,
                    late_fusion=args.fusion_way,
                    question_emb=args.question_emb,
                    neighbourhood_size=args.neighbourhood_size
                    )


    # move to CUDA
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Restore pre-trained model
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'])
    model.train(False)

    # Compute accuracy
    result = []
    correct = 0
    for step, next_batch in tqdm(enumerate(loader)):
        # move batch to cuda
        q_batch, a_batch, vote_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch = \
                batch_to_cuda(next_batch)

        # forward pass
        output = model(q_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch)

        qid_batch = next_batch[3]
        _, oix = output.data.max(1)
        # record predictions
        for i, qid in enumerate(qid_batch):
            if args.data_type == 'VQA':
                result.append({
                    'question_id': int(qid.numpy()),
                    'answer': dataset.a_itow[oix[i].item()]
                })
            if args.data_type == 'GQA':
                result.append({
                    "questionId": qid,
                    "prediction": dataset.a_itow[oix[i].item()]
                     })
        # compute batch accuracy
        correct += total_vqa_score_2(output, vote_batch)

    # compute and print average accuracy
    acc = correct/dataset.n_questions*100
    print("accuracy: {} %".format(acc))

    save_path = os.path.split(args.model_path)[0]
    filename = os.path.split(args.model_path)[1].split('.')[0] + '_result.json'
    # save predictions
    json.dump(result, open(os.path.join(save_path, filename), 'w'))
    print('Validation done')


def test(args):

    """
        Creates a result.json for predictions on
        the test set
    """
    # Check that the model path is accurate
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
    else:
        raise SystemExit('Need to provide model path.')

    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires CUDA')
 
    print('Loading data')
    if args.data_type == 'VQA':
        dataset = VQA_Graph_Dataset_Test(args.data_dir, args.emb, train=False)
        print('VQA Test dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize, 
                        shuffle=False, num_workers=8, 
                        collate_fn=collate_fn)
        print('VQA Test dataloader create over!')

        # Print data and model parameters
        print('Parameters:\n\t'
            'vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d' 
            '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words, args.emb,
                                                    dataset.feat_dim,
                                                    args.hid,
                                                    dataset.n_answers))
    if args.data_type == 'GQA':
        dataset = GQA_Graph_Dataset_Test(args.data_dir, args.emb,
                                     args.data_balanced, train=False)
        print('GQA Test dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize,
                            shuffle=False,
                            num_workers=8,
                            collate_fn=collate_fn)
        print('GQA Test dataloader create over!')

        # Print data and model parameters
        print('Parameters:\n\t'
                'q_vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
                '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words,
                    args.emb,
                    dataset.feat_dim,
                    args.hid,
                    ataset.n_answers))
    n_batches = len(dataset)//args.bsize
    print('n_batches:',n_batches)

    print('Initializing model')

    if args.data_type == 'VQA':
        K_vg = 100
        K_qg = 14
    if args.data_type == 'GQA':
        K_vg = 100
        K_qg = 14

    model = Model(q_vocab_size=dataset.q_words,
                    K_vg = K_vg,
                    K_qg = K_qg,
                    vg_nodes_dim=2052,
                    qg_nodes_dim=2048,
                    emb_dim=args.emb,
                    feat_dim=dataset.feat_dim,
                    hid_dim=args.hid,
                    out_dim=dataset.n_answers,
                    pretrained_wemb_q=dataset.pretrained_wemb_q,
                    dropout=args.dropout,
                    question_emb=args.question_emb,
                    neighbourhood_size=args.neighbourhood_size
                    )

    # move to CUDA
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Restore pre-trained model
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['state_dict'])
    model.train(False)

    result = []
    for step, next_batch in tqdm(enumerate(loader)):
        # Batch preparation

        q_batch, a_batch, vote_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch = \
                batch_to_cuda(next_batch)

        # forward pass
        output = model(q_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch)

        qid_batch = next_batch[3]
        _, oix = output.data.max(1)
        # record predictions
        for i, qid in enumerate(qid_batch):
            if args.data_type == 'VQA':
                result.append({
                    'question_id': int(qid.numpy()),
                    'answer': dataset.a_itow[oix[i].item()]
                })
            if args.data_type == 'GQA':
                result.append({
                    "questionId": qid,
                    "prediction": dataset.a_itow[oix[i].item()]
                     })
    save_path = os.path.split(args.model_path)[0]
    filename = os.path.split(args.model_path)[1].split('.')[0] + '_result.json'
    json.dump(result, open(os.path.join(save_path, filename), 'w'))
    print('Testing done')

def trainval(args):

    """
        Train a VQA model using the training + validation set
    """
    # set random seed
    torch.manual_seed(1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1000)
    else:
        raise SystemExit('No CUDA available, script requires cuda')

    # Load the VQA training set
    print('Loading data')
    if args.data_type == 'VQA':
        dataset = VQA_Graph_Dataset_Test(args.data_dir, args.emb)
        print('VQA trainval dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize,
                                    pin_memory=True,
                                    shuffle=True,
                                    num_workers=8,
                                    collate_fn=collate_fn)

        # Print data and model parameters
        print('Parameters:\n\t'
            'q_vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
            '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words,
                                                        args.emb,
                                                        dataset.feat_dim,
                                                        args.hid,
                                                        dataset.n_answers))
 

    if args.data_type == 'GQA':
        dataset = GQA_Graph_Dataset_Test(args.data_dir, args.emb, args.data_balanced)
        print('GQA trainval dataset create over!')
        loader = DataLoader(dataset, batch_size=args.bsize,
                                    pin_memory=True,
                                    shuffle=True,
                                    num_workers=8,
                                    collate_fn=collate_fn)

        # Print data and model parameters
        print('Parameters:\n\t'
                'q_vocab size: %d\n\tg_vocab size: %d\n\tembedding dim: %d\n\tfeature dim: %d'
                '\n\thidden dim: %d\n\toutput dim: %d' % (dataset.q_words,
                                                          args.emb,
                                                          dataset.feat_dim,
                                                          args.hid,
                                                          dataset.n_answers))
    n_batches = len(dataset)//args.bsize
    print('n_batches:',n_batches)
    print('Initializing model')

    if args.data_type == 'VQA':
        K_vg = 100
        K_qg = 14
    if args.data_type == 'GQA':
        K_vg = 100
        K_qg = 14

    model = Model(q_vocab_size=dataset.q_words,
                    K_vg = K_vg,
                    K_qg = K_qg,
                    vg_nodes_dim=2052,
                    qg_nodes_dim=2048,
                    emb_dim=args.emb,
                    feat_dim=dataset.feat_dim,
                    hid_dim=args.hid,
                    out_dim=dataset.n_answers,
                    pretrained_wemb_q=dataset.pretrained_wemb_q,
                    dropout=args.dropout,
                    question_emb=args.question_emb,
                    )

    criterion = nn.MultiLabelSoftMarginLoss()

    # Move it to multi GPU
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    criterion = criterion.cuda()

    # Define the optimiser
    lr_default = args.base_lr
    num_epochs = args.ep
    lr_decay_epochs = range(args.lr_decay_start, num_epochs,
                            args.lr_decay_step)
    gradual_warmup_steps = [0.5 * lr_default, 1.0 * lr_default,
                            1.5 * lr_default, 2.0 * lr_default]

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adamax(filter(lambda p: p.requires_grad,
                                model.parameters()),
                                lr=lr_default,
                                betas=(0.9, 0.999),
                                eps=1e-8,
                                weight_decay=args.weight_decay)


    # Continue training from saved model
    start_ep = 0
    if args.model_path and os.path.isfile(args.model_path):
        print('Resuming from checkpoint %s' % (args.model_path))
        ckpt = torch.load(args.model_path)
        start_ep = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    # Update the learning rate
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] = args.lr

    # Learning rate scheduler
    #scheduler = MultiStepLR(optimizer, milestones=[30], gamma=0.5,last_epoch = start_ep - 1)

    #scheduler = ExponentialLR(optimizer, gamma=0.9,last_epoch = start_ep - 1)
    #scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=3,verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

    # Train iterations
    print('Start training.')
    for ep in range(start_ep, start_ep+args.ep):

        #scheduler.step()
        ep_loss = 0.0
        ep_correct = 0.0
        ave_loss = 0.0
        ave_correct = 0.0
        losses = []
        count = 0

        if ep < len(gradual_warmup_steps):
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = gradual_warmup_steps[ep]
                print('gradual warmup lr: %.4f' %optimizer.param_groups[-1]['lr'])
        elif ep in lr_decay_epochs:
            for i in range(len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] *= args.lr_decay_rate
                print('decreased lr: %.4f' % optimizer.param_groups[-1]['lr'])
        else:
            print('lr: %.4f' % optimizer.param_groups[-1]['lr'])

        for step, next_batch in tqdm(enumerate(loader)):
            model.train()
            # Move batch to cuda


            q_batch, a_batch, vote_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch = \
                batch_to_cuda(next_batch)

            # forward pass

            output = model(q_batch, vgn_batch, vge_batch, qgn_batch, qge_batch, qglen, qlen_batch)

            #loss = criterion(output, a_batch)
            loss = criterion(output, a_batch) * dataset.n_answers

            # Compute batch accuracy based on vqa evaluation
            correct = total_vqa_score(output, vote_batch)
            ep_correct += correct
            # pytorch version 0.3
            ep_loss += loss.item()
            ave_correct += correct
            ave_loss += loss.item()
            losses.append(loss.cpu().item())

            # This is a 40 step average
            if step % 40 == 0 and step != 0:
                print('  Epoch %02d(%03d/%03d), ave loss: %.7f, ave accuracy: %.2f%%' %
                      (ep+1, step, n_batches, ave_loss/40,
                       ave_correct * 100 / (args.bsize*40)))

                ave_correct = 0
                ave_loss = 0

            # Compute gradient and do optimisation step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save model and compute accuracy for epoch
        epoch_loss = ep_loss / n_batches
        epoch_acc = ep_correct * 100 / (n_batches * args.bsize)
        #scheduler.step(epoch_acc)

        if ep >= 30:
            save(model, optimizer, ep, epoch_loss, epoch_acc,
                dir=args.save_dir, name=args.name+'_'+str(ep+1))

        print('Epoch %02d done, average loss: %.3f, average accuracy: %.2f%%' % (
              ep+1, epoch_loss, epoch_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description='Conditional Graph Convolutions for VQA or GQA')
    parser.add_argument('--train', action='store_true',
                        help='set this to training mode.')
    parser.add_argument('--trainval', action='store_true',
                        help='set this to train+val mode.')
    parser.add_argument('--eval', action='store_true',
                        help='set this to evaluation mode.')
    parser.add_argument('--test', action='store_true',
                        help='set this to test mode.')

    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--lr_decay_start', type=int, default=25)
    parser.add_argument('--lr_decay_rate', type=float, default=0.5)
    parser.add_argument('--lr_decay_step', type=int, default=2)
    parser.add_argument('--lr_decay_based_on_val', action='store_true',
                        help='Learning rate decay when val score descreases')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--ep', metavar='', type=int,
                        default=35, help='number of epochs.')
    parser.add_argument('--bsize', metavar='', type=int,
                        default=256, help='batch size.')

    parser.add_argument('--hid', metavar='', type=int,
                        default=1024, help='hidden dimension')
    parser.add_argument('--emb', metavar='', type=int, default=300,
                        help='question embedding dimension')
    parser.add_argument('--neighbourhood_size', metavar='', type=int, default=16,
                        help='number of graph neighbours to consider')
    parser.add_argument('--data_dir', metavar='', type=str, default='./data/GQA',
                        help='path to data directory')
    parser.add_argument('--data_type', metavar='', type=str, default='GQA',
                        help='type of data:GQA/VQA ')
    parser.add_argument('--data_balanced', metavar='', type=bool, default=True,
                        help='which data of GQA:balanced/all ')
    parser.add_argument('--fusion_way', metavar='', type=int, default=2,
                        help='which way of latefusion after GM ')
    parser.add_argument('--question_emb', metavar='', type=bool, default=True,
                        help='last layer whether combined the question embedding feature.')
    parser.add_argument('--save_dir', metavar='', type=str, default='./GQA_save')
    parser.add_argument('--name', metavar='', type=str,
                        default='model', help='model name')
    parser.add_argument('--dropout', metavar='', type=float, default=0.5,
                        help='probability of dropping out FC nodes during training')
    parser.add_argument('--model_path', metavar='', type=str,
                        help='trained model path.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0:
        raise SystemExit('Unknown argument: {}'.format(unparsed))
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if args.train:
        train(args)
    if args.trainval:
        trainval(args)
    if args.eval:
        eval_model(args)
    if args.test:
        test(args)
    if not args.train and not args.eval and not args.trainval and not args.test:
        parser.print_help()
