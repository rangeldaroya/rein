from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from collections import OrderedDict
import math
import numpy as np
import time

import networkx as nx
import torch.nn.init as init
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import MultiStepLR
from sklearn.decomposition import PCA
import logging
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value
import scipy.misc
import time as tm

from utils import *
from data import *
from args import Args


from pointnet import AutoencoderPoint, AutoencoderPoint2

# from train import *
from torchsummary import summary
import h5py

import pickle
import argparse
from progressbar import ProgressBar

import os

import visdom
vis = visdom.Visdom(port=8888, env='Points Autoencoder')

class ChamferLoss(torch.nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        # print('P: ', P, P.shape)
        # minimum distance from prediction to ground truth;
        mins, idx1 = torch.min(P, 1)
        #idx1 are indices in ground truth closest to corresponding indices in prediction
        loss_1 = torch.sum(mins)
        # gives miniimum distance per point in ground truth (ground truth PC to predicted PC)
        mins, idx2 = torch.min(P, 2)
        #idx2 are indices in prediction closest to corresponding indices in ground truth
        loss_2 = torch.sum(mins)

        # return loss_1 + loss_2, idx1
        return loss_1/10000.0, idx1, loss_2/10000.0, idx2   #NOTE: changed this for regloss-cdist-hdist

    def batch_pairwise_dist(self, x, y):
        # x = torch.unsqueeze(x, 0)  # ground truth
        # y = torch.unsqueeze(y, 0)  # prediction

        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        #brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(
            1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        # print('rx shape: ', rx.transpose(2, 1).shape, rx.transpose(2, 1))
        # print('ry shape: ', ry.shape, ry)
        # print('zz shape: ', zz.shape)
        P = (rx.transpose(2, 1) + ry - 2*zz)

        return P

class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None, pred_pts=False, latent_vec=False):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output
        self.pred_pts = pred_pts
        self.latent_vec = latent_vec

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )
        if self.pred_pts:
            self.output_pts = nn.Sequential(
                nn.Linear(input_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, 3)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.pred_pts:
            out_pts = self.output_pts(input_raw)
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        # print('rnn input shape: ', input, 'rnn output shape: ', output_raw, 'hidden: ', self.hidden.shape)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]

        if self.has_output:
            output_raw = self.output(output_raw)
        # print('final output shape: ', output_raw.shape)
        # return hidden state at each time step

        if self.pred_pts:
            return output_raw, out_pts

        return output_raw

def get_train_files(train_files, filepath, lines, ctr, verts, edges, faces):

    graph_curr = nx.Graph()
    graph_curr.add_edges_from(edges)
    for k in range(verts.shape[0]):
        graph_curr.add_node(k, pos=tuple(verts[k]))
    graph_curr.remove_nodes_from(list(nx.isolates(graph_curr)))
    adj_copy = np.asarray(nx.to_numpy_matrix(graph_curr))

    x_batch = np.zeros((500, 500))  # here zeros are padded for small graph
    x_batch[0,:] = 1 # the first input token is all ones
    
    len_batch = adj_copy.shape[0]
    x_idx = np.random.permutation(adj_copy.shape[0])
    adj_copy = adj_copy[np.ix_(x_idx, x_idx)]   #changes order of adj_copy elements randomly but maintains same group of elements per row and column (same number of 1s in every row and column as before)
    
    pos = nx.get_node_attributes(graph_curr, 'pos')      #maybe can eventually change this to the latent vecor of corresponding point cloud???
    xyz = np.array([list(pos[i]) for i in pos])
    xyz = xyz[x_idx]
    
    #""" #removed this starting model_save_mhull_latent_2.
    adj_copy_matrix = np.asmatrix(adj_copy)     #basically, changes orders of nodes (reassigns which node appears first)
    G = nx.from_numpy_matrix(adj_copy_matrix)
    # then do bfs in the permuted G
    start_idx = np.random.randint(adj_copy.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))     #run bfs to order the nodes
    # print('x_idx: ', x_idx, x_idx.shape)

    xyz = xyz[x_idx]
    adj_copy = adj_copy[np.ix_(x_idx, x_idx)]   #reorder the nodes based on bfs ordering
    


    adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=500)
    x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
    if np.array(mesh.vertices).shape[0] != xyz.shape[0]:
        print('numverts: ', np.array(mesh.vertices).shape[0], 'xyz.shape: ', xyz.shape)
        print('incosistent numcoor filepath: ', filepath)
        continue
    
    if (np.array(mesh.vertices).shape[0] <= 500):
        train_files.append(filepath)
        totverts += np.array(mesh.vertices).shape[0]
        if dataset == 'modelnet-1hull':
            train_files.append(os.path.join(path, lines[:-4] + '_simptemp_deform_a.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_simptemp_deform_b.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_simptemp_deform_c.obj'))
            ctr += 1
        elif dataset == 'modelnet10-1hull':
            train_files.append(os.path.join(path, lines[:-4] + '_simptemp_deform_a.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_simptemp_deform_b.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_simptemp_deform_c.obj'))
            ctr += 1
        elif dataset == 'modelnet10-split':
            train_files.append(os.path.join(path, lines[:-4] + '_split_deform_a.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_split_deform_b.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_split_deform_c.obj'))
            # print(os.path.join(path, lines[:-4] + '_split_deform_c.obj'))
            ctr += 1
        elif dataset == 'modelnet10-splitmidpt':
            train_files.append(os.path.join(path, lines[:-4] + '_splitmdpt_deform_a.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_splitmdpt_deform_b.obj'))
            ctr += 1
            train_files.append(os.path.join(path, lines[:-4] + '_splitmdpt_deform_c.obj'))
            ctr += 1
        ctr += 1
    return train_files, ctr

def get_filepath(dataset, path, lines, classname='all'):
    if dataset == 'modelnet-1hull':     #provision for modelnet40
        filepath = os.path.join(path, lines[:-4] + '_1hull.obj')
        classname_file = lines[:len(classname)]
        if (classname_file != classname) and (classname!='all'): 
            continue
        # print('classname_file: ', classname_file)
    elif dataset == 'modelnet10-1hull':
        filepath = os.path.join(path, lines[:-4] + '_1hull.obj')
    elif dataset == 'modelnet10-split':
        filepath = os.path.join(path, lines[:-4] + '_split.obj')
    elif dataset == 'modelnet10-splitmidpt':
        filepath = os.path.join(path, lines[:-4] + '_splitmdpt.obj')
        #print("fikepath: ", filepath)
    elif dataset == 'shapenet-split':
        filepath = os.path.join(path, lines, 'model_split.obj')
    elif dataset == 'shapenet-splitmidpt':
        filepath = os.path.join(path, lines, 'model_splitmdpt.obj')
    elif dataset == 'shapenet-1hull':
        filepath = os.path.join(path, lines, 'model_1hull.obj')
    elif dataset == 'shapenet-1hull-v2':        #when training for ShapeNet Wrapped and ShapeNet Patched
        filepath = os.path.join(path, lines, 'blender_model.obj')
    return filepath

def str2bool(v):
    """
    Convert string to bool for argument parsing in terminal
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    args = Args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix',args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)


    parser = argparse.ArgumentParser()
    parser.add_argument('--start', dest='start_epoch', help='start index of epoch', type=int, default=0)
    parser.add_argument('--end', dest='end_epoch', help='number of epochs to run before stopping training', type=int, default=105)
    parser.add_argument('--dataset', dest='dataset', help='dataset source for training', type=str, default='shapenet-split')
    # parser.add_argument('--ae_only', dest='ae_only', help='to train just ae or not', type=bool, default=False)
    parser.add_argument('--class', dest='classname', help='class to include in training (provision)', type=str, default='all')
    parser.add_argument('--toeval', dest='toeval', help='to evaluate chamfer distance, not train. for AE eval', type=bool, default=False)
    parser.add_argument('--ckptloc', dest='ckptloc', help='location of checkpoint loading and saving', type=str, default=None)
    parser.add_argument('--pretrain', dest='pretrain', help='to pretrain?(y/n; defeault n)', type=str2bool, default=False)
    parser.add_argument('--pretrain_ae_path', dest='pretrain_ae_path', help='specify pretrained AE path', type=str, default=None)
    parser.add_argument('--pretrain_rnn_path', dest='pretrain_rnn_path', help='specify pretrained RNN path', type=str, default=None)
    parser.add_argument('--pretrain_output_path', dest='pretrain_output_path', help='specify pretrained Output path', type=str, default=None)
    args2 = parser.parse_args()
    args.start_epoch = args2.start_epoch
    args.end_epoch = args2.end_epoch
    dataset = args2.dataset
    train_ae_only = args2.ae_only
    toeval = args2.toeval
    to_pretrain = args2.pretrain


    if dataset == 'modelnet-1hull':
        path = '/home/rangeldaroya/Documents/modelnet40_auto_aligned'
        if not toeval:
            f = open(os.path.join(path, 'train.txt'), 'r')
        else:
            f = open(os.path.join(path, 'test.txt'), 'r')
    elif dataset == 'modelnet10-1hull':
        path = '/home/rangeldaroya/Documents/ModelNet10'
        if not toeval:
            f = open(os.path.join(path, 'train.txt'), 'r')
        else:
            f = open(os.path.join(path, 'test.txt'), 'r')
    elif (dataset == 'modelnet10-split') or (dataset == 'modelnet10-splitmidpt'):
        path = '/home/rangeldaroya/Documents/ModelNet10'
        f = open(os.path.join(path, 'train.txt'), 'r')
    elif (dataset == 'shapenet-split') or (dataset == 'shapenet-splitmidpt'):
        path = '/home/rangeldaroya/Documents/FinalMeshDataset/'
        f = open(os.path.join(path, 'train_models.txt'), 'r')
    elif dataset == 'shapenet-1hull' or dataset == 'shapenet-1hull-v2':
        path = '/home/rangeldaroya/Documents/FinalMeshDataset/'
        f = open(os.path.join(path, 'train_models.txt'), 'r')

    print('Loading all train files')
    train_files = []
    ctr = 0
    totverts = 0
    # classname = 'airplane'
    classname = args2.classname
    for lines in f:
        lines = lines.strip()

        filepath = get_filepath(dataset, path, lines, classname=classname)

        try:
            mesh = trimesh.load(filepath)
        except:
            print('ERROR filepath: ', filepath)
            continue
        
        try:    #check if loaded mesh is an array; if array, it's not a single contiguous part
            if len(mesh)>1:
                print('mesh: ', mesh)
            elif len(mesh)==0:
                print('mesh: ', mesh, 'filepath: ', filepath)
            continue
        except:
            pass

        verts = mesh.vertices
        edges = mesh.edges
        faces = mesh.faces

        train_files, ctr = get_train_files(train_files, filepath, lines, ctr, verts, edges, faces)
        
    f.close()
    print('ctr: ', ctr)
    print('\n\ntotverts: ', totverts)
    # hf = h5py.File(os.path.join(path, 'traindata_500v.hdf5'), 'r')
    hf = None
    print('\n\n\nnum total train files: ', len(train_files))        
    print('\n\n\n')
    # random.seed(123)
    shuffle(train_files)
    print('Done loading and shuffling train files')


    args.max_prev_node = 500
    args.max_num_node = 500
    args.batch_ratio = len(train_files)
    args.epochs = args.end_epoch
    if toeval:
        args.epochs = 1
    args.epochs_save = 1

    args.latent_size = 64
    args.hidden_size_rnn = 128
    args.hidden_size_rnn_output = 16
    args.embedding_size_rnn = 64
    args.embedding_size_rnn_output = 8
    args.num_layers = 4

    if args2.ckptloc is not None:
        model_path = args2.ckptloc
    else:
        model_path = 'trained_models/'
    args.model_save_path = args.dir_input+model_path
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(model_path+'_logs'):
        os.makedirs(model_path+'_logs')

    args.latentonly = False
    args.vertlatent = True
    if dataset[:len('shapenet')] == 'shapenet':
        dataset = Graph_sequence_sampler_pytorch(train_files,hf,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node, tonormverts=False)
    else:
        dataset = Graph_sequence_sampler_pytorch(train_files,hf,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_train = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)

    print('GraphRNN_RNN')

    rnn = GRU_plain(input_size=args.latent_size+3, embedding_size=args.embedding_size_rnn,
                        hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=args.hidden_size_rnn_output, pred_pts=False, latent_vec=True).cuda()
    output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                        hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                        has_output=True, output_size=1, pred_pts=False).cuda()

    ### start training
    print('\n\nSTART TRAINING...\n\n')
    print('RNN NETWORK SUMMARY: ')
    if args.vertlatent==True:
        summary(rnn, (args.max_num_node,3+args.latent_size))
    elif rnn.latent_vec==True:
        summary(rnn, (args.max_num_node,args.max_prev_node+args.latent_size))
    else:
        summary(rnn, (args.max_num_node,args.max_prev_node))
    print('\n\nOUTPUT NETWORK SUMMARY: ')
    summary(output, (args.max_prev_node, 1))



    autoencoder = AutoencoderPoint2(num_points = args.max_num_node, latent_size=args.latent_size)
    autoencoder.cuda()
    autoencoder.train()
    ae_opt = optim.Adam(autoencoder.parameters(), lr=1e-5)

    ckpt_epoch = 0
    if to_pretrain:
        print('Loading pretrained network')
        ae_path = args2.pretrain_ae_path
        rnn_path = args2.pretrain_rnn_path
        output_path = args2.pretrain_output_path
        autoencoder.load_state_dict(torch.load(ae_path))
        rnn.load_state_dict(torch.load(rnn_path))
        output.load_state_dict(torch.load(output_path))

    if args.start_epoch != 0:   #here if you want to continue training from a saved checkpoint
        ckpt_epoch = args.start_epoch
        ae_path = os.path.join(model_path,'GraphRNN_RNN_meshes_4_128_ae_%d.dat' % ckpt_epoch)
        rnn_path = os.path.join(model_path,'GraphRNN_RNN_meshes_4_128_lstm_%d.dat' % ckpt_epoch)
        output_path = os.path.join(model_path,'GraphRNN_RNN_meshes_4_128_output_%d.dat' % ckpt_epoch)
        autoencoder.load_state_dict(torch.load(ae_path))
        rnn.load_state_dict(torch.load(rnn_path))
        output.load_state_dict(torch.load(output_path))
    rnn.train()
    output.train()
    autoencoder.train()
    # for p in autoencoder.parameters():  #freeze autoencoder weights
    #     p.requires_grad = False
    epoch = 1 + ckpt_epoch

    # initialize optimizer
    optimizer_rnn = optim.Adam(list(rnn.parameters()), lr=args.lr)
    optimizer_output = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_rnn = MultiStepLR(optimizer_rnn, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_output = MultiStepLR(optimizer_output, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    print('args.epochs: ', args.epochs)
    print('ckpt_spoch: ', ckpt_epoch)
    max_epoch = args.epochs + ckpt_epoch
    print('max_epoch: ', max_epoch)
    

    ###########################
    #         TRAINING        #
    ###########################
    for _ in range(args.epochs):
        rnn.train()
        output.train()
        loss_sum = 0
        chamloss_sum = 0
        binloss_sum = 0
        running_cd = 0
        # num_train = args.batch_ratio

        # train_idx = list(np.arange(0,num_train))
        # random.shuffle(train_idx)
        batch_idx = 0
        pbar = ProgressBar()
        
        for data in pbar(dataset_train):

            rnn.zero_grad()
            output.zero_grad()
            ae_opt.zero_grad()

            x_unsorted = data['x'].float()
            y_unsorted = data['y'].float()
            y_len_unsorted = data['len']
            mesh_pts = data['meshpts']
            
            
            mesh_pts = Variable(mesh_pts).float().cuda()
            if rnn.latent_vec==True and rnn.pred_pts==False:
                pts = data['pts']
                pts = Variable(pts).cuda()
                pts = pts.transpose(2,1).float()
                pred_pts, endpts = autoencoder(pts)
                latent_pts = endpts['embedding']
                if batch_idx % 10 == 0:
                    vis.scatter(X=pts.transpose(2,1).contiguous()[0].data.cpu(), win='INPUT', opts=dict(title='INPUT', markersize=2))
                    vis.scatter(X=pred_pts.transpose(2,1).contiguous()[0].data.cpu(), win='INPUT_RECONSTRUCTED', opts=dict(title='INPUT_RECONSTRUCTED', markersize=2))
                
            if rnn.pred_pts==True:
                # x_unsorted, y_unsorted, y_len_unsorted, points, mesh_pts = next(data_loader)
                points = data['pts']
                points = Variable(points).cuda()
                points = points.transpose(2,1).float()
                pred_pts, endpts = autoencoder(points)
                latent_pts = endpts['embedding']



            y_len_max = max(y_len_unsorted)
            x_unsorted = x_unsorted[:, 0:y_len_max, :]
            y_unsorted = y_unsorted[:, 0:y_len_max, :]
            # initialize lstm hidden state according to batch size
            rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))

            # sort input
            y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
            y_len = y_len.numpy().tolist()
            x = torch.index_select(x_unsorted,0,sort_index)
            y = torch.index_select(y_unsorted,0,sort_index)

            sort_index_meshpts = sort_index.cuda()
            mesh_pts = torch.index_select(mesh_pts,0,sort_index_meshpts)

            # input, output for output rnn module
            # a smart use of pytorch builtin function: pack variable--b1_l1,b2_l1,...,b1_l2,b2_l2,...
            y_reshape = pack_padded_sequence(y,y_len,batch_first=True).data
            # reverse y_reshape, so that their lengths are sorted, add dimension
            idx = [i for i in range(y_reshape.size(0)-1, -1, -1)]
            idx = torch.LongTensor(idx)
            y_reshape = y_reshape.index_select(0, idx)
            y_reshape = y_reshape.view(y_reshape.size(0),y_reshape.size(1),1)
            
            idx_meshpts = [i for i in range(mesh_pts.size(0)-1, -1, -1)]
            idx_meshpts = torch.LongTensor(idx_meshpts).cuda()
            mesh_pts = mesh_pts.index_select(0, idx_meshpts)

            output_x = torch.cat((torch.ones(y_reshape.size(0),1,1),torch.zeros(y_reshape[:,0:-1,0:1].shape)),dim=1)
            output_y = y_reshape
            # batch size for output module: sum(y_len)
            output_y_len = []
            output_y_len_bin = np.bincount(np.array(y_len))
            for i in range(len(output_y_len_bin)-1,0,-1):
                count_temp = np.sum(output_y_len_bin[i:]) # count how many y_len is above i
                output_y_len.extend([min(i,y.size(2))]*count_temp) # put them in output_y_len; max value should not exceed y.size(2)
            # pack into variable
            x = Variable(x).cuda()
            y = Variable(y).cuda()
            output_x = Variable(output_x).cuda()
            output_y = Variable(output_y).cuda()

            #concatenate latent vector and input to rn
            latent_new = latent_pts.repeat(1,x.shape[1]).view(latent_pts.shape[0], -1, latent_pts.shape[-1])
            inp = torch.cat([mesh_pts, latent_new], 2)
            h = rnn(inp, pack=True, input_len=y_len)
            h = pack_padded_sequence(h,y_len,batch_first=True).data # get packed hidden vector
            
            
            idx = [i for i in range(h.size(0) - 1, -1, -1)]
            idx = Variable(torch.LongTensor(idx)).cuda()
            h = h.index_select(0, idx)
            hidden_null = Variable(torch.zeros(args.num_layers-1, h.size(0), h.size(1))).cuda()
            output.hidden = torch.cat((h.view(1,h.size(0),h.size(1)),hidden_null),dim=0) # num_layers, batch_size, hidden_size
            
            
            y_pred = output(output_x, pack=True, input_len=output_y_len)
            y_pred = F.sigmoid(y_pred)
            
            y_pred = pack_padded_sequence(y_pred, output_y_len, batch_first=True)
            y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            output_y = pack_padded_sequence(output_y,output_y_len,batch_first=True)
            output_y = pad_packed_sequence(output_y,batch_first=True)[0]
            
            loss_adj = binary_cross_entropy_weight(y_pred, output_y)
            loss_pts = 0
            elif rnn.latent_vec==True:
                loss_forward, _, loss_backward, _ = ChamferLoss()(pred_pts.transpose(2,1), pts.transpose(2,1))
                loss_pts = torch.mean(loss_forward+loss_backward)*10e2
            else:
                loss_pts = 0
            loss = loss_pts + loss_adj
            loss.backward()
            # update deterministic and lstm
            optimizer_output.step()
            optimizer_rnn.step()
            scheduler_output.step()
            scheduler_rnn.step()

            if rnn.pred_pts==True or rnn.latent_vec==True:
                ae_opt.step()

            chamloss_sum += loss_pts
            binloss_sum += loss_adj
            if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
                print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}, chamferLoss: {}, binLoss: {}'.format(
                    epoch, args.epochs,loss.data, args.graph_type, args.num_layers, args.hidden_size_rnn, chamloss_sum/(batch_idx+1), binloss_sum/(batch_idx+1)))

            feature_dim = y.size(1)*y.size(2)
            loss_sum += loss.data*feature_dim
            batch_idx += 1
        
        print("saving lstm rnn")
        fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
        torch.save(rnn.state_dict(), fname)
        print("saving output rnn")
        fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
        torch.save(output.state_dict(), fname)
        print("saving ae")
        fname = args.model_save_path + args.fname + 'ae_' + str(epoch) + '.dat'
        torch.save(autoencoder.state_dict(), fname)
        epoch += 1