from __future__ import unicode_literals, print_function, division
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



def binary_cross_entropy_weight(y_pred, y,has_weight=False, weight_length=1, weight_max=10):
    '''

    :param y_pred:
    :param y:
    :param weight_length: how long until the end of sequence shall we add weight
    :param weight_value: the magnitude that the weight is enhanced
    :return:
    '''
    if has_weight:
        weight = torch.ones(y.size(0),y.size(1),y.size(2))
        weight_linear = torch.arange(1,weight_length+1)/weight_length*weight_max
        weight_linear = weight_linear.view(1,weight_length,1).repeat(y.size(0),1,y.size(2))
        weight[:,-1*weight_length:,:] = weight_linear
        loss = F.binary_cross_entropy(y_pred, y, weight=weight.cuda())
    else:
        loss = F.binary_cross_entropy(y_pred, y)
    return loss


def sample_tensor(y,sample=True, thresh=0.5):
    # do sampling
    if sample:
        y_thresh = Variable(torch.rand(y.size())).cuda()
        y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size())*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result

def gumbel_softmax(logits, temperature, eps=1e-9):
    '''

    :param logits: shape: N*L
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size())
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    noise = Variable(noise).cuda()

    x = (logits + noise) / temperature
    x = F.softmax(x)
    return x

# for i in range(10):
#     x = Variable(torch.randn(1,10)).cuda()
#     y = gumbel_softmax(x, temperature=0.01)
#     print(x)
#     print(y)
#     _,id = y.topk(1)
#     print(id)


def gumbel_sigmoid(logits, temperature):
    '''

    :param logits:
    :param temperature:
    :param eps:
    :return:
    '''
    # get gumbel noise
    noise = torch.rand(logits.size()) # uniform(0,1)
    noise_logistic = torch.log(noise)-torch.log(1-noise) # logistic(0,1)
    noise = Variable(noise_logistic).cuda()

    x = (logits + noise) / temperature
    x = F.sigmoid(x)
    return x

# x = Variable(torch.randn(100)).cuda()
# y = gumbel_sigmoid(x,temperature=0.01)
# print(x)
# print(y)

def sample_sigmoid(y, sample, thresh=0.5, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time>1:
            y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).cuda()
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda()
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def sample_sigmoid_supervised(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).cuda()
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current<y_len[i]:
            while True:
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                # print('current',current)
                # print('y_result',y_result[i].data)
                # print('y',y[i])
                y_diff = y_result[i].data-y[i]
                if (y_diff>=0).all():
                    break
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data>0).any():
                    break
    return y_result

def sample_sigmoid_supervised_simple(y_pred, y, current, y_len, sample_time=2):
    '''
        do sampling over unnormalized score
    :param y_pred: input
    :param y: supervision
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    y_pred = F.sigmoid(y_pred)
    # do sampling
    y_result = Variable(torch.rand(y_pred.size(0), y_pred.size(1), y_pred.size(2))).cuda()
    # loop over all batches
    for i in range(y_result.size(0)):
        # using supervision
        if current<y_len[i]:
            y_result[i] = y[i]
        # supervision done
        else:
            # do 'multi_sample' times sampling
            for j in range(sample_time):
                y_thresh = Variable(torch.rand(y_pred.size(1), y_pred.size(2))).cuda()
                y_result[i] = torch.gt(y_pred[i], y_thresh).float()
                if (torch.sum(y_result[i]).data>0).any():
                    break
    return y_result

################### current adopted model, LSTM+MLP || LSTM+VAE || LSTM+LSTM (where LSTM can be GRU as well)
#####
# definition of terms
# h: hidden state of LSTM
# y: edge prediction, model output
# n: noise for generator
# l: whether an output is real or not, binary

# plain LSTM model
class LSTM_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(LSTM_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                nn.Linear(hidden_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, output_size)
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda(),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)).cuda())

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw

# plain GRU model
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
        """
        if self.pred_pts:
            # print('output_raw: ', output_raw.shape, 'hidden: ', self.hidden.shape)
            out_pts = self.output_pts(output_raw)
            if self.has_output:
                output_raw = self.output(output_raw)
            # print('out_pts: ', out_pts.shape)
            return output_raw, out_pts
        """
        if self.has_output:
            output_raw = self.output(output_raw)
        # print('final output shape: ', output_raw.shape)
        # return hidden state at each time step

        if self.pred_pts:
            return output_raw, out_pts

        return output_raw



# a deterministic linear output
class MLP_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        return y

# a deterministic linear output, additional output indicates if the sequence should continue grow
class MLP_token_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_token_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )
        self.token_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        t = self.token_output(h)
        return y,t

# a deterministic linear output (update: add noise)
class MLP_VAE_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size) # mu
        self.encode_12 = nn.Linear(h_size, embedding_size) # lsgms

        self.decode_1 = nn.Linear(embedding_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size) # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size())).cuda()
        z = eps*z_sgm + z_mu
        # decoder
        y = self.decode_1(z)
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms

# a deterministic linear output (update: add noise)
class MLP_VAE_conditional_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_VAE_conditional_plain, self).__init__()
        self.encode_11 = nn.Linear(h_size, embedding_size)  # mu
        self.encode_12 = nn.Linear(h_size, embedding_size)  # lsgms

        self.decode_1 = nn.Linear(embedding_size+h_size, embedding_size)
        self.decode_2 = nn.Linear(embedding_size, y_size)  # make edge prediction (reconstruct)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        # encoder
        z_mu = self.encode_11(h)
        z_lsgms = self.encode_12(h)
        # reparameterize
        z_sgm = z_lsgms.mul(0.5).exp_()
        eps = Variable(torch.randn(z_sgm.size(0), z_sgm.size(1), z_sgm.size(2))).cuda()
        z = eps * z_sgm + z_mu
        # decoder
        y = self.decode_1(torch.cat((h,z),dim=2))
        y = self.relu(y)
        y = self.decode_2(y)
        return y, z_mu, z_lsgms








################################################## code that are NOT used for final version #############


# RNN that updates according to graph structure, new proposed model
class Graph_RNN_structure(nn.Module):
    def __init__(self, hidden_size, batch_size, output_size, num_layers, is_dilation=True, is_bn=True):
        super(Graph_RNN_structure, self).__init__()
        ## model configuration
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.num_layers = num_layers # num_layers of cnn_output
        self.is_bn=is_bn

        ## model
        self.relu = nn.ReLU()
        # self.linear_output = nn.Linear(hidden_size, 1)
        # self.linear_output_simple = nn.Linear(hidden_size, output_size)
        # for state transition use only, input is null
        # self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # use CNN to produce output prediction
        # self.cnn_output = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_size, 1, kernel_size=3, dilation=1, padding=1)
        # )

        if is_dilation:
            self.conv_block = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=2**i, padding=2**i) for i in range(num_layers-1)])
        else:
            self.conv_block = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1) for i in range(num_layers-1)])
        self.bn_block = nn.ModuleList([nn.BatchNorm1d(hidden_size) for i in range(num_layers-1)])
        self.conv_out = nn.Conv1d(hidden_size, 1, kernel_size=3, dilation=1, padding=1)


        # # use CNN to do state transition
        # self.cnn_transition = nn.Sequential(
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1),
        #     # nn.BatchNorm1d(hidden_size),
        #     nn.ReLU(),
        #     nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=1, padding=1)
        # )

        # use linear to do transition, same as GCN mean aggregator
        self.linear_transition = nn.Sequential(
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU()
        )


        # GRU based output, output a single edge prediction at a time
        # self.gru_output = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        # use a list to keep all generated hidden vectors, each hidden has size batch*hidden_dim*1, and the list size is expanding
        # when using convolution to compute attention weight, we need to first concat the list into a pytorch variable: batch*hidden_dim*current_num_nodes
        self.hidden_all = []

        ## initialize
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # print('linear')
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # print(m.weight.data.size())
            if isinstance(m, nn.Conv1d):
                # print('conv1d')
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # print(m.weight.data.size())
            if isinstance(m, nn.BatchNorm1d):
                # print('batchnorm1d')
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # print(m.weight.data.size())
            if isinstance(m, nn.GRU):
                # print('gru')
                m.weight_ih_l0.data = init.xavier_uniform(m.weight_ih_l0.data,
                                                                  gain=nn.init.calculate_gain('sigmoid'))
                m.weight_hh_l0.data = init.xavier_uniform(m.weight_hh_l0.data,
                                                                  gain=nn.init.calculate_gain('sigmoid'))
                m.bias_ih_l0.data = torch.ones(m.bias_ih_l0.data.size(0)) * 0.25
                m.bias_hh_l0.data = torch.ones(m.bias_hh_l0.data.size(0)) * 0.25

    def init_hidden(self,len=None):
        if len is None:
            return Variable(torch.ones(self.batch_size, self.hidden_size, 1)).cuda()
        else:
            hidden_list = []
            for i in range(len):
                hidden_list.append(Variable(torch.ones(self.batch_size, self.hidden_size, 1)).cuda())
            return hidden_list

    # only run a single forward step
    def forward(self, x, teacher_forcing, temperature = 0.5, bptt=True,bptt_len=20, flexible=True,max_prev_node=100):
        # x: batch*1*self.output_size, the groud truth
        # todo: current only look back to self.output_size nodes, try to look back according to bfs sequence

        # 1 first compute new state
        # print('hidden_all', self.hidden_all[-1*self.output_size:])
        # hidden_all_cat = torch.cat(self.hidden_all[-1*self.output_size:], dim=2)

        # # # add BPTT, detach the first variable
        # if bptt:
        #     self.hidden_all[0] = Variable(self.hidden_all[0].data).cuda()

        hidden_all_cat = torch.cat(self.hidden_all, dim=2)
        # print(hidden_all_cat.size())

        # print('hidden_all_cat',hidden_all_cat.size())
        # att_weight size: batch*1*current_num_nodes
        for i in range(self.num_layers-1):
            hidden_all_cat = self.conv_block[i](hidden_all_cat)
            if self.is_bn:
                hidden_all_cat = self.bn_block[i](hidden_all_cat)
            hidden_all_cat = self.relu(hidden_all_cat)
        x_pred = self.conv_out(hidden_all_cat)
        # 2 then compute output, using a gru
        # first try the simple version, directly give the edge prediction
        # x_pred = self.linear_output_simple(hidden_new)
        # x_pred = x_pred.view(x_pred.size(0),1,x_pred.size(1))

        # todo: use a gru version output
        # if sample==False:
        #     # when training: we know the ground truth, input the sequence at once
        #     y_pred,_ = self.gru_output(x, hidden_new.permute(2,0,1))
        #     y_pred = self.linear_output(y_pred)
        # else:
        #     # when validating, we need to sampling at each time step
        #     y_pred = Variable(torch.zeros(x.size(0), x.size(1), x.size(2))).cuda()
        #     y_pred_long = Variable(torch.zeros(x.size(0), x.size(1), x.size(2))).cuda()
        #     x_step = x[:, 0:1, :]
        #     for i in range(x.size(1)):
        #         y_step,_ = self.gru_output(x_step)
        #         y_step = self.linear_output(y_step)
        #         y_pred[:, i, :] = y_step
        #         y_step = F.sigmoid(y_step)
        #         x_step = sample(y_step, sample=True, thresh=0.45)
        #         y_pred_long[:, i, :] = x_step
        #     pass


        # 3 then update self.hidden_all list
        # i.e., model will use ground truth to update new node
        # x_pred_sample = gumbel_sigmoid(x_pred, temperature=temperature)
        x_pred_sample = sample_tensor(F.sigmoid(x_pred),sample=True)
        thresh = 0.5
        x_thresh = Variable(torch.ones(x_pred_sample.size(0), x_pred_sample.size(1), x_pred_sample.size(2)) * thresh).cuda()
        x_pred_sample_long = torch.gt(x_pred_sample, x_thresh).long()
        if teacher_forcing:
            # first mask previous hidden states
            hidden_all_cat_select = hidden_all_cat*x
            x_sum = torch.sum(x, dim=2, keepdim=True).float()

        # i.e., the model will use it's own prediction to attend
        else:
            # first mask previous hidden states
            hidden_all_cat_select = hidden_all_cat*x_pred_sample
            x_sum = torch.sum(x_pred_sample_long, dim=2, keepdim=True).float()

        # update hidden vector for new nodes
        hidden_new = torch.sum(hidden_all_cat_select, dim=2, keepdim=True) / x_sum

        hidden_new = self.linear_transition(hidden_new.permute(0, 2, 1))
        hidden_new = hidden_new.permute(0, 2, 1)

        if flexible:
            # use ground truth to maintaing history state
            if teacher_forcing:
                x_id = torch.min(torch.nonzero(torch.squeeze(x.data)))
                self.hidden_all = self.hidden_all[x_id:]
            # use prediction to maintaing history state
            else:
                x_id = torch.min(torch.nonzero(torch.squeeze(x_pred_sample_long.data)))
                start = max(len(self.hidden_all)-max_prev_node+1, x_id)
                self.hidden_all = self.hidden_all[start:]

        # maintaing a fixed size history state
        else:
            # self.hidden_all.pop(0)
            self.hidden_all = self.hidden_all[1:]

        self.hidden_all.append(hidden_new)

        # 4 return prediction
        # print('x_pred',x_pred)
        # print('x_pred_mean', torch.mean(x_pred))
        # print('x_pred_sample_mean', torch.mean(x_pred_sample))
        return x_pred, x_pred_sample

# batch_size = 8
# output_size = 4
# generator = Graph_RNN_structure(hidden_size=16, batch_size=batch_size, output_size=output_size, num_layers=1).cuda()
# for i in range(4):
#     generator.hidden_all.append(generator.init_hidden())
#
# x = Variable(torch.rand(batch_size,1,output_size)).cuda()
# x_pred = generator(x,teacher_forcing=True, sample=True)
# print(x_pred)




# current baseline model, generating a graph by lstm
class Graph_generator_LSTM(nn.Module):
    def __init__(self,feature_size, input_size, hidden_size, output_size, batch_size, num_layers):
        super(Graph_generator_LSTM, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear_input = nn.Linear(feature_size, input_size)
        self.linear_output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # initialize
        # self.hidden,self.cell = self.init_hidden()
        self.hidden = self.init_hidden()

        self.lstm.weight_ih_l0.data = init.xavier_uniform(self.lstm.weight_ih_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        self.lstm.weight_hh_l0.data = init.xavier_uniform(self.lstm.weight_hh_l0.data, gain=nn.init.calculate_gain('sigmoid'))
        self.lstm.bias_ih_l0.data = torch.ones(self.lstm.bias_ih_l0.data.size(0))*0.25
        self.lstm.bias_hh_l0.data = torch.ones(self.lstm.bias_hh_l0.data.size(0))*0.25
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data,gain=nn.init.calculate_gain('relu'))
    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers,self.batch_size, self.hidden_size)).cuda(), Variable(torch.zeros(self.num_layers,self.batch_size, self.hidden_size)).cuda())


    def forward(self, input_raw, pack=False,len=None):
        input = self.linear_input(input_raw)
        input = self.relu(input)
        if pack:
            input = pack_padded_sequence(input, len, batch_first=True)
        output_raw, self.hidden = self.lstm(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        output = self.linear_output(output_raw)
        return output






# a simple MLP generator output
class Graph_generator_LSTM_output_generator(nn.Module):
    def __init__(self,h_size, n_size, y_size):
        super(Graph_generator_LSTM_output_generator, self).__init__()
        # one layer MLP
        self.generator_output = nn.Sequential(
            nn.Linear(h_size+n_size, 64),
            nn.ReLU(),
            nn.Linear(64, y_size),
            nn.Sigmoid()
        )
    def forward(self,h,n,temperature):
        y_cat = torch.cat((h,n), dim=2)
        y = self.generator_output(y_cat)
        # y = gumbel_sigmoid(y,temperature=temperature)
        return y

# a simple MLP discriminator
class Graph_generator_LSTM_output_discriminator(nn.Module):
    def __init__(self, h_size, y_size):
        super(Graph_generator_LSTM_output_discriminator, self).__init__()
        # one layer MLP
        self.discriminator_output = nn.Sequential(
            nn.Linear(h_size+y_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self,h,y):
        y_cat = torch.cat((h,y),dim=2)
        l = self.discriminator_output(y_cat)
        return l



# GCN basic operation
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        # self.relu = nn.ReLU()
    def forward(self, x, adj):
        y = torch.matmul(adj, x)
        y = torch.matmul(y,self.weight)
        return y


# vanilla GCN encoder
class GCN_encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN_encoder, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        # self.bn1 = nn.BatchNorm1d(output_dim)
        # self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self,x,adj):
        x = self.conv1(x,adj)
        # x = x/torch.sum(x, dim=2, keepdim=True)
        x = self.relu(x)
        # x = self.bn1(x)
        x = self.conv2(x,adj)
        # x = x / torch.sum(x, dim=2, keepdim=True)
        return x
# vanilla GCN decoder
class GCN_decoder(nn.Module):
    def __init__(self):
        super(GCN_decoder, self).__init__()
        # self.act = nn.Sigmoid()
    def forward(self,x):
        # x_t = x.view(-1,x.size(2),x.size(1))
        x_t = x.permute(0,2,1)
        # print('x',x)
        # print('x_t',x_t)
        y = torch.matmul(x, x_t)
        return y


# GCN based graph embedding
# allowing for arbitrary num of nodes
class GCN_encoder_graph(nn.Module):
    def __init__(self,input_dim, hidden_dim, output_dim,num_layers):
        super(GCN_encoder_graph, self).__init__()
        self.num_layers = num_layers
        self.conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        # self.conv_hidden1 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        # self.conv_hidden2 = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.conv_block = nn.ModuleList([GraphConv(input_dim=hidden_dim, output_dim=hidden_dim) for i in range(num_layers)])
        self.conv_last = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        self.act = nn.ReLU()
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
                # init_range = np.sqrt(6.0 / (m.input_dim + m.output_dim))
                # m.weight.data = torch.rand([m.input_dim, m.output_dim]).cuda()*init_range
                # print('find!')
    def forward(self,x,adj):
        x = self.conv_first(x,adj)
        x = self.act(x)
        out_all = []
        out, _ = torch.max(x, dim=1, keepdim=True)
        out_all.append(out)
        for i in range(self.num_layers-2):
            x = self.conv_block[i](x,adj)
            x = self.act(x)
            out,_ = torch.max(x, dim=1, keepdim = True)
            out_all.append(out)
        x = self.conv_last(x,adj)
        x = self.act(x)
        out,_ = torch.max(x, dim=1, keepdim = True)
        out_all.append(out)
        output = torch.cat(out_all, dim = 1)
        output = output.permute(1,0,2)
        # print(out)
        return output

# x = Variable(torch.rand(1,8,10)).cuda()
# adj = Variable(torch.rand(1,8,8)).cuda()
# model = GCN_encoder_graph(10,10,10).cuda()
# y = model(x,adj)
# print(y.size())


def preprocess(A):
    # Get size of the adjacency matrix
    size = A.size(1)
    # Get the degrees for each node
    degrees = torch.sum(A, dim=2)

    # Create diagonal matrix D from the degrees of the nodes
    D = Variable(torch.zeros(A.size(0),A.size(1),A.size(2))).cuda()
    for i in range(D.size(0)):
        D[i, :, :] = torch.diag(torch.pow(degrees[i,:], -0.5))
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    # Create A hat
    # Return A_hat
    A_normal = torch.matmul(torch.matmul(D,A), D)
    # print(A_normal)
    return A_normal



# a sequential GCN model, GCN with n layers
class GCN_generator(nn.Module):
    def __init__(self, hidden_dim):
        super(GCN_generator, self).__init__()
        # todo: add an linear_input module to map the input feature into 'hidden_dim'
        self.conv = GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.act = nn.ReLU()
        # initialize
        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self,x,teacher_force=False,adj_real=None):
        # x: batch * node_num * feature
        batch_num = x.size(0)
        node_num = x.size(1)
        adj = Variable(torch.eye(node_num).view(1,node_num,node_num).repeat(batch_num,1,1)).cuda()
        adj_output = Variable(torch.eye(node_num).view(1,node_num,node_num).repeat(batch_num,1,1)).cuda()

        # do GCN n times
        # todo: try if residual connections are plausible
        # todo: add higher order of adj (adj^2, adj^3, ...)
        # todo: try if norm everytim is plausible

        # first do GCN 1 time to preprocess the raw features

        # x_new = self.conv(x, adj)
        # x_new = self.act(x_new)
        # x = x + x_new

        x = self.conv(x, adj)
        x = self.act(x)

        # x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        # then do GCN rest n-1 times
        for i in range(1, node_num):
            # 1 calc prob of a new edge, output the result in adj_output
            x_last = x[:,i:i+1,:].clone()
            x_prev = x[:,0:i,:].clone()
            x_prev = x_prev
            x_last = x_last
            prob = x_prev @ x_last.permute(0,2,1)
            adj_output[:,i,0:i] = prob.permute(0,2,1).clone()
            adj_output[:,0:i,i] = prob.clone()
            # 2 update adj
            if teacher_force:
                adj = Variable(torch.eye(node_num).view(1, node_num, node_num).repeat(batch_num, 1, 1)).cuda()
                adj[:,0:i+1,0:i+1] = adj_real[:,0:i+1,0:i+1].clone()
            else:
                adj[:, i, 0:i] = prob.permute(0,2,1).clone()
                adj[:, 0:i, i] = prob.clone()
            adj = preprocess(adj)
            # print(adj)
            # print(adj.min().data[0],adj.max().data[0])
            # print(x.min().data[0],x.max().data[0])
            # 3 do graph conv, with residual connection
            # x_new = self.conv(x, adj)
            # x_new = self.act(x_new)
            # x = x + x_new

            x = self.conv(x, adj)
            x = self.act(x)

            # x = x / torch.norm(x, p=2, dim=2, keepdim=True)
        # one = Variable(torch.ones(adj_output.size(0), adj_output.size(1), adj_output.size(2)) * 1.00).cuda().float()
        # two = Variable(torch.ones(adj_output.size(0), adj_output.size(1), adj_output.size(2)) * 2.01).cuda().float()
        # adj_output = (adj_output + one) / two
        # print(adj_output.max().data[0], adj_output.min().data[0])
        return adj_output


# #### test code ####
# print('teacher forcing')
# # print('no teacher forcing')
#
# start = time.time()
# generator = GCN_generator(hidden_dim=4)
# end = time.time()
# print('model build time', end-start)
# for run in range(10):
#     for i in [500]:
#         for batch in [1,10,100]:
#             start = time.time()
#             torch.manual_seed(123)
#             x = Variable(torch.rand(batch,i,4)).cuda()
#             adj = Variable(torch.eye(i).view(1,i,i).repeat(batch,1,1)).cuda()
#             # print('x', x)
#             # print('adj', adj)
#
#             # y = generator(x)
#             y = generator(x,True,adj)
#             # print('y',y)
#             end = time.time()
#             print('node num', i, '  batch size',batch, '  run time', end-start)


