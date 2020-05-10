import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from random import shuffle

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging

import random
import shutil
import os
import time
from model import binary_cross_entropy_weight
from utils import *


from trimesh.io.export import export_mesh
from trimesh.base import Trimesh
from trimesh.repair import *
import trimesh



import matplotlib.pyplot as plt

def plotModel(xyz2, edges, N):
    # N = G.number_of_nodes()-1
    Xn=[xyz2[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[xyz2[k][1] for k in range(N)]# y-coordinates
    Zn=[xyz2[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in edges:
        Xe+=[xyz2[e[0]][0],xyz2[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[xyz2[e[0]][1],xyz2[e[1]][1], None]  
        Ze+=[xyz2[e[0]][2],xyz2[e[1]][2], None]
    print('xn: ', len(Xn), 'yn: ', len(Yn),'zn: ', len(Zn),'xe: ', len(Xe), 'ye: ', len(Ye),'ze: ', len(Ze))
    import plotly.plotly as py
    import plotly.graph_objs as go

    trace1=go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125)', width=1),
                hoverinfo='none'
                )

    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                                size=6,
                                colorscale='Viridis',
                                line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )

    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

    layout = go.Layout(
            title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
            dict(
            showarrow=False,
                text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1] miserables.json</a>",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                size=14
                )
                )
            ],    )
    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)

    py.plot(fig, filename='Plot of Edges and Vertices')


def Graph_load_meshes(min_num_nodes = 20, max_num_nodes = 500, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    
    # load data
    # path = '/home/rangeldaroya/Documents/PCMeshDataset/table_train/'
    # path = '/hdd/DocumentFiles/PCMeshDataset/table_train/'
    path = '/home/rangeldaroya/Documents/FinalMeshDataset/'
    # path = '/home/rangeldaroya/Documents/PCMeshDataset/table_test/'

    train_files = []
    f = open(os.path.join(path, 'train_models.txt'))
    for lines in f:
        lines = lines.strip()
        filepath = os.path.join(path, lines, 'model_zip.obj')
        train_files.append(filepath)


    num_files = 34630-1
    # num_files = 1722
    
    graphs = []
    # points = []
    points = None
    max_nodes = 0
    for i in range(num_files):
        # print('train_files[i]: ', train_files[i])
        mesh = trimesh.load(train_files[i])
        verts = np.array(mesh.vertices)
        edges = np.array(mesh.edges)
        if verts.shape[0] > 500:
            # print(verts.shape[0])
            continue
        # faces = np.array(mesh.faces)
        """
        # pts = np.loadtxt(os.path.join(path, 'points_%07d'%i))
        verts = np.loadtxt(os.path.join(path, 'meshv_%07d'%i))
        edges = np.loadtxt(os.path.join(path, 'meshe_%07d'%i))
        faces = np.loadtxt(os.path.join(path, 'meshf_%07d'%i))
        """
        if verts.shape[0] == 212:
            # mesh = Trimesh(vertices=verts, faces=faces)
            # mesh.show()
            print('file  has 212 verts: ', i)
        if verts.shape[0] == 493:
            print('file  has 493 verts: ', i)
        
        G = nx.Graph()
        G.add_edges_from(edges)

        for k in range(verts.shape[0]):
            G.add_node(k, pos=tuple(verts[k]))
        G.remove_nodes_from(list(nx.isolates(G)))
        
        if G.number_of_nodes()>=min_num_nodes and G.number_of_nodes()<=max_num_nodes:
            graphs.append(G)
            # points.append(pts)
            if G.number_of_nodes() > max_nodes:
                max_nodes = G.number_of_nodes()


    print('Loaded')
    return graphs, points

# load ENZYMES and PROTEIN and DD dataset
def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    print('data adj: ', data_adj, data_adj.shape)
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    print('graph_num: ', graph_num)
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        # print('nodes: ', nodes.shape)
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            nx.draw(G_sub, pos=nx.spring_layout(G_sub), with_labels=True)
            # nx.draw_networkx_labels(G2, pos=nx.spring_layout(G2))
            # plt.show()
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs

def test_graph_load_DD():
    graphs, max_num_nodes = Graph_load_batch(min_num_nodes=10,name='DD',node_attributes=False,graph_labels=True)
    shuffle(graphs)
    plt.switch_backend('agg')
    plt.hist([len(graphs[i]) for i in range(len(graphs))], bins=100)
    plt.savefig('figures/test.png')
    plt.close()
    row = 4
    col = 4
    # draw_graph_list(graphs[0:row*col], row=row,col=col, fname='figures/test')
    print('max num nodes',max_num_nodes)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load cora, citeseer and pubmed dataset
def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pkl.load(open("dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G




def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output



def encode_adj(adj, max_prev_node=10, is_full = False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)    #diangonal and upper triangle of square matrix is zero
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]      #makes the adjacency matrix symmetric?
        # print('i:', i,'input start: ', input_start, 'input_end: ', input_end, 'output_start: ', output_start, 'output_end:', output_end, 'adj[i, input_start:input_end]: ', adj[i, input_start:input_end])
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order of line i (makes adjacency matrix descending in order; from nth node to node 1?)

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output



def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full




def encode_adj_full(adj):
    '''
    return a n-1*n-1*2 tensor, the first dimension is an adj matrix, the second show if each entry is valid
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]
    adj_output = np.zeros((adj.shape[0],adj.shape[1],2))
    adj_len = np.zeros(adj.shape[0])

    for i in range(adj.shape[0]):
        non_zero = np.nonzero(adj[i,:])[0]
        input_start = np.amin(non_zero)
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        # write adj
        adj_output[i,0:adj_slice.shape[0],0] = adj_slice[::-1] # put in reverse order
        # write stop token (if token is 0, stop)
        adj_output[i,0:adj_slice.shape[0],1] = 1 # put in reverse order
        # write sequence length
        adj_len[i] = adj_slice.shape[0]

    return adj_output,adj_len

def decode_adj_full(adj_output):
    '''
    return an adj according to adj_output
    :param
    :return:
    '''
    # pick up lower tri
    adj = np.zeros((adj_output.shape[0]+1,adj_output.shape[1]+1))

    for i in range(adj_output.shape[0]):
        non_zero = np.nonzero(adj_output[i,:,1])[0] # get valid sequence
        input_end = np.amax(non_zero)
        adj_slice = adj_output[i, 0:input_end+1, 0] # get adj slice
        # write adj
        output_end = i+1
        output_start = i+1-input_end-1
        adj[i+1,output_start:output_end] = adj_slice[::-1] # put in reverse order
    adj = adj + adj.T
    return adj

def test_encode_decode_adj_full():
########### code test #############
    # G = nx.ladder_graph(10)
    G = nx.karate_club_graph()
    # get bfs adj
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    adj_output, adj_len = encode_adj_full(adj)
    print('adj\n',adj)
    print('adj_output[0]\n',adj_output[:,:,0])
    print('adj_output[1]\n',adj_output[:,:,1])
    # print('adj_len\n',adj_len)
    
    adj_recover = decode_adj_full(adj_output)
    print('adj_recover\n', adj_recover)
    print('error\n',adj_recover-adj)
    print('error_sum\n',np.amax(adj_recover-adj), np.amin(adj_recover-adj))






########## use pytorch dataloader
class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, train_files, hf, max_num_node=None, max_prev_node=None, iteration=20000, tonormverts=True, usezsort=False):
        self.train_files = train_files
        self.hf = hf
        self.numtrain = len(self.train_files)
        self.tonormverts = tonormverts
        self.usezsort = usezsort

        
        
        # self.G_list = G_list
        # self.adj_all = []
        # self.len_all = []
        # self.feat_all = []
        # self.mesh_pts = []
        # for G in G_list:
        #     self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
        #     self.len_all.append(G.number_of_nodes())    
        #     pos = nx.get_node_attributes(G, 'pos')      #maybe can eventually change this to the latent vecor of corresponding point cloud???
        #     xyz = np.array([list(pos[i]) for i in pos])
        #     self.mesh_pts.append(xyz)
        # if Pt_list is not None:
        #     for pt in Pt_list:
        #         self.feat_all.append(pt)
        #     print('feat all: ', len(self.feat_all))
        if max_num_node is None:
            self.n = max(self.len_all)
            print('self.n (max num node) ', self.n)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        # return len(self.adj_all)
        return len(self.train_files)
    def __getitem__(self, idx):
        file_curr = self.train_files[idx]
        # print('file_curr: ', file_curr)
        # file_curr = '/home/rangeldaroya/Desktop/untitled.obj'
        mesh = trimesh.load(file_curr)
        # print(file_curr)
        # verts = self.hf[os.path.join(file_curr, 'verts')][:]
        # edges = self.hf[os.path.join(file_curr, 'edges')][:]
        verts = mesh.vertices
        # print('verts: ', verts.shape)
        edges = mesh.edges
        faces = mesh.faces
        graph_curr = nx.Graph()
        graph_curr.add_edges_from(edges)
        for k in range(verts.shape[0]):
            graph_curr.add_node(k, pos=tuple(verts[k]))
        graph_curr.remove_nodes_from(list(nx.isolates(graph_curr)))
        adj_copy = np.asarray(nx.to_numpy_matrix(graph_curr))

        # graph_curr = self.G_list[idx].copy()
        # adj_copy = self.adj_all[idx].copy()
        # pt_locs = self.feat_all[idx].copy()
        pt_locs = None
        # mesh_pts = self.mesh_pts[idx].copy()
        # pt_dim = pt_locs.shape[-1]
        # print('pt_locs: ', pt_locs.shape, 'pt_dim: ', pt_dim)
        # print('adj_copy: ', adj_copy, adj_copy.shape)
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]   #changes order of adj_copy elements randomly but maintains same group of elements per row and column (same number of 1s in every row and column as before)
        
        pos = nx.get_node_attributes(graph_curr, 'pos')      #maybe can eventually change this to the latent vecor of corresponding point cloud???
        xyz = np.array([list(pos[i]) for i in pos])
        xyz = xyz[x_idx]
        
        if not self.usezsort:
            #""" #removed this starting model_save_mhull_latent_2.
            adj_copy_matrix = np.asmatrix(adj_copy)     #basically, changes orders of nodes (reassigns which node appears first)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))     #run bfs to order the nodes
            # print('x_idx: ', x_idx, x_idx.shape)

            xyz = xyz[x_idx]


            # print('x_idx: ', x_idx, x_idx.shape)
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]   #reorder the nodes based on bfs ordering
            # """
        else:   #use z sort here (sort points based on z value)
            x_idx = np.argsort(xyz[:,2])
            xyz = xyz[x_idx]
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]   #reorder the nodes based on given ordering

        # print('xyz: ', xyz.shape, 'pt_locs: ', pt_locs.shape)
        # plotModel(xyz, np.transpose(adj_copy.nonzero()), xyz.shape[0])

        # np.savetxt('logs/adjorig1-%07d.txt' % (idx), adj_copy, fmt='%d')
        # np.savetxt('logs/ptsorig1-%07d.txt' % (idx), xyz)


        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
        # decoded_adj = decode_adj(adj_encoded)
        # plotModel(xyz, np.transpose(decoded_adj.nonzero()), xyz.shape[0])
        # print('adj_encoded shape: ', adj_encoded.shape, ' adj_copy shape: ', adj_copy.shape)
        # get x and 2y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        # print('pts: ', pt_locs.shape ,'x_batch: ', x_batch.shape, 'adj_encoded: ', adj_encoded.shape, 'adj_copy: ', adj_copy.shape)
        # print('x_batch: ', x_batch, x_batch.shape)      #label (y) is one step advanced than input (x)
        if pt_locs is not None:
            return {'x':x_batch,'y':y_batch, 'len':len_batch, 'pts':pt_locs, 'meshpts':xyz}
        else:
            orignum = xyz.shape[0]
            idx = np.random.randint(0,orignum,(self.n-orignum))
            pts = np.concatenate([xyz, xyz[idx]], axis=0)
            #normalize coordinates
            if self.tonormverts:
                pts = (pts-np.min(pts))/(np.max(pts)-np.min(pts))
                xyz = (xyz-np.min(xyz))/(np.max(xyz)-np.min(xyz))
            return {'x':x_batch,'y':y_batch, 'len':len_batch, 'pts':pts, 'meshpts':xyz, 'faces':np.array(faces), 'origpts':np.array(mesh.vertices)}

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node



########## use pytorch dataloader
class Graph_sequence_sampler_pytorch_nobfs(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.n-1)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

# dataset = Graph_sequence_sampler_pytorch_nobfs(graphs)
# print(dataset[1]['x'])
# print(dataset[1]['y'])
# print(dataset[1]['len'])








########### potential use: an encoder along with the GraphRNN decoder
# preprocess the adjacency matrix
def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = np.sum(A, axis=1)+1

    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(np.power(degrees, -0.5).flatten())
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    A_normal = np.dot(np.dot(D,A_hat),D)
    return A_normal
