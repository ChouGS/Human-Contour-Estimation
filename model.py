from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
# from torchvision import models
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

import pdb


# def get_pretrained_model(name='resnet50'):
#     model_dict = {
#         'resnet18': 'models.resnet18(pretrained=True)',
#         'resnet34': 'models.resnet34(pretrained=True)',
#         'resnet50': 'models.resnet50(pretrained=True)',
#         'resnet101': 'models.resnet101(pretrained=True)',
#         'resnet152': 'models.resnet152(pretrained=True)'
#     }
#     pretrained_model = eval(model_dict[name])

#     return pretrained_model


# def get_edge_index(edge_index, branch_id, num_nodes):
#     return edge_index + branch_id * num_nodes


def get_edge_index(inputs):
    # Input: batch_size * num_nodes * num_nodes
    assert len(inputs.shape) == 3 and inputs.shape[1] == inputs.shape[2]
    num_branches, num_nodes = inputs.shape[:2]
    inputs[inputs < torch.max(inputs) * 0.5] = 0
    nnz = torch.nonzero(inputs, as_tuple=False)
    nnz[:, 1] += nnz[:, 0] * num_nodes
    nnz[:, 2] += nnz[:, 0] * num_nodes

    edge_index = nnz[:, 1:].permute((1, 0))
    return edge_index


class AdjacencyNet(nn.Module):
    def __init__(self, num_nodes, node_feature_dim):
        super(AdjacencyNet, self).__init__()
        self.fc = nn.Linear(node_feature_dim, num_nodes)

        self.similarity_pooling = nn.MaxPool2d(kernel_size=(num_nodes,
                                                            num_nodes),
                                               stride=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, inputs, num_branches):
        feature_dim = inputs.shape[1]
        inputs = inputs.reshape(num_branches, -1, feature_dim)

        adj = self.fc(inputs)
        adj = self.relu(adj)
        adj = adj / torch.norm(adj, p=2, dim=(1, 2), keepdim=True)
        node_norm = torch.norm(inputs, p=2, dim=2, keepdim=True)
        inputs = inputs / node_norm

        similarity = torch.matmul(inputs, inputs.permute((0, 2, 1)))
        adj_loss = torch.mean(torch.sum(torch.square(
            torch.add(adj, -similarity)),
                                        axis=(1, 2)),
                              axis=0)

        edge_index = get_edge_index(adj)

        return adj, edge_index, adj_loss


class GCNModel(nn.Module):
    def __init__(self, num_nodes, node_feature_dim, std_edge):
        super(GCNModel, self).__init__()
        self.gcn1 = GCNConv(node_feature_dim, node_feature_dim)
        self.gcn2 = GCNConv(node_feature_dim, node_feature_dim)
        self.gcn3 = GCNConv(node_feature_dim, node_feature_dim)
        self.gcn4 = GCNConv(node_feature_dim, node_feature_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.adj = AdjacencyNet(num_nodes, node_feature_dim)
        self.readout = nn.Conv1d(in_channels=node_feature_dim,
                                 out_channels=2,
                                 kernel_size=1,
                                 stride=1)
        self.std_edge = std_edge

    def forward(self, inputs):
        assert len(inputs.shape) == 3
        num_branches, num_nodes, feature_dim = inputs.shape
        edges_per_branch = self.std_edge.shape[1]
        inputs = inputs.reshape(-1, feature_dim)

        edge_list = np.arange(0, num_branches * edges_per_branch,
                              1).astype(np.int32)
        edge_offsets = np.floor_divide(edge_list, edges_per_branch) * num_nodes
        edge_offsets = np.repeat(np.expand_dims(edge_offsets, 1), 2, axis=1).T
        edge_index = np.tile(self.std_edge.detach().cpu().numpy(),
                             (1, num_branches)) + edge_offsets
        edge_index = torch.from_numpy(edge_index).cuda()
        # GCN layer 1
        # a, loss1 = self.adj(inputs, num_branches)
        x = self.gcn1(inputs, edge_index)
        x = self.relu1(x)

        # GCN layer 2
        a, edge_index, loss1 = self.adj(x, num_branches)
        # a = a.long()
        # pdb.set_trace()
        # x = torch.matmul(a, x)
        x = self.gcn2(x, edge_index)
        x = self.relu2(x)

        # GCN layer 3
        a, edge_index, loss2 = self.adj(x, num_branches)
        # a = a.long()
        # x = torch.matmul(a, x)
        x = self.gcn3(x, edge_index)
        x = self.relu3(x)

        # GCN layer 4
        x = self.gcn4(x, self.std_edge)
        x = x.reshape((num_branches, num_nodes, feature_dim))

        x = x.permute((0, 2, 1))
        x = self.readout(x)
        x = x.permute((0, 2, 1))

        loss = loss1 + loss2

        return x, loss


class SAGEModel(nn.Module):
    def __init__(self, num_nodes, node_feature_dim, std_edge):
        super(SAGEModel, self).__init__()
        self.sage1 = SAGEConv(node_feature_dim, node_feature_dim)
        self.sage2 = SAGEConv(node_feature_dim, node_feature_dim)
        self.sage3 = SAGEConv(node_feature_dim, node_feature_dim)
        self.sage4 = SAGEConv(node_feature_dim, node_feature_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.att_w = AdjacencyNet(num_nodes, node_feature_dim)
        self.readout = nn.Conv1d(in_channels=node_feature_dim,
                                 out_channels=2,
                                 kernel_size=1,
                                 stride=1)
        self.std_edge = std_edge

    def forward(self, inputs):
        assert len(inputs.shape) == 3
        num_branches, num_nodes, feature_dim = inputs.shape
        edges_per_branch = self.std_edge.shape[1]
        inputs = inputs.reshape(-1, feature_dim)

        edge_list = np.arange(0, num_branches * edges_per_branch,
                              1).astype(np.int32)
        edge_offsets = np.floor_divide(edge_list, edges_per_branch) * num_nodes
        edge_offsets = np.repeat(np.expand_dims(edge_offsets, 1), 2, axis=1).T
        edge_index = np.tile(self.std_edge.detach().cpu().numpy(),
                             (1, num_branches)) + edge_offsets
        edge_index = torch.from_numpy(edge_index).cuda()
        # GCN layer 1
        # a, loss1 = self.adj(inputs, num_branches)
        x = self.sage1(inputs, edge_index)
        x = self.relu1(x)

        # GCN layer 2
        a, edge_index, loss1 = self.adj(x, num_branches)
        # a = a.long()
        # pdb.set_trace()
        # x = torch.matmul(a, x)
        x = self.sage2(x, edge_index)
        x = self.relu2(x)

        # GCN layer 3
        a, edge_index, loss2 = self.adj(x, num_branches)
        # a = a.long()
        # x = torch.matmul(a, x)
        x = self.sage3(x, edge_index)
        x = self.relu3(x)

        # GCN layer 4
        x = self.sage4(x, self.std_edge)
        x = x.reshape((num_branches, num_nodes, feature_dim))

        x = x.permute((0, 2, 1))
        x = self.readout(x)
        x = x.permute((0, 2, 1))

        return x


class GATModel(nn.Module):
    def __init__(self, num_nodes, node_feature_dim, std_edge, num_heads=2):
        super(GATModel, self).__init__()
        self.heads = num_heads
        self.gat1 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat2 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat3 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat4 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat5 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat6 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat7 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat8 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat9 = GATConv(node_feature_dim,
                            node_feature_dim//self.heads,
                            heads=self.heads)
        self.gat10 = GATConv(node_feature_dim,
                             node_feature_dim//self.heads,
                             heads=self.heads)
        self.gat11 = GATConv(node_feature_dim,
                             node_feature_dim//self.heads,
                             heads=self.heads)
        self.gat12 = GATConv(node_feature_dim,
                             node_feature_dim//self.heads,
                             heads=self.heads)

        self.readout1 = nn.Linear(node_feature_dim, 16)
        self.readout2 = nn.Linear(16, 2)

        self.std_edge = std_edge

    def forward(self, inputs):
        # Produce a whole-size graph (for batch training)
        assert len(inputs.shape) == 3
        num_branches, num_nodes, feature_dim = inputs.shape
        edges_per_branch = self.std_edge.shape[1]
        inputs = inputs.reshape(-1, feature_dim)

        edge_list = np.arange(0, num_branches * edges_per_branch,
                              1).astype(np.int32)
        edge_offsets = np.floor_divide(edge_list, edges_per_branch) * num_nodes
        edge_offsets = np.repeat(np.expand_dims(edge_offsets, 1), 2, axis=1).T
        edge_index = np.tile(self.std_edge.detach().cpu().numpy(),
                             (1, num_branches)) + edge_offsets
        edge_index = torch.from_numpy(edge_index).cuda()

        # GAT layer 1
        x = self.gat1(inputs, edge_index)  # , heads=self.heads)
        inputs = inputs + x

        # GAT layer 2
        x = self.gat2(inputs, edge_index)  # , heads=self.heads)
        inputs = inputs + x

        # GAT layer 3
        x = self.gat3(inputs, edge_index)  # , heads=self.heads)
        inputs = inputs + x

        # GAT layer 4
        x = self.gat4(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 5
        x = self.gat5(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 6
        x = self.gat6(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 7
        x = self.gat7(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 8
        x = self.gat8(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 9
        x = self.gat9(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 10
        x = self.gat10(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 11
        x = self.gat11(inputs, edge_index)
        inputs = inputs + x

        # GAT layer 12
        x = self.gat12(inputs, edge_index)
        outputs = inputs + x

        _, feature_dim = outputs.shape
        outputs = outputs.reshape((num_branches, num_nodes, feature_dim))

        outputs = self.readout1(outputs)
        outputs = self.readout2(outputs)

        return outputs


class NodeFeatureNet(nn.Module):
    def __init__(self, in_dim, out_dim, in_channel, out_channel):
        super(NodeFeatureNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, 128, (3, 3), padding=1)
        self.pool1 = nn.AvgPool2d((2, 2), (2, 2))
        self.conv2 = nn.Conv2d(128, out_channel, (3, 3), padding=1)
        self.pool2 = nn.AvgPool2d((2, 2), (2, 2))
        self.fc1 = nn.Linear(in_dim, out_dim)
        # self.fc2 = nn.Linear(256, out_dim)
        self.relu = nn.ReLU()

        # self.conv1 = nn.Conv1d(in_channels=in_channel,
        #                        out_channels=out_channel,
        #                        kernel_size=1,
        #                        stride=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        bsize, channels, _, _ = x.shape
        x = x.reshape(bsize, channels, -1)
        x = self.fc1(x)
        x = self.relu(x)

        return x


class MyModel(nn.Module):
    def __init__(self, cfg, std_edge, gnn_mode='GCN'):
        super(MyModel, self).__init__()
        assert gnn_mode in ['GCN', 'SAGE', 'GAT']
        self.fnet = NodeFeatureNet(in_dim=cfg.MODEL.IN_DIM,
                                   out_dim=cfg.MODEL.NODE_FEATURE_DIM,
                                   in_channel=cfg.MODEL.IN_CHANNEL,
                                   out_channel=cfg.MODEL.NUM_NODES)
        self.std_edge = std_edge
        self.gnn = None
        if gnn_mode == 'GCN':
            self.gnn = GCNModel(num_nodes=cfg.MODEL.NUM_NODES,
                                node_feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
                                std_edge=std_edge)
        elif gnn_mode == 'SAGE':
            self.gnn = SAGEModel(num_nodes=cfg.MODEL.NUM_NODES,
                                 node_feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
                                 std_edge=std_edge)
        elif gnn_mode == 'GAT':
            self.gnn = GATModel(num_nodes=cfg.MODEL.NUM_NODES,
                                node_feature_dim=cfg.MODEL.NODE_FEATURE_DIM,
                                std_edge=std_edge)

    def forward(self, inputs):
        # print(inputs.shape)
        nodes = self.fnet(inputs)
        # pdb.set_trace()
        gnn_outp = self.gnn(nodes)

        return gnn_outp


# fnet 输出 b*n*f (64 * 63 * 128) Tensor，代表64张独立的图，每张63个顶点，每个顶点128维feature
# 现需要将其填到一张大图中
# 期望大图拥有64*63个顶点，分别隶属于64个连通分支中
# 需要增加的功能：
#   - 对连通分支进行编号
#   - 给定编号和一组edge_index（范围限定为0~62），返回对应连通分支对应的edge_index
#   - merge所有edge_index
#   - 实现edge_weights的加入
#   - 实现edge_weights loss的计算
