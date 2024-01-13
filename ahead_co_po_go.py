import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, HGTConv, RGCNConv
import random
from sklearn.metrics import roc_auc_score
from torch_geometric.data import HeteroData
import scipy.io as sio

data2 = sio.loadmat(f'data/CoAID.mat')  # help: data/CoAID.mat; data/politifact.mat; data/gossipcop.mat
node_types = ['news', 'source']
data1 = HeteroData()
data1['news'].x = torch.FloatTensor(data2['X1'])
data1['source'].x = torch.FloatTensor(data2['X2'])
for i in range(data2['edge_index'].shape[1]):
    data2['edge_index'][1, i] -= data1['news'].x.shape[0]
data1[('news', 'to', 'source')].edge_index = torch.LongTensor(data2['edge_index'])
arr = np.copy(data2['edge_index'])
arr[[0, 1], :] = arr[[1, 0], :]
data1[('source', 'to', 'news')].edge_index = torch.LongTensor(arr)
# data1[('news', 'to', 'director')].edge_index = torch.LongTensor(data[('news', 'to', 'director')]['edge_index'])
# data1[('director', 'to', 'news')].edge_index = torch.LongTensor(data[('director', 'to', 'news')]['edge_index'])
gnd = data2['label'][0, :]
gnd1 = gnd[0:data1['news'].x.shape[0]]
gnd2 = gnd[data1['news'].x.shape[0]:]
X = dict()
for node_type in node_types:
    X[node_type] = data1[node_type].x

T = dict()
T['news'] = torch.zeros((data1['news'].x.shape[0], 2))
T['source'] = torch.zeros((data1['source'].x.shape[0], 2))
for i in range(data1['news'].x.shape[0]):
    T['news'][i, 0] = 1
for i in range(data1['source'].x.shape[0]):
    T['source'][i, 1] = 1

A = dict()

flag_news = []
for i in range(data1.x_dict['news'].shape[1]):
    flag_news.append(random.randint(1, 3))
flag_news = np.array(flag_news)
flag_source = []
for i in range(data1.x_dict['source'].shape[1]):
    flag_source.append(random.randint(1, 2))
flag_source = np.array(flag_source)


def multiview(x, num1, num2):
    X_hat_views = dict()
    X_hat_views['news'] = []
    X_hat_views['source'] = []
    index_news = []
    index_source = []
    n = 0
    m = 0
    for i in range(num1):
        while n < len(flag_news):
            if flag_news[n] == i + 1:
                index_news.append(n)
            n += 1
        X_hat_views['news'].append(x['news'][:, index_news])
        index_news = []
        n = 0
    for i in range(num2):
        while m < len(flag_source):
            if flag_source[m] == i + 1:
                index_source.append(m)
            m += 1
        X_hat_views['source'].append(x['source'][:, index_source])
        index_source = []
        m = 0
    return X_hat_views


X_views = multiview(X, 3, 2)
y_dict1 = dict.copy(data1.x_dict)
y_dict1['news'] = X_views['news'][0]
y_dict1['source'] = X_views['source'][0]
y_dict2 = dict.copy(data1.x_dict)
y_dict2['news'] = X_views['news'][1]
y_dict2['source'] = X_views['source'][0]
y_dict3 = dict.copy(data1.x_dict)
y_dict3['news'] = X_views['news'][0]
y_dict3['source'] = X_views['source'][1]
y_dict4 = dict.copy(data1.x_dict)
y_dict4['news'] = X_views['news'][2]
y_dict4['source'] = X_views['source'][0]
y_dict5 = dict.copy(data1.x_dict)
y_dict5['news'] = X_views['news'][1]
y_dict5['source'] = X_views['source'][1]
y_dict6 = dict.copy(data1.x_dict)
y_dict6['news'] = X_views['news'][2]
y_dict6['source'] = X_views['source'][1]
list1 = [y_dict1, y_dict2, y_dict3, y_dict4, y_dict5, y_dict6]

for edge_type in data1.edge_index_dict.keys():
    A[edge_type] = torch.sparse.FloatTensor(data1.edge_index_dict[edge_type],
                                            torch.LongTensor(np.ones(data1.edge_index_dict[edge_type].shape[1])),
                                            torch.Size([data1[edge_type[0]].x.shape[0],
                                                        data1[edge_type[2]].x.shape[0]])).to_dense()


class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, num_view):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_view), requires_grad=True)
        self.weight_node_type = torch.nn.Parameter(torch.randn(2), requires_grad=True)
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin_dict[node_type] = torch.nn.ModuleList()
            for i in range(6):
                if i == 0:
                    self.lin_dict[node_type].append(Linear(y_dict1[node_type].shape[1], hidden_channels))
                if i == 1:
                    self.lin_dict[node_type].append(Linear(y_dict2[node_type].shape[1], hidden_channels))
                if i == 2:
                    self.lin_dict[node_type].append(Linear(y_dict3[node_type].shape[1], hidden_channels))
                if i == 3:
                    self.lin_dict[node_type].append(Linear(y_dict4[node_type].shape[1], hidden_channels))
                if i == 4:
                    self.lin_dict[node_type].append(Linear(y_dict5[node_type].shape[1], hidden_channels))
                if i == 5:
                    self.lin_dict[node_type].append(Linear(y_dict6[node_type].shape[1], hidden_channels))

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data1.metadata(), num_heads, group='sum')
            self.convs.append(conv)

        self.out_dict = torch.nn.ModuleDict()
        for node_type in node_types:
            self.out_dict[node_type] = Linear(hidden_channels, out_channels)
        self.lin = torch.nn.ModuleDict()
        for node_type in node_types:
            self.lin[node_type] = Linear(out_channels, data1.x_dict[node_type].shape[1])
        self.Tlin = torch.nn.ModuleDict()
        for node_type in node_types:
            self.Tlin[node_type] = Linear(out_channels, 2)

    def forward(self, x_dict, edge_index_dict):
        list3 = dict()
        for node_type, _ in x_dict[0].items():
            list3[node_type] = []
        for i in range(6):
            for node_type, x in x_dict[i].items():
                x_dict[i][node_type] = self.lin_dict[node_type][i](x).relu_()

            for conv in self.convs:
                x_dict[i] = conv(x_dict[i], edge_index_dict)

            for node_type, _ in x_dict[0].items():
                list3[node_type].append(self.out_dict[node_type](x_dict[i][node_type]))

        weight_norm = F.softmax(self.weight, dim=0)
        weight_type = F.softmax(self.weight_node_type, dim=0)
        Z_dict = dict()
        A_head = dict()
        X_head = dict()
        t_head = dict()
        T_head = dict()
        for node_type, _ in x_dict[0].items():
            Z_dict[node_type] = weight_norm[0] * list3[node_type][0] + weight_norm[1] * list3[node_type][1] + \
                                weight_norm[2] * list3[node_type][2] + weight_norm[3] * list3[node_type][3] + \
                                weight_norm[4] * list3[node_type][4] + weight_norm[5] * list3[node_type][5]
            X_head[node_type] = self.lin[node_type](Z_dict[node_type])
            t_head[node_type] = self.Tlin[node_type](Z_dict[node_type])
            T_head[node_type] = torch.zeros((t_head[node_type].shape[0], t_head[node_type].shape[1]))
            for i in range(T_head[node_type].shape[0]):
                T_head[node_type][i, 0] = weight_type[0] * t_head[node_type][i, 0]
                T_head[node_type][i, 1] = weight_type[1] * t_head[node_type][i, 1]
            T_head[node_type] = F.softmax(T_head[node_type], dim=1)
        for edge_type in edge_index_dict.keys():
            A_head[edge_type] = torch.sigmoid(torch.mm(Z_dict[edge_type[0]], Z_dict[edge_type[2]].T))

        return A_head, X_head, T_head, weight_norm, weight_type


lr = [0.1, 0.01, 0.001, 0.0001]
for i in lr:
    model = HGT(hidden_channels=64, out_channels=16, num_heads=2, num_layers=2, num_view=6)
    device = torch.device('cpu')
    data1, model = data1.to(device), model.to(device)
    '''
    with torch.no_grad():  # Initialize lazy modules.
        out = model(list1, data.edge_index_dict)
    '''
    optimizer = torch.optim.Adam(model.parameters(), lr=i, weight_decay=0.00001)


    def multiview(x, num1, num2):
        X_hat_views = dict()
        X_hat_views['news'] = []
        X_hat_views['source'] = []
        index_news = []
        index_source = []
        n = 0
        m = 0
        for i in range(num1):
            while n < len(flag_news):
                if flag_news[n] == i + 1:
                    index_news.append(n)
                n += 1
            X_hat_views['news'].append(x['news'][:, index_news])
            index_news = []
            n = 0
        for i in range(num2):
            while m < len(flag_source):
                if flag_source[m] == i + 1:
                    index_source.append(m)
                m += 1
            X_hat_views['source'].append(x['source'][:, index_source])
            index_source = []
            m = 0
        return X_hat_views


    def train():
        model.train()
        optimizer.zero_grad()
        input_list = copy.deepcopy(list1)
        A_hat, X_hat, T_hat, wX, wT = model(input_list, data1.edge_index_dict)
        loss = 0
        loss += torch.norm(A_hat[('news', 'to', 'source')] - A[('news', 'to', 'source')])
        loss += torch.norm(A_hat[('source', 'to', 'news')] - A[('source', 'to', 'news')])
        '''without attribute'''
        for node_type in node_types:
            loss += torch.norm(T_hat[node_type] - T[node_type])
            loss += torch.norm(X_hat[node_type] - X[node_type])
        loss = loss / float(gnd.shape[0])
        loss.backward()
        optimizer.step()
        return float(loss)


    @torch.no_grad()
    def test():
        model.eval()
        input_list = copy.deepcopy(list1)
        out = model(input_list, data1.edge_index_dict)
        ano_score = []

        for i in range(data1['news'].x.shape[0]):
            ano_score.append(0.47 * torch.norm(out[0][('news', 'to', 'source')][i] - A[('news', 'to', 'source')][i]) +
                             0.47 * torch.norm(out[1]['news'][i] - X['news'][i]) + 0.06 * torch.norm(
                out[2]['news'][i] - torch.FloatTensor(np.array(T['news']))))
        for i in range(data1['source'].x.shape[0]):
            ano_score.append(0.47 * torch.norm(out[0][('source', 'to', 'news')][i] - A[('source', 'to', 'news')][i]) +
                             0.47 * torch.norm(out[1]['source'][i] - X['source'][i]) + 0.06 * torch.norm(
                out[2]['source'][i] - torch.FloatTensor(np.array(T['source']))))

        auc = roc_auc_score(gnd, np.array(ano_score) / max(ano_score))
        return auc


    best_auc = 0
    for epoch in range(1, 101):
        loss = train()
        auc = test()
        if auc > best_auc:
            best_auc = auc
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, AUC:{auc:.4f}')
    print(best_auc)
