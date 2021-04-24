import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import scipy.sparse as sp
from sklearn.cluster import KMeans

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def to_onehot(self, prelabel):
        k = len(np.unique(prelabel))
        label = np.zeros([prelabel.shape[0], k])
        label[range(prelabel.shape[0]), prelabel] = 1
        label = label.T
        return label

    def square_dist(self, prelabel, feature):
        if sp.issparse(feature):
            feature = feature.todense()
        feature = np.array(feature)
     
        onehot = self.to_onehot(prelabel) # num_labels x nodes. For each label (row) which nodes (columns) have the specific label (value 1)
        m, n = onehot.shape
        count = onehot.sum(1).reshape(m, 1)
        count[count==0] = 1 # to avoid division by zero
     
        mean = onehot.dot(feature) / count
        # mean = onehot.dot(feature)/(count*(count - 1))
        a2 = (onehot.dot(feature * feature) / count).sum(1)
        # a2 = (onehot.dot(feature*feature)/(count*(count - 1))).sum(1)
        pdist2 = np.array(a2 + a2.T - 2*mean.dot(mean.T))
     
        intra_dist = pdist2.trace()
        intra_dist /= m
        return intra_dist

    def drop_groups(self, feature, drop_rate, n_clusters, device):
        feature = feature.detach().cpu()
        u, s, v = sp.linalg.svds(feature, k=n_clusters, which='LM')  # matrix u of SVD is equal to calculating the kernel X*X_T
        kmeans = KMeans(n_clusters=n_clusters).fit(u)
        predict_labels = kmeans.predict(u)
        # print(self.square_dist(predict_labels, feature))

        sel = np.random.permutation(n_clusters)[:int(n_clusters*drop_rate)]
        mask = torch.zeros(feature.shape[0])
        for i in sel:
            mask += predict_labels==i
        mask = (1.0 - mask.unsqueeze(1))
        return mask.to(device)

    def forward(self, input, adj , h0 , lamda, alpha, l, drop_rate=None, n_clusters=None, device=None):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        cur_layer_learner = torch.mm(support, self.weight)
        if drop_rate is None:
            output = theta*cur_layer_learner+(1-theta)*r
        else:
            drop_mask = self.drop_groups(cur_layer_learner, drop_rate, n_clusters, device)
            output = drop_mask * theta * cur_layer_learner/(1.0-drop_rate) + (1-theta) * r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha, variant, 
                 dropnode_rate, device, n_clusters):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda
        self.dropnode_rate = dropnode_rate
        self.device = device
        self.n_clusters = n_clusters

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            if self.training:
                drop_rate = self.dropnode_rate
                n_clusters = self.n_clusters
                layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1,drop_rate,n_clusters,self.device))
            else:
                layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner


if __name__ == '__main__':
    pass






