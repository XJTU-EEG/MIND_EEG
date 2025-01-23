import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
from utils import SE_Block, normalize, CBAM, feature_trans
from Codebook import Codebook
from einops import rearrange


class get_adj(nn.Module):
    def __init__(self, in_features, num_chan, num_embedding, embedding_dim):
        super(get_adj, self).__init__()
        self.in_features = in_features
        self.num_chan = num_chan
        self.num_embedding = num_embedding
        self.embedding_dim = embedding_dim

        self.p = nn.Parameter(torch.empty(self.num_chan, self.num_chan))
        self.bias = nn.Parameter(torch.empty(self.num_chan, self.in_features))

        self.q = nn.Parameter(torch.empty(self.in_features, self.in_features))
        self.theta = nn.Parameter(torch.empty(self.in_features, self.num_chan * self.in_features))
        self.at = nn.ELU()
        self.senet = SE_Block(self.in_features)
        self.cb = Codebook(self.in_features, self.num_embedding, self.num_chan, self.embedding_dim, 0.25)
        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.p)
        nn.init.zeros_(self.bias)
        nn.init.xavier_normal_(self.q)
        nn.init.xavier_normal_(self.theta)

    def forward(self, x):
        o = torch.einsum("ij, bjd->bid", self.p, x) + self.bias
        g = self.at(torch.matmul(torch.matmul(o, self.q), self.theta))
        adj = normalize(g.reshape(g.shape[0], g.shape[1], g.shape[1], -1).permute(0, 3, 1, 2))
        vq_loss, adj_recon, usage, codebook = self.cb(adj)
        adj_recon = self.at(self.senet(adj_recon) + adj_recon)
        adj_recon = normalize(self.at(torch.sum(adj_recon, dim=1)))
        return vq_loss, adj_recon, usage, codebook



class res_gcn(nn.Module):
    def __init__(self, in_features, num_chan, dp=0.05):
        super(res_gcn, self).__init__()
        self.in_features = in_features
        self.num_chan = num_chan
        self.weight1 = Parameter(torch.empty(self.in_features, self.in_features + 10))
        self.weight2 = Parameter(torch.empty(self.in_features + 15, self.in_features + 25))
        self.CBAM1 = CBAM(self.num_chan)
        self.CBAM2 = CBAM(self.num_chan)
        self.dropout = nn.Dropout(p=dp)
        self.dp = nn.Dropout(0.1)
        self.at = nn.SELU()
        self.init_weight()

    def init_weight(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj):
        adj1 = self.dp(adj)
        x1_ = torch.einsum("b i j, j d -> b i d", x, self.weight1)
        x1 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x1_))
        x1 = torch.concat((x, x1), 2)
        x1 = self.CBAM1(x1)
        x11 = self.dropout(x1)
        # adj2 = self.dp(adj)
        x2_ = torch.einsum("b i j, j d -> b i d", x11, self.weight2)
        x2 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x2_))
        x2 = torch.concat((x1, x2), 2)
        x2 = self.CBAM2(x2)
        return x2


class res_gcn2(nn.Module):
    def __init__(self, in_features, k):
        super(res_gcn2, self).__init__()
        self.in_features = in_features
        self.weight1 = Parameter(torch.empty(self.in_features, self.in_features + 5))
        self.weight2 = Parameter(torch.empty(self.in_features + 5, self.in_features + 10))
        self.CBAM1 = CBAM(7)
        self.CBAM2 = CBAM(7)
        self.dropout = nn.Dropout(p=0.05)
        self.dp = nn.Dropout(0.1)
        self.at = nn.SELU()
        self.init_weight()

    def init_weight(self):
        stdv1 = 1. / math.sqrt(self.weight1.size(1))
        self.weight1.data.uniform_(-stdv1, stdv1)
        stdv2 = 1. / math.sqrt(self.weight2.size(1))
        self.weight2.data.uniform_(-stdv2, stdv2)

    def forward(self, x, adj):
        adj1 = self.dp(adj)
        x1_ = torch.einsum("b i j, j d -> b i d", x, self.weight1)
        x1 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x1_))
        x1 = self.CBAM1(x1) + x1
        x11 = self.dropout(x1)
        # adj2 = self.dp(adj)
        x2_ = torch.einsum("b i j, j d -> b i d", x11, self.weight2)
        x2 = self.at(torch.einsum("b i j, b j d->b i d", adj1, x2_))
        x2 = self.CBAM2(x2) + x2

        return x2


class State_encoder(Module):
    def __init__(self, in_features, out_features):
        super(State_encoder, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.get_adj = get_adj(5, 62, 64, 1024)
        self.res_gcn = res_gcn(5, 62, 0.1)

    def forward(self, x):
        vq_loss, adj_recon, usage, codebook_local = self.get_adj(x)
        x = self.res_gcn(x, adj_recon)
        return vq_loss, x, usage, codebook_local


class Self_attention(nn.Module):
    def __init__(self, in_features, out_feature):
        super(Self_attention, self).__init__()

        self.in_features = in_features
        self.out_features = out_feature
        self.num_heads = 5
        self.selu = nn.SELU()

        self.get_qk = nn.Linear(self.in_features, self.in_features * 2)
        nn.init.xavier_uniform_(self.get_qk.weight)

        self.equ_weights = Parameter(torch.FloatTensor(self.num_heads))
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.bias = Parameter(torch.FloatTensor(self.out_features))
        self.reset_parameters()

    def forward(self, h):
        h_ = self.cal_att_matrix(h)
        output = torch.matmul(h_, self.weight) + self.bias
        return output

    def cal_att_matrix(self, feature):
        qk = rearrange(self.get_qk(feature), "b n (h d qk) -> (qk) b h n d", h=self.num_heads, qk=2)
        queries, keys = qk[0], qk[1]
        values = feature
        dim_scale = (queries.size(-1)) ** -0.5
        dots = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) * dim_scale
        at = torch.einsum("b g i j -> b i j", dots)
        adj_matrix = self.dropout_layer(at)
        at = F.softmax(adj_matrix / 0.3, dim=2)

        out_feature = torch.einsum('b i j, b j d -> b i d', at, values)

        return out_feature

    def dropout_layer(self, at):
        att_subview_, _ = at.sort(2, descending=True)
        att_threshold = att_subview_[:, :, att_subview_.size(2) // 6]
        att_threshold = rearrange(att_threshold, 'b i -> b i 1')
        att_threshold = att_threshold.repeat(1, 1, at.size()[2])
        at[at < att_threshold] = -1e-7
        return at

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.equ_weights.data.uniform_(-stdv, stdv)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)


class Region(nn.Module):
    def __init__(self, subgraph_num, trainable_vector):
        super(Region, self).__init__()
        self.subgraph_num = subgraph_num

        self.at = nn.ELU()

        self.graph_list = self.sort_subgraph(subgraph_num)
        self.emb_size = 50

        self.get_adj1 = get_adj(5, 5, 32, 16)
        self.get_adj2 = get_adj(5, 9, 32, 32)
        self.get_adj3 = get_adj(5, 9, 32, 32)
        self.get_adj4 = get_adj(5, 25, 32, 256)
        self.get_adj5 = get_adj(5, 9, 32, 32)
        self.get_adj6 = get_adj(5, 9, 32, 32)
        self.get_adj7 = get_adj(5, 12, 32, 64)

        self.get_adj_co = get_adj(50, 7, 128, 32)

        self.res_gcn1 = res_gcn(5, 5)
        self.res_gcn2 = res_gcn(5, 9)
        self.res_gcn3 = res_gcn(5, 9)
        self.res_gcn4 = res_gcn(5, 25)
        self.res_gcn5 = res_gcn(5, 9)
        self.res_gcn6 = res_gcn(5, 9)
        self.res_gcn7 = res_gcn(5, 12)

        self.res_gcn_co = res_gcn2(50, 2)

        self.softmax = nn.Softmax(dim=0)
        self.att_softmax = nn.Softmax(dim=1)

        self.trainable_vec1 = Parameter(torch.FloatTensor(trainable_vector))
        self.weight1 = Parameter(torch.FloatTensor(self.emb_size, 80))
        self.reset_parameters()

    def reset_parameters(self):
        stdv1 = 1. / math.sqrt(self.trainable_vec1.size(0))
        self.trainable_vec1.data.uniform_(-stdv1, stdv1)
        self.weight1.data.uniform_(-stdv1, stdv1)

    def forward(self, x):
        re_x1, re_x2, re_x3, re_x4, re_x5, re_x6, re_x7 = self.region_x(x)
        loss1, adj1, usage1, codebook_1 = self.get_adj1(re_x1)
        x1 = self.at(self.res_gcn1(re_x1, adj1))

        loss2, adj2, usage2, codebook_2 = self.get_adj2(re_x2)
        x2 = self.at(self.res_gcn2(re_x2, adj2))

        loss3, adj3, usage3, codebook_3 = self.get_adj3(re_x3)
        x3 = self.at(self.res_gcn3(re_x3, adj3))

        loss4, adj4, usage4, codebook_4 = self.get_adj4(re_x4)
        x4 = self.at(self.res_gcn4(re_x4, adj4))

        loss5, adj5, usage5, codebook_5 = self.get_adj5(re_x5)
        x5 = self.at(self.res_gcn5(re_x5, adj5))

        loss6, adj6, usage6, codebook_6 = self.get_adj6(re_x6)
        x6 = self.at(self.res_gcn6(re_x6, adj6))

        loss7, adj7, usage7, codebook_7 = self.get_adj7(re_x7)
        x7 = self.at(self.res_gcn7(re_x7, adj7))

        region_x = torch.concat((x1, x2, x3, x4, x5, x6, x7), dim=1)

        coarsen_x = self.att_coarsen(region_x, self.graph_list, self.weight1)

        loss_co, adj_co, usage_co, codebook_co = self.get_adj_co(coarsen_x)
        x_co = self.res_gcn_co(coarsen_x, adj_co)

        usage = [usage1, usage2, usage3, usage4, usage5, usage6, usage7, usage_co]
        loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + 2 * loss_co
        codebook = [codebook_1, codebook_2, codebook_3, codebook_4, codebook_5, codebook_6, codebook_7, codebook_co]

        return loss, x_co, usage, codebook

    def att_coarsen(self, features, graph_list, weight):
        coarsen_feature = []

        idx_head = 0
        for index_length in graph_list:
            idx_tail = idx_head + index_length
            sub_feature = features[:, idx_head:idx_tail]

            feature_with_weight = torch.einsum('b j g, g h -> b j h', sub_feature, weight)
            feature_T = rearrange(feature_with_weight, 'b j h -> b h j')
            att_weight_matrix = torch.einsum('b j h, b h i -> b j i', feature_with_weight, feature_T)
            att_weight_vector = torch.sum(att_weight_matrix, dim=2)
            att_vec = self.att_softmax(att_weight_vector)
            sub_feature_ = torch.einsum('b j, b j g -> b g', att_vec, sub_feature)

            coarsen_feature.append(rearrange(sub_feature_, "b g -> b 1 g"))
            idx_head = idx_tail

        coarsen_features = torch.cat(tuple(coarsen_feature), 1)
        return coarsen_features

    def region_x(self, features):
        features = feature_trans(self.subgraph_num, features)
        x1 = features[:, 0:5]
        x2 = features[:, 5:14]
        x3 = features[:, 14:23]
        x4 = features[:, 23:48]
        x5 = features[:, 48:57]
        x6 = features[:, 57:66]
        x7 = features[:, 66:78]
        return x1, x2, x3, x4, x5, x6, x7

    def sort_subgraph(self, subgraph_num):
        subgraph_7 = [5, 9, 9, 25, 9, 9, 12]
        graph = None
        if subgraph_num == 7:
            graph = subgraph_7

        return graph





