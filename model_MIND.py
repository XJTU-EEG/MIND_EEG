import torch.nn as nn
from layer_MIND import State_encoder, Self_attention, Region
import torch
from einops import rearrange

class Mind(nn.Module):
    def __init__(self, args):
        super(Mind, self).__init__()

        self.args = args
        self.nclass = args.n_class
        self.dropout = args.dropout

        self.selu = nn.SELU()
        self.at = nn.SELU()

        self.embed = nn.Linear(50, 60)

        self.encoder = State_encoder(args.in_feature, 50)
        self.region = Region(subgraph_num=7, trainable_vector=78)

        self.sa = Self_attention(60, 85)

        self.mlp1 = nn.Linear(69*145, 2048)
        self.mlp2 = nn.Linear(2048, 512)
        self.mlp3 = nn.Linear(512, self.nclass)

        self.BN1 = nn.BatchNorm1d(5)
        self.BN2 = nn.BatchNorm1d(7)
        self.BN3 = nn.BatchNorm1d(62)
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(self.dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x):
        if self.args.dataset == 'SEED5':
            x = rearrange(x, 'b (h c) -> b h c', h=62)
        vq_loss, local_x1, usage_local, codebook_local = self.encoder(x)
        # res_local = self.at(local_x1)
        res_local = self.at(self.embed(local_x1))

        loss, region_x, usage, codebook_region = self.region(x)
        region_x = self.at(region_x)
        usage = [usage_local] + usage
        codebook = [codebook_local] + codebook_region
        x = torch.concat((res_local, region_x), dim=1)

        x_ = self.at(self.sa(x))

        x = torch.concat((x, x_), dim=2)

        x = x.view(x.size(0), -1)

        x = self.at(self.bn1(self.mlp1(x)))
        x = self.dropout(x)
        x = self.at(self.bn2(self.mlp2(x)))
        x = self.dropout(x)
        x = self.mlp3(x)
        return x, vq_loss, loss, usage, codebook