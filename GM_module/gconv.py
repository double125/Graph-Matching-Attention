import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class Gconv(nn.Module):
    """
    (Intra) graph convolution operation, with single convolutional layer
    """
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.dropout = nn.Dropout(p=0.1)

        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.utils.weight_norm(self.u_fc)

    def forward(self, A, x, x_mask=None, norm=True):
        A = A.type(torch.cuda.FloatTensor)
        if x_mask is not None:
            A = A.masked_fill(x_mask, -1e9)
        if norm is True:
            A = F.softmax(A, dim=-1)
            #A = F.normalize(A, p=1, dim=-1)

        ux = F.relu(self.u_fc(x))
        x = x + torch.bmm(A, ux)   # has size (bs, N, num_outputs)
        return x

class Siamese_Gconv(nn.Module):
    """
    Perform graph convolution on two input graphs (g1, g2)
    """
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        return emb1, emb2

