import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *
from nystrom_attention import NystromAttention
import admin_torch

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class TransFusion(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.translayer = TransLayer(norm_layer, dim)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        x = self.translayer(x)
        return x[:, :x1.shape[1], :]

class BottleneckTransFusion(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.n_bottlenecks = n_bottlenecks
        self.attn1 = TransLayer(nn.LayerNorm, dim=dim)
        self.attn2 = TransLayer(nn.LayerNorm, dim=dim)
        self.bottleneck = torch.rand((1,n_bottlenecks,dim)).cuda()

    def forward(self, x1, x2):
        b, seq, dim_len = x1.shape
        bottleneck = torch.cat([self.bottleneck, x2], dim=1)
        bottleneck = self.attn2(bottleneck)[:,:self.n_bottlenecks, :]
        x = torch.cat([x1, bottleneck], dim=1)
        x = self.attn1(x)
        return x[:, :seq, :]

class AddFusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class DropX2Fusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()

    def forward(self, x1, x2):
        return x1

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class RoutingNetwork(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        super(RoutingNetwork, self).__init__()
        self.bnum = branch_num
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.clsfer = nn.Linear(dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        x1, x2 = self.fc1(x1), self.fc2(x2)
        x = x1.mean(dim=1) + x2.mean(dim=1)
        logits = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=1)
        return logits

class MoME(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256,):
        super().__init__()
        self.TransFusion = TransFusion(norm_layer, dim)
        self.BottleneckTransFusion = BottleneckTransFusion(n_bottlenecks, norm_layer, dim)
        self.AddFusion = AddFusion(norm_layer, dim)
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        self.routing_network = RoutingNetwork(4, dim=dim)
        self.routing_dict = {
            0: self.TransFusion,
            1: self.BottleneckTransFusion,
            2: self.AddFusion,
            3: self.DropX2Fusion,
        }

    def forward(self, x1, x2, hard=False):
        logits = self.routing_network(x1, x2, hard)
        if hard:
            corresponding_net_id = torch.argmax(logits, dim=1).item()
            x = self.routing_dict[corresponding_net_id](x1, x2)
        else:
            x = torch.zeros_like(x1)
            for branch_id, branch in self.routing_dict.items():
                x += branch(x1, x2)
        return x

class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.residual_attn = admin_torch.as_module(8)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,    # number of landmarks
            pinv_iterations = 6,    # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual = True,         # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout = 0.1
        )

    def forward(self, x):
        x = self.residual_attn(x,self.attn(self.norm(x)))
        return x

class MoMETransformer(nn.Module):
    def __init__(self, n_bottlenecks, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,):
        super(MoMETransformer, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}

        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)

        ### MoMEs
        self.MoME_genom1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_patho1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_genom2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])
        self.MoME_patho2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2])

        ### Classifier
        self.multi_layer1 = TransLayer(dim=size[2])
        self.cls_multimodal = torch.rand((1, size[2])).cuda()
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        
        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)

        h_path_bag = h_path_bag.unsqueeze(0)
        h_omic_bag = h_omic_bag.unsqueeze(0)

        h_path_bag = self.MoME_patho1(h_path_bag, h_omic_bag, hard=True)
        h_omic_bag = self.MoME_genom1(h_omic_bag, h_path_bag, hard=True)

        h_path_bag = self.MoME_patho2(h_path_bag, h_omic_bag, hard=True)
        h_omic_bag = self.MoME_genom2(h_omic_bag, h_path_bag, hard=True)

        h_path_bag = h_path_bag.squeeze()
        h_omic_bag = h_omic_bag.squeeze()
        
        h_multi = torch.cat([self.cls_multimodal, h_path_bag, h_omic_bag], dim=0).unsqueeze(0)
        h = self.multi_layer1(h_multi)[:,0,:]
        
        ### Survival Layer
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        attention_scores = {}
        
        return hazards, S, Y_hat, attention_scores
