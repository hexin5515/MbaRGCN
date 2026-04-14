import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch import Tensor
import torch_geometric
from torch_geometric.nn import GATConv, GCNConv, ChebConv, GCN2Conv, SGConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix, dropout_adj
import scipy.sparse as sp
from scipy.special import comb
from torch_sparse import SparseTensor
from utils import one_hot
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch.nn import Linear, ModuleList, Module, Dropout, ReLU, GELU, Sequential

from einops import rearrange, repeat, einsum

class GCN_mamba_liner(torch.nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, with_bias=False):
        super(GCN_mamba_liner, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = input @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
class GCN_mamba_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_mamba_Net, self).__init__()
        self.dropout = args.dropout
        self.args = args
        if args.dataset in ['Polblogs']:
            self.lin1 = GCN_mamba_liner(len(dataset[0].y), args.d_model, with_bias=args.bias)
        else:
            self.lin1 = GCN_mamba_liner(dataset.num_features, args.d_model, with_bias=args.bias)
        self.layer_num = args.layer_num
        self.mamba = GCN_mamba_block(args)
        self.norm_1 = RMSNorm(args.d_model)
        self.norm_2 = RMSNorm(args.d_model)
        self.lin2 = GCN_mamba_liner(args.d_model, dataset.num_classes, with_bias=args.bias)
        self.bn_1 = torch.nn.BatchNorm1d(args.d_model)
        self.bn_2 = torch.nn.BatchNorm1d(args.d_model)
        self.reset_parameters()
    def getmamba(self):
        return self.mamba
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x_input = data.x
        adj_t = data.adj_t
        
        x_input = self.lin1(x_input)
        x = self.norm_1(x_input)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        all_layers_output = self.mamba(x, adj_t, self.args.layer_num)
        output = all_layers_output[:,-1,:] + x

        output = self.norm_2(output)
        all_layers_output = F.relu(all_layers_output)
        all_layers_output = F.dropout(all_layers_output, p=self.dropout, training=self.training)
        output = F.relu(output)
        output = F.dropout(output, p=self.dropout, training=self.training)

        y = self.lin2(output)

        return F.log_softmax(y, dim=-1)
    
    def normalize_adj_tensor(self, adj):
        mx = adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
    
class GCN_mamba_block(torch.nn.Module):
    def __init__(self, args):
        super(GCN_mamba_block, self).__init__()
        self.args = args
        self.in_proj = GCN_mamba_liner(args.d_model, args.d_inner * 2, with_bias=args.bias)

        # x_proj takes in `x` and outputs the input-specific Δ, B, C
        self.x_proj = GCN_mamba_liner(args.d_inner, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.mamba_dropout = args.mamba_dropout
        # dt_proj projects Δ from dt_rank to d_in
        self.dt_proj = GCN_mamba_liner(args.dt_rank, args.d_inner, with_bias=args.bias)
        self.bns = torch.nn.ModuleList()
        for i in range(args.layer_num-1):
            self.bns.append(torch.nn.BatchNorm1d(args.d_inner))

        # 这是模型原本的初始化
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)

        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = GCN_mamba_liner(args.d_inner, args.d_model, with_bias=args.bias)

        self.in_act_net = ActionNet(args)
        self.out_act_net = ActionNet(args)
        self.reset_parameters()

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()
        self.in_act_net.reset_parameters()
        self.out_act_net.reset_parameters()


    def forward(self, x, adj, layer_num):
        """
        Mamba block forward. This looks the same as Figure 3 in Section 3.4 in the Mamba paper [1].
    
        Args:
            x: shape (b, l, d)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d)
        
        Official Implementation:
            class Mamba, https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py#L119
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
        """

        (b, d) = x.shape
        expanded_x = x.unsqueeze(1)
        x = expanded_x.expand(b, layer_num, d)
        (b, l, d) = x.shape
        

        y = self.ssm(x, self.args, adj)

        output = self.out_proj(y)

        return output
    
    def ssm(self, x, args, adj):
        """Runs the SSM. See:
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        Args:
            x: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
    
        Returns:
            output: shape (b, l, d_in)

        Official Implementation:
            mamba_inner_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L311
            
        """
        (d_in, n) = self.A_log.shape

        # Compute ∆ A B C D, the state space parameters.
        #     A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
        #     ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
        #                                  and is why Mamba is called **selective** state spaces)
        # 这是原本的初始化方式
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)

        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)
        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D, args, adj)
        
        return y

    
    def selective_scan(self, u, delta, A, B, C, D, args, adj):
        """Does selective scan algorithm. See:
            - Section 2 State Space Models in the Mamba paper [1]
            - Algorithm 2 in Section 3.2 in the Mamba paper [1]
            - run_SSM(A, B, C, u) in The Annotated S4 [2]

        This is the classic discrete state space formula:
            x(t + 1) = Ax(t) + Bu(t)
            y(t)     = Cx(t) + Du(t)
        except B and C (and the step size delta, which is used for discretization) are dependent on the input x(t).
    
        Args:
            u: shape (b, l, d_in)    (See Glossary at top for definitions of b, l, d_in, n...)
            delta: shape (b, l, d_in)
            A: shape (d_in, n)
            B: shape (b, l, n)
            C: shape (b, l, n)
            D: shape (d_in,)
    
        Returns:
            output: shape (b, l, d_in)
    
        Official Implementation:
            selective_scan_ref(), https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/selective_scan_interface.py#L86
            Note: I refactored some parts out of `selective_scan_ref` out, so the functionality doesn't match exactly.
            
        """
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # Discretize continuous parameters (A, B)
        # - A is discretized using zero-order hold (ZOH) discretization (see Section 2 Equation 4 in the Mamba paper [1])
        # - B is discretized using a simplified Euler discretization instead of ZOH. From a discussion with authors:
        #   "A is the more important term and the performance doesn't change much with the simplification on B"

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        
        # Perform selective scan (see scan_SSM() in The Annotated S4 [2])
        # Note that the below is sequential, while the official implementation does a much faster parallel scan that
        # is additionally hardware-aware (like FlashAttention).
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = [] 
        for i in range(args.layer_num):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            x = x.reshape(b, d_in * n)
            x = adj @ x
            x = x.reshape(b, d_in, n)
            x = F.relu(x)
            x = F.dropout(x, p=self.mamba_dropout, training=self.training)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)

        y = y + u * D

        return y
    
    def create_edge_weight(self, edge_index, keep_in_prob: Tensor, keep_out_prob: Tensor) -> Tensor:
        u, v = edge_index
        edge_in_prob = keep_in_prob[v]
        edge_out_prob = keep_out_prob[u]
        return edge_in_prob * edge_out_prob
    
class GCN_mamba_Net_pro_max(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_mamba_Net_pro_max, self).__init__()
        self.dropout = args.dropout
        self.args = args
        if args.dataset in ['Polblogs']:
            self.lin1 = GCN_mamba_liner(len(dataset[0].y), args.d_model, with_bias=args.bias)
        else:
            self.lin1 = GCN_mamba_liner(dataset.num_features, args.d_model, with_bias=args.bias)
        self.layer_num = args.layer_num
        self.mamba = GCN_mamba_block_pro_max(args)
        self.norm_1 = RMSNorm(args.d_model)
        self.norm_2 = RMSNorm(args.d_model)

        self.lin2 = GCN_mamba_liner(args.d_model, dataset.num_classes, with_bias=args.bias)
        self.lin3 = GCN_mamba_liner(args.d_model * 2, dataset.num_classes, with_bias=args.bias)
        self.bn_1 = torch.nn.BatchNorm1d(args.d_model)
        self.bn_2 = torch.nn.BatchNorm1d(args.d_model)
        self.reset_parameters()
    def getmamba(self):
        return self.mamba
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, args, data, is_val=False):
        x_input = data.x
        adj_t = data.adj_t
        
        x_input = self.lin1(x_input)
        x = self.norm_1(x_input)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # alpha = 0.05
        # features = [x]
        # xi = x
        # for i in range(args.layer_num - 1):
        #     xi = adj_t @ xi
        #     # xi = (1-alpha)*xi+alpha*x
        #     features.append(xi)

        # import ipdb;ipdb.set_trace()

        all_layers_output = self.mamba(x, adj_t, self.args.layer_num) + x.unsqueeze(1)
        all_layers_output = F.relu(all_layers_output)
        all_layers_output = F.dropout(all_layers_output, p=self.dropout, training=self.training)

        # output = all_layers_output[:,-1,:]
        output = all_layers_output

        y = self.lin2(output)
        # import ipdb;ipdb.set_trace()
        return F.log_softmax(y, dim=-1)
        # if is_val == False:
        #     cluster_features = torch.mm(data.cluster_train_id.t(), x[data.idx_train]) / data.cluster_train_id.sum(0).unsqueeze(1)
        #     x1 = cluster_features[data.cluster_train_id.argmax(1)]
        #     # output = torch.cat((torch.cat((x[data.idx_train], x1), 1), torch.cat((x1, x[data.idx_train]), 1)), 0)
        #     # output_prompt = torch.cat((x[data.idx_train], x1), 1)
        #     # output_prompt = self.lin3(output_prompt)
        #     output = x[data.idx_train]
        #     y = self.lin2(output)
        #     return F.log_softmax(y, dim=-1) # , F.log_softmax(output_prompt, dim=-1)

        # else:
        #     cluster_id = one_hot(data.index.cpu(), args.cluster).to(args.device)
        #     cluster_index = data.idx_train + data.idx_val
        #     cluster_features = torch.mm(cluster_id[cluster_index].t(), x[cluster_index]) / cluster_id[cluster_index].sum(0).unsqueeze(1)  
        #     x1 = cluster_features[cluster_id.argmax(1)]
        #     # output = torch.cat((x, x1), 1)
        #     output = x
        #     y = self.lin2(output)
        #     return F.log_softmax(y, dim=-1)
    
    def normalize_adj_tensor(self, adj):
        mx = adj
        rowsum = mx.sum(1)
        r_inv = rowsum.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        mx = r_mat_inv @ mx
        mx = mx @ r_mat_inv
        return mx
    
class GCN_mamba_block_pro_max(torch.nn.Module):
    def __init__(self, args):
        super(GCN_mamba_block_pro_max, self).__init__()
        self.args = args
        self.in_proj = GCN_mamba_liner(args.d_model, args.d_inner * 2, with_bias=args.bias)
        self.x_proj = GCN_mamba_liner(args.d_inner, args.dt_rank + args.d_state * 2, with_bias=args.bias)
        self.mamba_dropout = args.mamba_dropout
        self.dt_proj = GCN_mamba_liner(args.dt_rank, args.d_inner, with_bias=args.bias)

        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)

        self.A_log = torch.nn.Parameter(torch.log(A))
        self.D = torch.nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = GCN_mamba_liner(args.d_inner, args.d_model, with_bias=args.bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.in_proj.reset_parameters()
        self.x_proj.reset_parameters()
        self.dt_proj.reset_parameters()
        self.out_proj.reset_parameters()


    def forward(self, x, adj, layer_num):

        (b, d) = x.shape
        expanded_x = x.unsqueeze(1)
        x = expanded_x.expand(b, layer_num, d)
        (b, l, d) = x.shape
        
        y = self.ssm(x, self.args, adj)

        output = self.out_proj(y)

        return output
    
    def ssm(self, x, args, adj):
        (d_in, n) = self.A_log.shape
        
        A = -torch.exp(self.A_log.float())  # shape (d_in, n)
        D = self.D.float()

        x_dbl = self.x_proj(x)  # (b, l, dt_rank + 2*n)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)  # delta: (b, l, dt_rank). B, C: (b, l, n)

        delta = F.softplus(self.dt_proj(delta))  # (b, l, d_in)
        
        y = self.selective_scan(x, delta, A, B, C, D, args, adj)
        
        return y
    
    def selective_scan(self, u, delta, A, B, C, D, args, adj):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))

        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = [] 
        for i in range(args.layer_num):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            x = x.reshape(b, d_in * n)
            x = adj @ x
            x = x.reshape(b, d_in, n)
            x = F.relu(x)
            x = F.dropout(x, p=self.mamba_dropout, training=self.training)
            ys.append(y)

        y = torch.stack(ys, dim=1)  # shape (b, l, d_in)
        y = y + u * D

        return y
    
class RMSNorm(torch.nn.Module):
    def __init__(self,
                 d_model: int,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

        return output