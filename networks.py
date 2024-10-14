from typing import Optional
from torch_geometric.typing import OptTensor

import torch
import torch.nn.functional as F
import numpy as np
import math

from torch.nn import Parameter,Linear, ModuleList
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian, remove_self_loops


def sin_taylor_coefficients(K, omega, D):
    # Initialize a tensor to store the coefficients
    coefficients = torch.zeros((K, D + 1), dtype=torch.float32)
    
    # Compute the coefficients for sin(n * omega * pi)
    x_values = (torch.arange(1, K + 1) * omega * torch.pi).unsqueeze(1)  # Shape (k, 1)
    
    # Calculate odd-order terms (1, 3, 5, ...) up to D
    odd_orders = torch.arange(1, D + 1, step=2, dtype=torch.float32)  # Odd orders up to D
    signs = (-1) ** ((odd_orders - 1) // 2)  # Alternating signs for odd powers

    # Compute factorials manually for odd orders
    factorials = torch.tensor([math.factorial(int(order)) for order in odd_orders], dtype=torch.float32)

    # Populate the coefficients matrix
    for n in range(K):
        coefficients[n, 1:D+1:2] = signs * (x_values[n] ** odd_orders) / factorials

    return coefficients


def cos_taylor_coefficients(K, omega, D):
    # Initialize a tensor to store the coefficients
    coefficients = torch.zeros((K, D + 1), dtype=torch.float32)
    
    # Compute the coefficients for cos(n * omega * pi)
    x_values = (torch.arange(1, K + 1) * omega * torch.pi).unsqueeze(1)  # Shape (k, 1)
    
    # Calculate even-order terms (0, 2, 4, ...) up to D
    even_orders = torch.arange(0, D + 1, step=2, dtype=torch.float32)  # Even orders up to D
    signs = (-1) ** (even_orders // 2)  # Alternating signs for even powers

    # Compute factorials manually for even orders
    factorials = torch.tensor([math.factorial(int(order)) for order in even_orders], dtype=torch.float32)

    # Populate the coefficients matrix
    for n in range(K):
        coefficients[n, 0:D+1:2] = signs * (x_values[n] ** even_orders) / factorials

    return coefficients


class Trigo_prop(MessagePassing):
    '''
    propagation class for TPGNN
    '''

    def __init__(self, K, omega, D, device, **kwargs):
        super(Trigo_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.omega = omega
        self.D = D

        TEMP = torch.nn.init.xavier_normal_(torch.empty(1, K))
        self.alpha = Parameter(TEMP.float())

        TEMP = torch.nn.init.xavier_normal_(torch.empty(1, K))
        self.beta = Parameter(TEMP.float())

        self.sincoeff = sin_taylor_coefficients(K, omega, D).to(device)

        self.coscoeff = cos_taylor_coefficients(K, omega, D).to(device)
        
    def __norm__(self, edge_index, num_nodes: Optional[int],
                 edge_weight: OptTensor, normalization: Optional[str] = "sym",
                 lambda_max: OptTensor = None, dtype: Optional[int] = None,
                 batch: OptTensor = None):

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)

        if lambda_max is None:
            lambda_max = 2.0 * edge_weight.max()
        elif not isinstance(lambda_max, torch.Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(edge_weight == float('inf'), 0)

        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 fill_value=-1.,
                                                 num_nodes=num_nodes)
        assert edge_weight is not None

        return edge_index, edge_weight

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, omega={}, D={})'.format(self.__class__.__name__, self.K, 
                                                 self.omega, self.D)

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = self.__norm__(edge_index, x.size(0), edge_weight, dtype=x.dtype)
        # x_list, eta_list = [], []
        Tx = x

        COEFF = self.alpha@self.sincoeff + self.beta@self.coscoeff # coefficient mixing

        hidden = COEFF[0,0] * Tx
        
        for d in range(1, self.D + 1):
            Tx = self.propagate(edge_index, x=Tx, norm=norm)
            hidden = hidden + COEFF[0,d] * Tx
            
        return hidden + x


class TFGNN(torch.nn.Module):
    def __init__(self, args):
        super(TFGNN, self).__init__()
        self.lin1 = Linear(args.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, args.num_classes)

        self.prop1 = Trigo_prop(args.K, args.omega, args.D, args.device)

        self.dprate = args.dprate
        self.dropout = args.dropout

    def forward(self, feature, edges):
        x, edge_index = feature, edges
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)  
        
        x = F.dropout(x, p=self.dprate, training=self.training)
        x = self.prop1(x, edge_index)
        
        return F.log_softmax(x, dim=1)