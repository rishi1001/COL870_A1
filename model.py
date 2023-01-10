import torch
from torch.nn import Linear
import torch.nn.functional as F


class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_features, num_layers, num_classes):
        super().__init__()
        torch.manual_seed(1234567)

        # for loop for multiple layers
        self.num_layers = num_layers
        self.linear_list = [Linear(num_features, hidden_channels)]
        for i in range(self.num_layers-1):
            self.linear_list.append(Linear(hidden_channels, hidden_channels))
        
        self.batch_num_list = [torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)]
        for i in range(self.num_layers-1):
            self.batch_num_list.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False))

        self.last_linear = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, device):
        
        A = torch.sparse.FloatTensor(edge_index, torch.ones(edge_index.shape[1]).to(device), torch.Size([x.shape[0], x.shape[0]])).to(device)
        # print(A.shape)

        # add self loop
        A = A + torch.sparse.FloatTensor(torch.eye(x.shape[0]).nonzero().t().to(device), torch.ones(x.shape[0]).to(device), torch.Size([x.shape[0], x.shape[0]])).to(device)

        for i in range(self.num_layers):
            # do gnn mean aggregation
            x = torch.spmm(A, x).to(device)            # TODO check this
            # do linear transformation
            x = self.linear_list[i].to(device)(x)
            # do batch norm
            x = self.batch_num_list[i](x)
            # do relu
            x = x.relu()

        # do final linear transformation
        x = self.last_linear(x)
        # x = x.relu()       # TODO remove this

        return x