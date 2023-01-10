import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class GNN(MessagePassing):
    def __init__(self, hidden_channels, num_features, num_layers, num_classes):
        super().__init__(aggr='add')

        torch.manual_seed(1234567)


        # for loop for multiple layers
        self.num_layers = num_layers
        self.linear_list = [Linear(num_features, hidden_channels, bias=False)]
        for i in range(self.num_layers-1):
            self.linear_list.append(Linear(hidden_channels, hidden_channels, bias=False))
        
        self.batch_num_list = [torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False)]
        for i in range(self.num_layers-1):
            self.batch_num_list.append(torch.nn.BatchNorm1d(hidden_channels,track_running_stats=False,affine=False))

        self.bias_list = [Parameter(torch.Tensor(hidden_channels))]
        for i in range(self.num_layers-1):
            self.bias_list.append(Parameter(torch.Tensor(hidden_channels)))

        self.last_linear = Linear(hidden_channels, num_classes)

        self.reset_parameters()


    def reset_parameters(self):

        for i in range(self.num_layers):
            self.linear_list[i].reset_parameters()
        
        for i in range(self.num_layers):
            self.batch_num_list[i].reset_parameters()

        for i in range(self.num_layers-1):
            self.bias_list[i].data.zero_()

        self.last_linear.reset_parameters()

    def forward(self, x, edge_index, device):
        # x has shape [N, features]
        # edge_index has shape [2, E]

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        for i in range(self.num_layers):

            x = self.linear_list[i](x)

            x = self.propagate(edge_index, x=x)

            x += self.bias_list[i]

            x = self.batch_num_list[i](x)

            x = x.relu()  

        # do final linear transformation
        x = self.last_linear(x)          

        return x