from torch.nn import LSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MultiHeadGATConv
from torch_geometric.nn import global_max_pool as gmp
 MHGAT model
class MHGATNet(torch.nn.Module):
    def __init__(self, num_features_xd=78, n_output=1, num_features_xt=25,
                     n_filters=32, embed_dim=128, output_dim=128, dropout=0.2, n_heads=8):
        super(MHGATNet, self).__init__()

        # graph layers
        self.gcn1 = MultiHeadGATConv(num_features_xd, num_features_xd, heads=n_heads, dropout=dropout)
        self.gcn2 = MultiHeadGATConv(num_features_xd * n_heads, output_dim, heads=n_heads, dropout=dropout)
        self.fc_g1 = nn.Linear(output_dim, output_dim)

        # cell line feature
        self.conv_xt_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_1 = nn.MaxPool1d(3)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=8)
        self.pool_xt_2 = nn.MaxPool1d(3)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters*2, out_channels=n_filters*4, kernel_size=8)
        self.pool_xt_3 = nn.MaxPool1d(3)
        self.lstm = nn.LSTM(input_size=n_filters, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True)
        self.fc_xt = nn.Linear(lstm_hidden_size, output_dim)
        
        # combined layers
        self.fc1 = nn.Linear(2*output_dim, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        # graph input feed-forward
        # graph input feed-forward
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.dropout(x, p=0.2, training=self.training)
        x = F.elu(self.gcn1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)          # global max pooling
        x = self.fc_g1(x)
        x = self.relu(x)

        # protein input feed-forward:
        target = data.target
        target = target[:,None,:]
        # 1d conv layers
        conv_xt = self.conv_xt_1(target)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_1(conv_xt)
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_2(conv_xt)
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = F.relu(conv_xt)
        conv_xt = self.pool_xt_3(conv_xt)

        # flatten
        #xt = conv_xt.view(-1, conv_xt.shape[1] * conv_xt.shape[2])
        # LSTM
        #lstm_out, (hn, cn) = self.lstm(xt)
        lstm_out, (h_n, h_c) = self.lstm(conv_xt)
        xt = h_n[-1, :, :]
        xt = self.fc_xt(xt)
        xt = self.relu(xt)
        xt = self.dropout(xt)
        
        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.out(xc)
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out, x