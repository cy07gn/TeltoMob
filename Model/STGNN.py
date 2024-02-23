
# this efficient implementation comes from https://github.com/xptree/DeepInf/
class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()

        print('BatchMultiHeadGraphAttention', n_heads, in_channel, num_nodes, dropout)
        self.n_head = n_heads
        self.f_in = num_nodes
        #self.w = nn.Parameter(torch.Tensor(self.n_head, num_nodes, num_nodes))
        self.a_src = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head, num_nodes, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        #nn.init.xavier_uniform_(self.w, gain=1.414)
        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, ch, n, dim = h.size()
        #h_prime = torch.matmul(h, self.w)
        h_prime = h
        attn_src = torch.matmul(h, self.a_src)
        attn_dst = torch.matmul(h, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )

        '''
        ##############
        '''

        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        return output + self.bias, attn

class GAT(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, alpha):
        super(GAT, self).__init__()

        self.dropout = dropout

        self.gat_layer = BatchMultiHeadGraphAttention(
                    n_heads, in_channel, num_nodes, dropout
                )

        #self.norm = torch.nn.InstanceNorm2d(32).cuda()

    def forward(self, x):
        bs,ch,n,dim = x.size()
        #x = self.norm(x) # instance norm for 32 channel
        x, attn = self.gat_layer(x)
        #x = F.elu(x.transpose(1, 2).contiguous().view(bs, ch, n, -1))

        return x


class GATEncoder(nn.Module):
    def __init__(self, kern, dilation_factor, temporal_len, n_heads, in_channel, num_nodes, mlp, mlp2, dropout, alpha):
        super(GATEncoder, self).__init__()

        print('GATEncoder', n_heads, in_channel, num_nodes, dropout, alpha)
        self.gat_net = GAT(n_heads, in_channel, num_nodes, dropout, alpha)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = 32
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

        self.mlp_convs2 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()
        last_channel = n_heads
        print('mlp2', mlp2)
        for out_channel in mlp2:
            self.mlp_convs2.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

        #self.lay_norm = nn.LayerNorm([32, temporal_len, num_nodes])
        #self.lay_norm2 = nn.LayerNorm([n_heads,temporal_len, num_nodes])

        #self.bn_norm1 = nn.BatchNorm2d(8)
        self.bn_norm2 = nn.BatchNorm2d(out_channel)
        self.bn_norm3 = nn.BatchNorm2d(out_channel)

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

        self.mlp = (nn.Conv2d(32,32,(1,kern),dilation=(1,dilation_factor)))

    def forward(self,x):

        bs, ch, n, dim = x.size()

        x_input = x.permute(0,1,3,2)
        x_input_cpy = x_input

        #-------------relu(CNN)-------------#
        for i, conv in enumerate(self.mlp_convs):
            #print('1x_input in', x_input.shape)
            x_input = F.relu((conv(x_input)))
            #print('1x_input out', x_input.shape)
        #-------------relu(CNN)-------------#

        #-------------GAT-------------#
        x_input_cpy2 = x_input

        x_input = self.gat_net(x_input)

        x_input = x_input_cpy2+ self.dropout1(x_input)
        #-------------GAT-------------#

        #print('x_input1', x_input.shape)
        #-------------relu(CNN)-------------#
        for i, conv in enumerate(self.mlp_convs2):
          #print('x_input in', x_input.shape)
          x_input = F.relu((conv(x_input)))
          #print('x_input out', x_input.shape)
        #-------------relu(CNN)-------------#
        #print('x_input', x_input.shape)
        x_input = (x_input_cpy + self.dropout2(x_input)).permute(0,1,3,2)

        x_input = self.bn_norm2(x_input)

        #最後一維度緊收
        x_input = F.relu(self.mlp(x_input))
        x_input = self.bn_norm3(x_input)

        return x_input

import numpy as np

def create_matrix(n, k):
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(k):
            if i-j >= 0:
                mat[i][i-j] = 1
    return mat


def pearson_corr2(tensor): # all: (64,1,n,dim)
  # Input tensor shape: (batch_size, num_nodes, time_steps)
  batch_size, num_nodes, _ = tensor.shape
  tensor = tensor - tensor.mean(dim=2, keepdim=True)
  std = tensor.std(dim=2, keepdim=True)
  tensor = tensor / (std + 1e-8)
  correlation_matrix = torch.matmul(tensor, tensor.transpose(1, 2))
  correlation_matrix = correlation_matrix / (tensor.shape[2] - 1)
  return correlation_matrix


def topK(attn, top_num ):

  # Get the top K values and their indices for each row
  top_k_values, top_k_indices = attn.topk(top_num, dim=3)

  # Create a mask with the same shape as the input tensor, filled with zeros
  mask = torch.zeros_like(attn)

  # Set the top K values in the mask to 1
  mask.scatter_(3, top_k_indices, 1)

  # Multiply the input tensor with the mask to get the result
  attn = attn * mask

  return  attn



# this efficient implementation comes from https://github.com/xptree/DeepInf/
class S_BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, n_heads, num_nodes, dropout, bias=True):
        super(S_BatchMultiHeadGraphAttention, self).__init__()

        print('S_BatchMultiHeadGraphAttention', n_heads, num_nodes, dropout)
        self.n_head = n_heads
        self.f_in = num_nodes
        #self.w = nn.Parameter(torch.Tensor(self.n_head, num_nodes, num_nodes))
        self.a_src = nn.Parameter(torch.Tensor(self.n_head*2, num_nodes, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head*2, num_nodes, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_nodes))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)


        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

        self.W_si = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_si.data, gain=1.414)
        self.W_ei = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.W_ei.data, gain=1.414)

    def forward(self, h, a):

        bs, ch, n, dim = h.size()
        #h_prime = torch.matmul(h, self.w)
        h_prime = h
        attn_src = torch.matmul(torch.tanh(h), self.a_src)
        attn_dst = torch.matmul(torch.tanh(h), self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )

        attn_2 = self.leaky_relu(attn)
        attn_2 = self.softmax(attn_2)
        attn_2 = self.dropout(attn_2)

        attn_2 = abs(self.W_ei)*attn_2+abs(self.W_si)*a

        output_2 = torch.matmul(attn_2, h_prime)

        return output_2, attn

# MutiChannel_GAT(kern, dilation_factor, n_heads, num_nodes, mlp, mlp2, dropout)
class S_MutiChannel_GAT(nn.Module):
    def __init__(self, kern, dilation_factor, n_heads, num_nodes, mlp, mlp2, dropout):
        super(S_MutiChannel_GAT, self).__init__()

        print('S_MutiChannel_GAT', n_heads, num_nodes, dropout)

        self.gat_layer = S_BatchMultiHeadGraphAttention(
            n_heads, num_nodes, dropout
        )
        self.gat_layer2 = S_BatchMultiHeadGraphAttention(
            n_heads, num_nodes, dropout
        )

        self.mlp1 =  nn.Conv2d(in_channels=32,
                                    out_channels=16,
                                    kernel_size=(1, 1))

        self.mlp2 =  nn.Conv2d(in_channels=48,
                                    out_channels=32,
                                    kernel_size=(1, 1))

    def forward(self,x_input, a_f):
        x_input_cpy = x_input

        x = x_input
        x = self.mlp1(x)

        x1, attn = self.gat_layer(x,a_f)
        a_f = torch.matmul(a_f,a_f)
        x2, attn = self.gat_layer2(x1,a_f)

        x_out = self.mlp2(torch.cat([x,x1,x2],dim=1))

        return x_out