
class A_GMAT_base(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, bias=True):
        super(A_GMAT_base, self).__init__()

        print('A_GMAT_base', n_heads, in_channel, num_nodes, dropout)
        self.n_head = n_heads
        self.f_in = num_nodes
        self.a_src = nn.Parameter(torch.Tensor(self.n_head, 10, 1))
        self.a_dst = nn.Parameter(torch.Tensor(self.n_head, 10, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(10))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.a_src, gain=1.414)
        nn.init.xavier_uniform_(self.a_dst, gain=1.414)

    def forward(self, h):
        bs, ch, n, dim = h.size()
        h_prime = h
        attn_src = torch.matmul(h, self.a_src)
        attn_dst = torch.matmul(h, self.a_dst)
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(
            0, 1, 3, 2
        )
        attn = self.leaky_relu(attn)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.matmul(attn, h_prime)
        return output + self.bias, attn

class A_GMAT(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, dropout, alpha):
        super(A_GMAT, self).__init__()

        self.dropout = dropout

        self.layer = A_GMAT_base(
                    n_heads, in_channel, num_nodes, dropout
                )

    def forward(self, x):
        bs,ch,n,dim = x.size()
        x, attn = self.layer(x)

        return x


class A_GMAT_module(nn.Module):
    def __init__(self, n_heads, in_channel, num_nodes, mlp, mlp2, dropout, alpha):
        super(A_GMAT_module, self).__init__()
        print('A_GMAT_module', n_heads, in_channel, num_nodes, dropout, alpha)
        self.net = A_GMAT(n_heads, in_channel, num_nodes, dropout, alpha)

        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = 32
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

        self.mlp_convs2 = nn.ModuleList()
        self.mlp_bns2 = nn.ModuleList()
        last_channel = n_heads
        for out_channel in mlp2:
            self.mlp_convs2.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.6)
    def forward(self,x):
        #print('in MGAT x.shape',x.shape)
        #sys.exit()
        bs, ch, n, dim = x.size()

        x_input = x

        x_all = []
        x_input_cpy = x_input

        for i, conv in enumerate(self.mlp_convs):
          x_input = ((conv(x_input)))

        x_input_cpy2 = x_input
        x_input = self.net(x_input)
        x_input = x_input_cpy2+ self.dropout1(x_input)

        for i, conv in enumerate(self.mlp_convs2):
          x_input = F.relu((conv(x_input)))

        x_input = x_input_cpy+ self.dropout2(x_input)

        return x_input
