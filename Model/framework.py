

class gginet(nn.Module):
    def __init__(self, model_type,  num_nodes, device, predefined_A=None,kernel_set=None, static_feat=None, dropout=0.3, node_dim=40, dilation_exponential=1, conv_channels=32, residual_channels=32, skip_channels=64, end_channels=128, seq_length=12, in_dim=2, out_dim=12, layers=3, propalpha=0.05, tanhalpha=3, layer_norm_affline=True):
        super(gginet, self).__init__()

        self.model_type = model_type

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.layers = layers
        self.seq_length = seq_length

        self.t_gat1 = nn.ModuleList()
        self.t_gat2 = nn.ModuleList()

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.s_gat = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))




        if self.model_type == "GMAT":
            # Paepr eq 11: R=1+(c-1)(q^m -1)/(q -1).
            kernel_size = 7
            if dilation_exponential>1:
                self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
            else:
                self.receptive_field = layers*(kernel_size-1) + 1

        print("# Model Type", self.model_type)
        print("# receptive_field", self.receptive_field)
        i=0
        if dilation_exponential>1:
            rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            rf_size_i = i*layers*(kernel_size-1)+1
        new_dilation = 1

        self.receptive_field = 10
        target_len = self.receptive_field


        self.t_len = []
        for j in range(1,layers+1):


            if self.model_type == "GMAT":
                if dilation_exponential > 1:
                    rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
                else:
                    rf_size_j = rf_size_i+j*(kernel_size-1)

            if j % 2 == 1:
                new_dilation = 1
            elif j % 2 == 0:
                new_dilation = 2
            dilation_factor = new_dilation
            kern = 2

            in_channel = 32
            n_heads = 8
            dropout = 0
            alpha = 0.2
            self.t_gat1.append(
                GATEncoder(
                  kern= kern, dilation_factor=dilation_factor, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[16,n_heads],mlp2=[16,32], dropout=dropout, alpha=alpha
                )
            )

            self.t_gat2.append(
                GATEncoder(
                  kern= kern, dilation_factor=dilation_factor, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[16,n_heads],mlp2=[16,32], dropout=dropout, alpha=alpha
                )
            )

            target_len -= dilation_factor
            self.t_len.append(target_len)

            if self.model_type == "GMAT" :
                '''
                # skip_convs #
                (0): Conv2d(32, 64, kernel_size=(1, 13), stride=(1, 1))
                (1): Conv2d(32, 64, kernel_size=(1, 7), stride=(1, 1))
                (2): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
                '''
                if self.seq_length>self.receptive_field:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, target_len)))
                else:
                    self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                    out_channels=skip_channels,
                                                    kernel_size=(1, target_len)))
            dilation_factor = 1
            n_heads = 8

            self.s_gat.append(S_MutiChannel_GAT(kern, dilation_factor, n_heads, target_len, [24,16,8], [16,24,32], dropout))

            #####   GCN   ##### END

            #####   Normalization   ##### START
            if self.model_type == "GMAT":
                if self.seq_length>self.receptive_field:
                    print('1', self.seq_length - rf_size_j + 1)
                    self.norm.append(LayerNorm((residual_channels, num_nodes, target_len),elementwise_affine=layer_norm_affline))
                else:
                    print('2', self.receptive_field - rf_size_j + 1)
                    self.norm.append(LayerNorm((residual_channels, num_nodes, target_len),elementwise_affine=layer_norm_affline))
            #####   Normalization   ##### END

            new_dilation *= dilation_exponential



        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)




        #####   SKIP layer   ##### START
        if self.model_type == "GMAT":
            '''
            (skip0): Conv2d(2, 64, kernel_size=(1, 19), stride=(1, 1))
            (skipE): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
            '''
            if self.seq_length > self.receptive_field:
                self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
                self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

            else:
                self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
                self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)
        #####   SKIP layer   ##### END

        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, idx=None):
        seq_len = input.size(3)

        #**fusion**#
        #assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'
        if hasattr(self, 'a_gmat_list'): # 代表有fusion
          #print("in pretrain!!!!!!!!!!!!")
          self.seq_length = input.shape[3] # 讓(...,4)補成(...,10)跟原本ST對齊


        # Step0: 檢查receptive_field, 不足則padding0
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))

        #print('input2', input.shape, 'self.seq_length', self.seq_length)

        # Step1: turn([64, 2, 207, 19]) to ([64, 32, 207, 19])
        x = self.start_conv(input)

        #**fusion**#
        if hasattr(self, 'a_gmat_list'): # 代表有fusion
          skip = 0

          x_all = []
          for i in range(len(adj_edges)):
              target_node = F.elu(self.test_conv_1(x[:,:,i].unsqueeze(2)))

              for out in outgoing[i]:
                  outgoing_nodes = F.elu(self.test_conv_1(x[:,:, out].unsqueeze(2)))
                  outgoing_nodes_representations = F.elu(self.test_conv_2(outgoing_nodes-target_node))
                  #print('outgoing_nodes_representations', outgoing_nodes_representations.shape)

                  ingoing_nodes = F.elu(self.test_conv_1(x[:,:, ingoing[i]]))
                  intgoing_nodes_representations = F.elu(self.test_conv_3(target_node-ingoing_nodes))
                  #print('intgoing_nodes_representations', intgoing_nodes_representations.shape)


                  outgoing_nodes_representations = ((outgoing_nodes_representations))
                  intgoing_nodes_representations = ((intgoing_nodes_representations))


                  tmp_x = torch.cat([outgoing_nodes_representations,intgoing_nodes_representations],dim=2)

                  tmp_x = self.a_gmat_list[0](tmp_x)
                  x_all.append(tmp_x[:,:,0].unsqueeze(2))
          x = torch.cat(x_all, dim=2)

        else:
          skip = self.skip0(F.dropout(input, self.dropout, training=self.training))


        #    -- START #
        # Layers : 3層 : 19->13->7->1 (取決於TCN取的維度)
        for i in range(self.layers):

            # Step2: Temporal Model --START #
            # 為上一層輸出, ex:  [64, 32, 207, 19] -> [64, 32, 207, 13] -> [64, 32, 207, 7]-> [64, 32, 207, 1]
            residual = x

            #x = x.permute(0,1,3,2)

            filter = self.t_gat1[i](x)
            filter = torch.tanh(filter)

            gate = self.t_gat2[i](x)
            gate = torch.sigmoid(gate)

            x = filter * gate
            if self.model_type == "GMAT":
                x = F.dropout(x, self.dropout, training=self.training)
            # Step2: Temporal Model --END #

            # Step3: Skip after TCN --START #
            s = x

            # fusion output:([64, 32, 207, 13])
            # skip_convsL 0:([64, 64, 207, 1])
            s = self.skip_convs[i](s)


            skip = s + skip

            # Step3: Skip after TCN --END #

            x = self.s_gat[i](x, self.predefined_A[0])

            # x 經過dilated處理後, 會減少feature維度, ex: 19->13->7->1
            # 而residual為上一層輸出, 維度為: 19, 13 ...
            # 所以需要配合x進行維度調整: [:, :, :, -x.size(3):], 然後進行elemenet-wise相加
            x = x + residual[:, :, :, -x.size(3):]

            if self.model_type == "GMAT":
                if idx is None:
                    x = self.norm[i](x,self.idx)
                else:
                    x = self.norm[i](x,idx)
            # Step4: GCN --END #

        #    -- END #

        if self.model_type == "GMAT":
            #(skipE): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1))
            skip = self.skipE(x) + skip

        #sys.exit()
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)

        return x