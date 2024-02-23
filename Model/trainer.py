class Trainer():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler

        #**fusion**#
        model.a_gmat_list = nn.ModuleList()
        in_channel = 32
        n_heads = 8
        dropout = 0
        alpha = 0.2
        t_len = -1
        model.a_gmat_list.append(
            A_GMAT_module(
              n_heads=n_heads, in_channel= in_channel, num_nodes=t_len, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
            )
        )

        model.test_conv_1 = nn.Conv2d(in_channels=32,
                                             out_channels=32,
                                             kernel_size=(1,1),
                                             bias=True)
        model.test_conv_2 = nn.Conv2d(in_channels=32,
                                             out_channels=32,
                                             kernel_size=(1,1),
                                             bias=True)
        model.test_conv_3 = nn.Conv2d(in_channels=32,
                                             out_channels=32,
                                             kernel_size=(1,1),
                                             bias=True)

        self.model = model
        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.loss = masked_mae
        self.clip = clip
        self.step = step_size
        self.iter = 1
        self.task_level = 1
        self.seq_out_len = seq_out_len
        self.cl = cl



    def train(self, input, real_val, idx=None):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(input, idx=idx)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)

        predict = self.scaler.inverse_transform(output)

        if self.iter%self.step==0 and self.task_level<=self.seq_out_len:
            self.task_level +=1
            print("### cl learning\n iter",self.iter,"\niter%step",self.iter%self.step,"\ntask_level",self.task_level)
            print("# predict len:", len(predict[:, :, :, :self.task_level]))

        if self.cl:
            loss = masked_mae(predict[:, :, :, :self.task_level], real[:, :, :, :self.task_level], 0.0)
        else:
            loss = masked_mae(predict, real, 0.0)

        loss.backward()

        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

        self.optimizer.step()
        mae = masked_mae(predict,real,0.0).item()
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        smape = masked_smape(predict,real,0.0).item()
        self.iter += 1
        return mae,mape,rmse,smape

    def eval(self, input, real_val):
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)
        real = torch.unsqueeze(real_val,dim=1)

        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        mape = masked_mape(predict,real,0.0).item()
        rmse = masked_rmse(predict,real,0.0).item()
        smape = masked_smape(predict,real,0.0).item()
        return loss.item(),mape,rmse,smape

