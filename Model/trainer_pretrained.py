
#**fusion**#
class Trainer_pretrained():
    def __init__(self, model, lrate, wdecay, clip, step_size, seq_out_len, scaler, device, cl=True):
        self.scaler = scaler

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


    def eval(self, input, real_val):
        #print('@Trainer_pretrained, input', input.shape)
        #print('@Trainer_pretrained, real_val', real_val.shape)
        self.model.eval()
        output = self.model(input)
        output = output.transpose(1,3)

        #**fusion**#
        output =   torch.cat([output,real_val[:,1,:output.size()[2]].unsqueeze(1)],dim=1)
        #print('@Trainer_pretrained, output2', output.shape)
        return output

