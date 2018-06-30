import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, n_input, n_output, n_class, anchor_affnty, batch_size=20):
        super(GCN, self).__init__()

        self.anchor_affnty = anchor_affnty
        self.batch_size = batch_size

        self.gconv1 = nn.Linear(n_input, 2048)
        self.BN1 = nn.BatchNorm1d(2048)
        self.act1 = nn.ReLU()

        self.gconv2 = nn.Linear(2048, 2048)
        self.BN2 = nn.BatchNorm1d(2048)
        self.act2 = nn.ReLU()

        self.gconv3 = nn.Linear(2048, n_output)
        self.BN3 = nn.BatchNorm1d(n_output)
        self.act3 = nn.Tanh()

        self.fc = nn.Linear(n_output, n_class)

    def forward(self, x, in_affnty, out_affnty):
        out = self.gconv1(x)
        out = in_affnty.mm(out)
        out = self.BN1(out)
        out = self.act1(out)

        # block 2
        out = self.gconv2(out)
        out = self.anchor_affnty.mm(out)
        out = self.BN2(out)
        out = self.act2(out)

        # block 3
        out = self.gconv3(out)
        out = out_affnty.mm(out)
        out = self.BN3(out)
        out = self.act3(out)

        out_masked = out[:self.batch_size,:]
        pred = self.fc(out_masked)

        return out, pred

class GCN_stack(nn.Module):
    def __init__(self, n_input, n_output, n_class, anchor_affnty,
                 batch_size=40, depth=1, n_dim=2048):
        super(GCN_stack, self).__init__()
        self.anchor_affnty = anchor_affnty
        self.batch_size = batch_size

        # input block
        self.gconv1 = nn.Linear(n_input, n_dim)
        self.BN1 = nn.BatchNorm1d(n_dim)
        self.act1 = nn.ReLU()
    
        # infer block
        self.gconvs = nn.ModuleList()
        self.BNs = nn.ModuleList()
        self.acts = nn.ModuleList()

        for _ in range(depth):
            self.gconvs.append(nn.Linear(n_dim, n_dim))
            self.BNs.append(nn.BatchNorm1d(n_dim))
            self.acts.append(nn.ReLU())

        # output block
        self.gconv3 = nn.Linear(n_dim, n_output)
        self.BN3 = nn.BatchNorm1d(n_output)
        self.act3 = nn.Tanh()

        # prediction block 
        self.fc = nn.Linear(n_output, n_class)
    
    def forward(self, x, in_affnty, out_affnty, s):
        out = self.gconv1(x)
        out = in_affnty.mm(out)
        out = self.BN1(out)
        out = self.act1(out)
 
        for gconv, BN, act in zip(self.gconvs, self.BNs, self.acts):
            r = out
            out = gconv(out)
            out = self.anchor_affnty.mm(out)
            out = BN(out)
            out += r
            out = act(out)    

        out = self.gconv3(out)
        out = out_affnty.mm(out)
        out = self.BN3(out)
        out = self.act3(out)

        out_masked = out[:s,:]
        pred = self.fc(out_masked)

        return out, pred
        




