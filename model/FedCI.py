import torch.nn as nn
import torch
import pickle
from lib.utils import SpatialEmbedding
class BasicBlock(nn.Module):
    
    def __init__(self, args, dX, dropout=0.1):
        super(BasicBlock, self).__init__()
        self.linear = nn.Linear(dX, dX)
        self.norm = nn.LayerNorm(dX)
        # self.norm = nn.BatchNorm2d(12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.args = args
    def forward(self, x):
        
        residual = x  
        out = self.linear(x)
        out = self.norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = out + residual  
        return out
    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
    def fedavg(self):
        model_dict = self.comm_socket(self.linear.state_dict())
        self.linear.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.norm.state_dict())
        self.norm.load_state_dict(model_dict)
class MLPModule(nn.Module):
   
    def __init__(self, args, dX, num_blocks=1, dropout=0.1):
        super(MLPModule, self).__init__()
        self.blocks = nn.ModuleList([BasicBlock(args ,dX, dropout) for _ in range(num_blocks)])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    def fedavg(self):
        for model in self.blocks: model.fedavg()


class FedCI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_dim = args.input_dim
        self.steps_per_day= args.steps_per_day
        self.tod_embedding_dim = args.tod_embedding_dim
        self.dow_embedding_dim = args.dow_embedding_dim
        self.num_nodes = args.num_nodes
        self.dsp = args.dsp
        self.dsu = args.dsu
        self.in_steps = args.in_steps
        self.out_steps = args.out_steps
        self.args = args
        self.tod_embedding = nn.Embedding(self.steps_per_day, self.tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, self.dow_embedding_dim)
        self.SpatialEmbedding = SpatialEmbedding(args, self.num_nodes, self.dsp, self.dsu)
        self.data_embedding = nn.Linear(self.input_dim, 96)
        self.MLPA = MLPModule(args, self.tod_embedding_dim + self.dow_embedding_dim, 1, dropout=0.1)
        self.MLPB = MLPModule(args, self.dsp + self.dsu, 1, dropout=0.1)
        self.MLPC = MLPModule(args, 96, 1, dropout=0.1)
        self.MLPD = MLPModule(args, self.tod_embedding_dim + self.dow_embedding_dim + self.dsp + self.dsu, 1, dropout=0.1)
        self.MLPE = MLPModule(args, self.tod_embedding_dim + self.dow_embedding_dim + self.dsp + self.dsu + 96, 3, dropout=0.1)
        self.MLPF = MLPModule(args, self.in_steps, 2, dropout=0)
        self.steps_linear = nn.Linear(self.in_steps, self.out_steps)
        self.out_linear = nn.Linear(self.tod_embedding_dim + self.dow_embedding_dim + self.dsp + self.dsu + 96, 1)

        self.data_l = nn.Parameter(torch.randn(96))
        
    def forward(self, x, node_indices):
        batchsize = x.shape[0]

        # print(x.shape)

        tod = x[..., 1].clone()
        dow = x[..., 2].clone()
        x = x[..., : self.input_dim]
        tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )
        dow_emb = self.dow_embedding(
                dow.long()
            )
        # tod_emb = tod_emb + self.tod_embedding_l
        # dow_emb = dow_emb + self.dow_embedding_l
        spa_n_emb, spa_u_emb= self.SpatialEmbedding(node_indices)
        tem_emb = torch.cat((tod_emb, dow_emb), dim=-1)
        tem_emb = self.MLPA(tem_emb)
        spa_emb = torch.cat((spa_n_emb, spa_u_emb), dim=-1)

        # print(step.shape, spa_emb.shape)
        spa_emb = spa_emb.unsqueeze(1)  
        spa_emb = spa_emb.repeat(1, self.in_steps, 1, 1)  
        spa_emb = self.MLPB(spa_emb)
        st_emb = torch.cat((spa_emb, tem_emb), dim=-1)
        st_emb = self.MLPD(st_emb)
        x = self.data_embedding(x)
        x = self.MLPC(x)

        x = x + self.data_l

        step = torch.cat((st_emb, x), dim=-1)
        step = self.MLPE(step)
        step = step.permute(0, 3, 2, 1)
        # print(step.shape)
        step = self.MLPF(step)
        step = self.steps_linear(step)
        step = step.permute(0, 3, 2, 1)
        # print(step.shape)
        out = self.out_linear(step)
        return out
    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
    def fedavg(self):
        model_dict = self.comm_socket(self.tod_embedding.state_dict())
        self.tod_embedding.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.dow_embedding.state_dict())
        self.dow_embedding.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.data_embedding.state_dict())
        self.data_embedding.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.steps_linear.state_dict())
        self.steps_linear.load_state_dict(model_dict)
        model_dict = self.comm_socket(self.out_linear.state_dict())
        self.out_linear.load_state_dict(model_dict)
        self.SpatialEmbedding.fedavg()
        self.MLPA.fedavg()
        self.MLPB.fedavg()
        self.MLPC.fedavg()
        self.MLPD.fedavg()
        self.MLPE.fedavg()
        self.MLPF.fedavg()
