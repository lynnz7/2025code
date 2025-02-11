import copy, torch
import torch.nn as nn
import numpy as np
import pickle
def split(ary, indices_or_sections):
    import numpy.core.numeric as _nx
    Ntotal = len(ary)
    Nsections = int(indices_or_sections)
    Neach_section, extras = divmod(Ntotal, Nsections)
    section_sizes = ([0] +
                        extras * [Neach_section+1] +
                        (Nsections-extras) * [Neach_section])
    div_points = _nx.array(section_sizes, dtype=_nx.intp).cumsum()

    sub_arys = []
    for i in range(Nsections):
        st = div_points[i]
        end = div_points[i + 1]
        sub_arys.append(ary[st:end])

    return sub_arys


def avg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

class SpatialEmbedding(nn.Module):
    def __init__(self, args, N, dsp, dsu):
        super(SpatialEmbedding, self).__init__()
    
        self.args = args
        self.N = N
        
        self.B_sp = nn.Parameter(torch.randn(N, dsp))  
        self.B_su = nn.Parameter(torch.randn(N, dsu))  
        self.node_indices = None
    
    

    def forward(self, node_indices):
        self.node_indices = node_indices
        
        mask = torch.ones(self.N, dtype=bool)
        mask[node_indices] = False
        Esp = self.B_sp
        E_i_sp = Esp[node_indices]  

        E_i_su = self.B_su[node_indices]

        return E_i_sp, E_i_su
    def comm_socket(self, msg, device=None):
        self.args.socket.send(msg)
        if device:
            return self.args.socket.recv().to(device)
        else:
            return self.args.socket.recv()
    def fedavg(self):  
        
        msg = {
            'node_indices': self.node_indices, 
            'weights': {                        
                'B_sp': self.B_sp.data.clone(),
                'B_su': self.B_su.data.clone()
            }
        }

        updated_embeddings = self.comm_socket(msg)

        if 'B_sp' in updated_embeddings and 'B_su' in updated_embeddings:
            self.B_sp.data = updated_embeddings['B_sp'].to(self.args.device)
            self.B_su.data = updated_embeddings['B_su'].to(self.args.device)



def smooth_fill_zeros(matrix):
    steps, nodes = matrix.shape
    for node in range(nodes):
        
        data = matrix[:, node]

        indices = np.arange(steps)
        non_zero_mask = data != 0
        
        if not np.any(non_zero_mask):
            continue
        
        filled_data = np.interp(indices, indices[non_zero_mask], data[non_zero_mask])

        matrix[:, node] = filled_data

    return matrix