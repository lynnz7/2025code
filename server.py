from lib.server_socket import ServerSocket
import argparse
import time
import copy
import torch
import collections

def FedAvg(w, node_indices_list=None, N=None, dsp=None, dsu=None, device=None):
    target_device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    first_client_weights = w[0]
    contains_embedding = 'B_sp' in first_client_weights and 'B_su' in first_client_weights
    
    if contains_embedding and node_indices_list:
        
        global_B_sp = torch.zeros(N, dsp).to(target_device)  
        global_B_su = torch.zeros(N, dsu).to(target_device)  
        
        for client_idx, client_weights in enumerate(w):
            node_indices = node_indices_list[client_idx]
            client_weights = {k: v.to(target_device) for k, v in client_weights.items()}
            node_indices = node_indices.to(target_device)
            global_B_sp[node_indices] = client_weights['B_sp'][node_indices]
            global_B_su[node_indices] = client_weights['B_su'][node_indices]
        
        return {'B_sp': global_B_sp.to('cpu'), 'B_su': global_B_su.to('cpu')}
    
    
    

    w_avg = copy.deepcopy(w[0])
    
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].to(target_device)
    
    for i in range(len(w)):
        for k in w[i].keys():
            w[i][k] = w[i][k].to(target_device)
    
    for k in w_avg.keys():
        if k not in ['B_sp', 'B_su']:
            for i in range(1, len(w)):
                w_avg[k] += w[i][k]
            w_avg[k] = torch.div(w_avg[k], len(w))
    for k in w_avg.keys():
        w_avg[k] = w_avg[k].to('cpu')
    return w_avg


class Server():
    def __init__(self, n_clients, port, ip, N, dsp, dsu, device):
        self.socket = ServerSocket(n_clients, port, ip)
        
        while True:
            rcvd_msgs = self.socket.recv()
            if rcvd_msgs:
                if isinstance(rcvd_msgs[0], dict) and 'node_indices' in rcvd_msgs[0]:
    
                    node_indices_list = [msg['node_indices'] for msg in rcvd_msgs]
                    weights_list = [msg['weights'] for msg in rcvd_msgs]
                    
                    self.socket.send(FedAvg(weights_list, node_indices_list, N, dsp, dsu, device))
                
                elif type(rcvd_msgs[0]) == collections.OrderedDict or type(rcvd_msgs[0]) == dict:
                    
                    self.socket.send(FedAvg(rcvd_msgs))
                
                
                
                else:
                    
                    self.socket.send(sum(rcvd_msgs))
            
            else:
                print("[SERVER RECVED NONE]")
                self.socket.close()
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', dest='n')
    parser.add_argument('-p', dest='port')
    parser.add_argument('-i', dest='ip')
    parser.add_argument('-N', dest='N', type=int, help="Total number of nodes")
    parser.add_argument('-dsp', dest='dsp', type=int, help="Dimension of known structure embedding")
    parser.add_argument('-dsu', dest='dsu', type=int, help="Dimension of unknown space embedding")
    parser.add_argument('--device', dest='device', default='cuda:0', help="Specify the device to use (e.g., 'cuda:0', 'cpu')")

    args = parser.parse_args()

    
    server = Server(
        n_clients=int(args.n), 
        port=int(args.port), 
        ip=args.ip,
        N=args.N,
        dsp=args.dsp,
        dsu=args.dsu,
        device=args.device
    )