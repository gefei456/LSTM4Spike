import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pickle
import sklearn.decomposition as skd
# import torchrec.sparse.jagged_tensor as tsj
import random

def a_norm(Q, K):
    m = torch.matmul(Q, K.transpose(2,1).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())
    
    return torch.softmax(m , -1)


def attention(Q, K, V):
    #Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K) #(batch_size, dim_attn, seq_length)
    
    return  torch.matmul(a,  V) #(batch_size, seq_length, seq_length)

class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)
        self.key = Key(dim_val, dim_attn)
        self.query = Query(dim_val, dim_attn)
    
    def forward(self, x, kv = None):
        if(kv is None):
            #Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))
        
        #Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))
    
class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))
        
        self.heads = nn.ModuleList(self.heads)
        
        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias = False)
                      
        
    def forward(self, x, kv = None):
        a = []
        for h in self.heads:
            a.append(h(x, kv = kv))
            
        a = torch.stack(a, dim = -1) #combine heads
        a = a.flatten(start_dim = 2) #flatten all head outputs
        
        x = self.fc(a)
        
        return x
    
class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val
        
        self.fc1 = nn.Linear(dim_input, dim_val, bias = False)
        #self.fc2 = nn.Linear(5, dim_val)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return x

class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):
        x = self.fc1(x)
        #x = self.fc2(x)
        
        return x

class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn
        
        self.fc1 = nn.Linear(dim_input, dim_attn, bias = False)
        #self.fc2 = nn.Linear(5, dim_attn)
    
    def forward(self, x):
        
        x = self.fc1(x)
        #print(x.shape)
        #x = self.fc2(x)
        
        return x

# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        return x 
    
def get_data(batch_size, input_sequence_length, output_sequence_length):
    i = input_sequence_length + output_sequence_length
    
    t = torch.zeros(batch_size,1).uniform_(0,100 - i).int()
    b = torch.arange(-30, -30 + i).unsqueeze(0).repeat(batch_size,1) + t
    
    s = torch.sigmoid(b.float())
    return s[:, :input_sequence_length].unsqueeze(-1), s[:,-output_sequence_length:]

def PCA_decomposition(X, n_component):
    pca = skd.PCA(n_components = n_component)
    # data_spike_merge = np.array(data_spike_merge)
    # data_spike_merge = np.reshape(X, (-1, np.prod(X.shape[-2::])))
    pca.fit(X)
    print('total explained variance ratio: ', np.cumsum(pca.explained_variance_ratio_)[-1])
    X = pca.transform(X)
    
    return X, pca
   

def spike_trace(file_path, session_names, bin_size, spike_lag, isAlign, PCA_ncomp):
    data_spike_merge, data_traj_merge, data_trial_merge = [], [], []
    for session_name in session_names:
        with open(f"{file_path}{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f:
            data = pickle.load(f)
            # data_spike_merge.append(data['spike_list'])
            data_spike_merge += data['spike_list']
            # data_traj_merge.append(data['truth_traj'])
            data_traj_merge += data['truth_traj']
            data_trial_merge += data['label_list']
    
    cat_spike = np.array([]).reshape((0, *data_spike_merge[0].shape[1:]))
    cat_trace = np.array([]).reshape((0, *data_traj_merge[0].shape[1:]))
    for x in range(len(data_spike_merge)):
        cat_spike = np.concatenate((cat_spike, data_spike_merge[x]), axis=0)
        cat_trace = np.concatenate((cat_trace, data_traj_merge[x]), axis=0)
    
    # PCA to reduce the feature of data
    cat_spike, _ = PCA_decomposition(cat_spike, PCA_ncomp)
     
    # return torch.from_numpy(cat_spike), torch.from_numpy(cat_trace)
    # return torch.tensor(cat_spike, dtype=torch.float32).reshape((-1, int(cat_spike.shape[1]/enc_seq_len), enc_seq_len)), torch.tensor(cat_trace, dtype = torch.float32)
    return torch.tensor(cat_spike, dtype=torch.float32).unsqueeze(-1), torch.tensor(cat_trace, dtype = torch.float32)
        
    
def load_data(file_path, session_names, bin_size, spike_lag, isAlign, test_ratio):
    data_spike_merge, data_traj_merge, data_trial_merge = [], [], []
    for session_name in session_names:
        with open(f"{file_path}{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f:
            data = pickle.load(f)
            # data_spike_merge.append(data['spike_list'])
            data_spike_merge += data['spike_list']
            # data_traj_merge.append(data['truth_traj'])
            data_traj_merge += data['truth_traj']
            data_trial_merge += data['label_list']
    
    train_data_spike = data_spike_merge[0:-math.ceil(len(data_spike_merge) * test_ratio)]
    train_data_trace = data_traj_merge[0:-math.ceil(len(data_traj_merge) * test_ratio)]
    train_data_trial = data_trial_merge[0:-math.ceil(len(data_trial_merge) * test_ratio)]
    train_data_trial = [int(x) for x in train_data_trial]
    test_data_spike = data_spike_merge[-math.ceil(len(data_spike_merge) * test_ratio)::]
    test_data_trace = data_traj_merge[-math.ceil(len(data_traj_merge) * test_ratio)::]
    test_data_trial = data_trial_merge[-math.ceil(len(data_trial_merge) * test_ratio)::]
    test_data_trial = [int(x) for x in test_data_trial]
    
    return train_data_spike, train_data_trace, train_data_trial, test_data_spike, test_data_trace, test_data_trial 

def load_data_pos(file_path, session_names, bin_size, spike_lag, isAlign, test_ratio):
    data_spike_merge, data_traj_merge, data_trial_merge, data_tgpos_merge = [], [], []
    for session_name in session_names:
        with open(f"{file_path}{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f:
            data = pickle.load(f)
            # data_spike_merge.append(data['spike_list'])
            data_spike_merge += data['spike_list']
            # data_traj_merge.append(data['truth_traj'])
            data_traj_merge += data['truth_traj']
            data_trial_merge += data['label_list']
            data_tgpos_merge += data['tgpos_list']
    
    train_data_spike = data_spike_merge[0:-math.ceil(len(data_spike_merge) * test_ratio)]
    train_data_trace = data_traj_merge[0:-math.ceil(len(data_traj_merge) * test_ratio)]
    train_data_trial = data_trial_merge[0:-math.ceil(len(data_trial_merge) * test_ratio)]
    train_data_trial = [int(x) for x in train_data_trial]
    train_data_tgpos = data_tgpos_merge[0:-math.ceil(len(data_tgpos_merge) * test_ratio)]
    test_data_spike = data_spike_merge[-math.ceil(len(data_spike_merge) * test_ratio)::]
    test_data_trace = data_traj_merge[-math.ceil(len(data_traj_merge) * test_ratio)::]
    test_data_trial = data_trial_merge[-math.ceil(len(data_trial_merge) * test_ratio)::]
    test_data_trial = [int(x) for x in test_data_trial]
    test_data_tgpos = data_tgpos_merge[-math.ceil(len(data_tgpos_merge) * test_ratio)::]
    
    return train_data_spike, train_data_trace, train_data_trial, train_data_tgpos, test_data_spike, test_data_trace, test_data_trial, test_data_tgpos 


def ragged_spike_trace(file_path, session_names, bin_size, spike_lag, isAlign, PCA_ncomp, trace_dim, *kwargs):
    data_spike_merge, data_traj_merge, data_trial_merge = [], [], []
    for session_name in session_names:
        with open(f"{file_path}{session_name}_{bin_size}_{isAlign}_{spike_lag}_ReNN.pkl", "rb") as f:
            data = pickle.load(f)
            # data_spike_merge.append(data['spike_list'])
            data_spike_merge += data['spike_list']
            # data_traj_merge.append(data['truth_traj'])
            data_traj_merge += data['truth_traj']
            data_trial_merge += data['label_list']
    
    cat_spike = np.array([]).reshape((0, *data_spike_merge[0].shape[1:]))
    cat_trace = np.array([]).reshape((0, *data_traj_merge[0].shape[1:]))
    cat_index = []
    
    for x in range(len(data_spike_merge)):
        cat_spike = np.concatenate((cat_spike, data_spike_merge[x]), axis=0)
        cat_trace = np.concatenate((cat_trace, data_traj_merge[x]), axis=0)
        cat_index.append(data_spike_merge[x].shape[0])
    
    cat_index = np.hstack(([0], np.cumsum(cat_index)))
        
    cat_spike = np.reshape(cat_spike, (cat_spike.shape[0], -1))
    # PCA to reduce the feature of data
    if kwargs == (None or True):
        cat_spike, pca = PCA_decomposition(cat_spike, PCA_ncomp)
        
    
    if trace_dim == 'pos':
        cat_trace = cat_trace[:, 0:2] 
    elif trace_dim == 'vel':
        cat_trace = cat_trace[:, 2::]
     
    
    nested_spike, nested_trace = [], []
    for x in range(len(cat_index) - 1):
        nested_spike.append(torch.tensor(cat_spike[cat_index[x]:cat_index[x+1], :], requires_grad=False))
        nested_trace.append(torch.tensor(cat_trace[cat_index[x]:cat_index[x+1], :], requires_grad=False))
        
    nested_spike = torch.nested.nested_tensor(nested_spike, dtype=torch.float32)
    nested_trace = torch.nested.nested_tensor(nested_trace, dtype=torch.float32)
    
    # jt = tsj.JaggedTensor(values = cat_spike, lengths = cat_index)
    # cat_spike = jt.to_dense()
    # jt = tsj.JaggedTensor(values = cat_trace, lengths = cat_index)
    # cat_trace = jt.to_dense()
    
    # return cat_spike, cat_trace
    return nested_spike, nested_trace, data_trial_merge

def concat_spike_trace(data_spike, data_trace, trace_dim, PCA_type, pca_param):
    # fit PCA model to test data
    
    cat_spike = np.array([]).reshape((0, *data_spike[0].shape[1:]))
    cat_trace = np.array([]).reshape((0, *data_trace[0].shape[1:]))
    cat_index = []
    
    for x in range(len(data_spike)):
        cat_spike = np.concatenate((cat_spike, data_spike[x]), axis=0)
        cat_trace = np.concatenate((cat_trace, data_trace[x]), axis=0)
        cat_index.append(data_spike[x].shape[0])
    
    cat_index = np.hstack(([0], np.cumsum(cat_index)))
        
    # PCA to reduce the feature of data
    # if kwargs is (None or True):
    #     cat_spike, pca = PCA_decomposition(cat_spike, PCA_ncomp)
    # else:
    
    # cat_spike = np.reshape(cat_spike, (-1, np.prod(cat_spike.shape[-2::])))
    cat_spike = np.reshape(cat_spike, (cat_spike.shape[0], -1))
    if PCA_type == 'fit':
        cat_spike, pca = PCA_decomposition(cat_spike, pca_param)
    elif PCA_type == 'transform':
        # for pca in args:
        # pca = *args[0]
        cat_spike = pca_param.transform(cat_spike)
    
    if trace_dim == 'pos':
        cat_trace = cat_trace[:, 0:2] 
    elif trace_dim == 'vel':
        cat_trace = cat_trace[:, 2::]
    
    if PCA_type == 'fit':    
        return cat_spike, cat_trace, cat_index, pca
    elif PCA_type == 'transform':
        return cat_spike, cat_trace, cat_index
                  
def concat_target_pos(data_tgpos, cat_index):
    cat_tgpos = np.array([]).reshape((0, *data_tgpos[0].shape[1:]))
    for x in range(len(data_tgpos)):
        cat_tgpos = np.concatenate((cat_tgpos, np.tile(data_tgpos[x], (cat_index[x + 1] - cat_index[x], 1))), axis=0)
        
    return cat_tgpos    
    
def ragged_spike_trace_split(data_spike, data_trace, trace_dim, PCA_type, pca_param):
      
    if PCA_type == 'fit':    
        cat_spike, cat_trace, cat_index, pca = concat_spike_trace(data_spike, data_trace, trace_dim, PCA_type, pca_param)
    elif PCA_type == 'transform':  
        cat_spike, cat_trace, cat_index = concat_spike_trace(data_spike, data_trace, trace_dim, PCA_type, pca_param)
             
    nested_spike, nested_trace = [], []
    for x in range(len(cat_index) - 1):
        nested_spike.append(torch.tensor(cat_spike[cat_index[x]:cat_index[x+1], :], requires_grad=False))
        nested_trace.append(torch.tensor(cat_trace[cat_index[x]:cat_index[x+1], :], requires_grad=False))
        
    nested_spike = torch.nested.nested_tensor(nested_spike, dtype=torch.float32)
    nested_trace = torch.nested.nested_tensor(nested_trace, dtype=torch.float32)
    
    # jt = tsj.JaggedTensor(values = cat_spike, lengths = cat_index)
    # cat_spike = jt.to_dense()
    # jt = tsj.JaggedTensor(values = cat_trace, lengths = cat_index)
    # cat_trace = jt.to_dense()
    
    # return cat_spike, cat_trace
    if PCA_type == 'fit':
        return nested_spike, nested_trace, pca
    elif PCA_type == 'transform':
        return nested_spike, nested_trace
    
def ragged_spike_trace_single(data_spike, data_trace, trace_dim, PCA_type, pca_param, n_bins = 1):
    if PCA_type == 'fit':    
        cat_spike, cat_trace, cat_index, pca = concat_spike_trace(data_spike, data_trace, trace_dim, PCA_type, pca_param)
    elif PCA_type == 'transform':  
        cat_spike, cat_trace, cat_index = concat_spike_trace(data_spike, data_trace, trace_dim, PCA_type, pca_param)
             
    nested_spike, nested_trace = [], []
    for x in range(len(cat_index) - 1):
    # for x in range(cat_spike.shape[0]):
        for ind in range(cat_index[x], cat_index[x+1]):
            if (ind - cat_index[x]) < n_bins:
                nested_spike.append(torch.tensor(cat_spike[cat_index[x]:(ind+1), :], requires_grad=False).unsqueeze(0))
                nested_trace.append(torch.tensor(cat_trace[cat_index[x]:(ind+1), :], requires_grad=False).unsqueeze(0))
            else:    
                nested_spike.append(torch.tensor(cat_spike[(ind+1-n_bins):(ind+1), :], requires_grad=False).unsqueeze(0))
                nested_trace.append(torch.tensor(cat_trace[(ind+1-n_bins):(ind+1), :], requires_grad=False).unsqueeze(0))
    
    nested_spike = torch.nested.nested_tensor(nested_spike, dtype=torch.float32)
    nested_trace = torch.nested.nested_tensor(nested_trace, dtype=torch.float32)
    
    if PCA_type == 'fit':
        return nested_spike, nested_trace, cat_index, pca
    elif PCA_type == 'transform':
        return nested_spike, nested_trace, cat_index
    
def ragged_spike_trace_simulate(data_spike, data_trace, trace_dim, pca_param):
    data_spike = pca_param.transform(data_spike)
    if trace_dim == 'pos':
        data_trace = data_trace[:, 0:2] 
    elif trace_dim == 'vel':
        data_trace = data_trace[:, 2::]
    
    # nested_spike = torch.nested.nested_tensor([torch.tensor(data_spike)], dtype=torch.float32)
    # nested_trace = torch.nested.nested_tensor([torch.tensor(data_trace)], dtype=torch.float32)
    nested_spike = torch.tensor(data_spike, dtype = torch.float32)
    nested_trace = torch.tensor(data_trace, dtype = torch.float32)
    
    return nested_spike, nested_trace
     
def evaluate(cat_spike, cat_trace, test_size, transformer):
    transformer.eval()
    losses_val = 0
    if test_size > len(cat_spike):
        test_size = len(cat_spike)
    with torch.no_grad():
        for testind in range(-test_size, -1):
            if len(cat_spike[testind].shape) < 3:
                x, y = cat_spike[testind].unsqueeze(0), cat_trace[testind].unsqueeze(0)
            else:
                x, y = cat_spike[testind], cat_trace[testind]
            pred = transformer(x)
            loss = torch.mean((pred - y) ** 2)
            losses_val += loss
        
    losses_val /= test_size
    print(f"Avg Test Loss: {losses_val:>8f} \n")
    
    return losses_val           
            
def evaluate_random(cat_spike, cat_trace, batch_size, transformer):
    transformer.eval()
    losses_val = 0
    with torch.no_grad():
        # for testind in range(-test_size, -1):
        testind = random.sample([x for x in range(cat_spike.size(0))], batch_size)
        x, y = cat_spike[testind].unsqueeze(1), cat_trace[testind].unsqueeze(1)
        pred = transformer(x)
        loss = torch.mean((pred - y) ** 2)
        losses_val += loss
        
    losses_val /= batch_size
    print(f"Avg Test Loss: {losses_val:>8f} \n")
    
    return losses_val           
            

def R2_score(targts, preds):
    SS_res = np.sum((targts - preds) ** 2)
    # print(SS_res)
    SS_tot = np.sum((targts - np.mean(targts, axis = 0)) ** 2)
    # print(SS_tot)
    r2 = 1 - SS_res / SS_tot
    # print('R^2: ', r2)
    return r2

def final_dist(targts, preds):
    fd = np.sqrt(np.max([np.sum(targts[-1, :] ** 2) + np.sum(preds[-1, :] ** 2) \
        - 2 * targts[-1, :].dot(preds[-1, :]), 0]))
    
    return fd

def refit_velocity(nout_vel, curr_pos, targ_pos, alpha):
    if len(nout_vel.shape) < 2:
        nout_vel = np.expand_dims(nout_vel, axis = 0)
    vel_norm = np.linalg.norm(nout_vel, axis = 1)
    correct_vec = targ_pos - curr_pos
    rot_vel = np.expand_dims(vel_norm, axis = 1) * correct_vec / np.expand_dims(np.linalg.norm(correct_vec, axis = 1), axis = 1)
    angle = np.arccos((rot_vel[:,0]*nout_vel[:,0] + rot_vel[:,1]*nout_vel[:,1]) / (np.linalg.norm(rot_vel, axis = 1) * np.linalg.norm(nout_vel, axis = 1)))
    
    rela_phase = nout_vel[:, 0] * rot_vel[:, 1] - nout_vel[:, 1] * rot_vel[:, 0]
    new_vel = np.array([]).reshape(0, 2)
    for x in range(rela_phase.shape[0]):
        if rela_phase[x] < 0:
            angle[x] = -angle[x]
    
        angle[x] *= alpha
    
        trans_mat = [[math.cos(angle[x]), math.sin(angle[x])],
                    [-math.sin(angle[x]), math.cos(angle[x])]]
        
        new_vel = np.concatenate((new_vel, np.expand_dims(nout_vel[x] @ trans_mat, axis = 0)), axis = 0)
        
    return new_vel

    
    