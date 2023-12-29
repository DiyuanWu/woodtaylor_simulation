import torch
import torch.nn as nn
import torch.optim as optim
import math

# For reproduce the results
torch.manual_seed(1234)

import numpy as np


from data_gen import randomly_zero, sparse_linear_data

from algorithms import topk





d = 128

k_star = 16

k = 64 # The sparsity of weights during training

    
num_steps = 7

def target_func(w:torch.Tensor, w_star: torch.Tensor):

    return torch.pow(torch.norm( w - w_star,2 ) , 4)


# generate a k-sparse signal

w_star = torch.randn( [d,1] ) 

w_star = randomly_zero(w_star, d - k_star) 


# initialization
w = torch.randn( [d,1] ) 

for step in range(num_steps):

    y = target_func(w, w_star)
    
    gradient =  4*torch.pow(torch.norm( w - w_star,2 ) , 2)*(w - w_star)
    hessian = 8*torch.matmul((w - w_star).t(), (w - w_star)) + 4* torch.pow(torch.norm( w - w_star,2 ) , 2)*torch.eye(d)

    w = topk( w- hessian.inverse() @ gradient, k) 

    print(target_func(w,w_star))

    print(torch.norm(w - w_star,2 ))
