import torch
import torch.nn as nn
import torch.optim as optim
import math

# For reproduce the results
torch.manual_seed(1234)

import numpy as np
from matplotlib import pyplot as plt



from data_gen import randomly_zero, sparse_linear_data

from algorithms import topk


d = 128

k_star = 16

k = 64 # The sparsity of weights during training

num_steps = 20    


def target_func(w:torch.Tensor, w_star: torch.Tensor):

    return 3*torch.pow(torch.norm( w - w_star,2 ) , 4)

# generate a k-sparse signal

w_star = (1/math.sqrt(d))*torch.randn( [d,1] ) 

w_star = randomly_zero(w_star, d - k_star) 




# initialization
w_topkwt = (1/math.sqrt(d))*torch.randn( [d,1] ) 

w_kiht = w_topkwt



loss_topkwt = torch.zeros([num_steps,1])

loss_kiht = torch.zeros([num_steps,1])

for step in range(num_steps):

    loss_topkwt[step] = target_func(w_topkwt, w_star)

    loss_kiht[step] = target_func(w_kiht, w_star)

    print( (target_func(w_topkwt, w_star),target_func(w_kiht, w_star) ))
    
    # The update for topkwt
    gradient =  4*torch.pow(torch.norm( w_topkwt - w_star,2 ) , 2)*(w_topkwt - w_star)
    hessian = 8*torch.matmul((w_topkwt - w_star).t(), (w_topkwt - w_star)) + 4* torch.pow(torch.norm( w_topkwt - w_star,2 ) , 2)*torch.eye(d)

    w_topkwt, _ = topk( w_topkwt- hessian.inverse() @ gradient, k)



    # The update for kiht

    gradient =  4*torch.pow(torch.norm( w_kiht - w_star,2 ) , 2)*(w_kiht - w_star)
    hessian = 8*torch.matmul((w_kiht - w_star).t(), (w_kiht - w_star)) + 4* torch.pow(torch.norm( w_kiht - w_star,2 ) , 2)*torch.eye(d)

    hessian_sym = 0.5*(hessian + hessian.t())
        
    eigvals =  torch.real(torch.linalg.eigvals(hessian_sym)) 

    max_lr = 1/torch.max ( eigvals, dim = 0).values.item()

    w_kiht, _ = topk( w_kiht - max_lr*gradient.view(w_kiht.shape), k )


   



fig, ax = plt.subplots( )

plot_t = np.linspace(0, num_steps, num_steps)

ax.plot(plot_t, loss_topkwt, label='topk-WT', color = 'blue')

ax.plot(plot_t, loss_kiht, label='k-IHT', color = 'red')

ax.set_xlabel('# of steps')

ax.set_ylabel('Loss')

ax.legend()

figname = 'opt.png'

plt.savefig(figname)









