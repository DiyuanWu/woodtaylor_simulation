import torch 

from algorithms import topk, OBC

from data_gen import sparse_linear_data

import torch
import torch.nn as nn
import torch.optim as optim

from matplotlib import pyplot as plt

# For reproduce the results
#torch.manual_seed(1234)

import numpy as np


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.linear(x)


num_expr = 1

n = 256

d = 128

k_star = 16

k = 64 # The sparsity of weights during training

    
num_steps = 100

X, Y, w_star = sparse_linear_data(n, d, k_star)

obc_loss_steps = torch.zeros([num_expr, num_steps  ])

obc_dist_steps = torch.zeros([num_expr, num_steps  ])
   
obc_maskdist_steps = torch.zeros([num_expr, num_steps  ])


model_obc= LinearRegressionModel(d, 1)

criterion = nn.MSELoss()

# Hessian is analytically computed
hessian = torch.matmul(X.t(), X)

h_inv = hessian.inverse()

print(X.shape, Y.shape)


for expr in range(num_expr):
# The topk_WoodTaylor method
    




    for step in range(num_steps):
    # Forward pass
        predictions = model_obc.forward(X)

        # Compute the loss
        loss = criterion(predictions,Y)

        obc_loss_steps[expr,step] = loss.item()

        # Backward pass
        loss.backward()

        #print(loss.item())

        # Estimate the Hessian of each parameters of the model
        # In the case of linear regression, the Hessian is fixed and can be analytcally computed


        # Update parameters using the WoodTaylor optimizer, in this special case of Linear regression 


        for param in model_obc.parameters():

            gradient = param.grad.data
            
            data_new = param.data - gradient @ hessian.inverse()

            data_new , mask_obc = OBC( param.data, h_inv, d, k )


            _, mask_topk = topk(data_new, k)

            param.data = data_new.view(param.data.shape)

            # param.data = torch.mul(mask_obc.view(param.shape), data_new)

            obc_maskdist_steps[expr, step] = torch.sum( torch.abs( mask_obc - mask_topk.view(mask_obc.shape)  ) )


        # compute the distance to the optimal weight
        obc_dist_steps[expr, step] = torch.norm(model_obc.linear.weight.data.view(w_star.shape) - w_star, 2)

        # Zero the gradients for the next iteration
        model_obc.linear.weight.grad.data.zero_()
        

plot_t = np.linspace(0,num_steps,num_steps)

plt.plot(plot_t, obc_loss_steps[0,:])

plt.savefig("./test.png")