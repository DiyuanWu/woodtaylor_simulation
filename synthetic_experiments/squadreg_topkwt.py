import torch
import torch.nn as nn
import torch.optim as optim


# For reproduce the results
# torch.manual_seed(1234)

import numpy as np


from data_gen import sparse_linear_data

from algorithms import topk


class QuadRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(QuadRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return torch.pow(self.linear(x), 2)


num_expr = 20

n = 256

d = 128

k_star = 16

k = 64 # The sparsity of weights during training

    
num_steps = 750



loss_steps = torch.zeros([num_expr, num_steps  ])

dist_steps = torch.zeros([num_expr, num_steps  ])

for expr in range(num_expr):

    X, Y, w_star = sparse_linear_data(n, d, k_star)

    model = QuadRegressionModel(d, 1)

    criterion = nn.MSELoss()

    print(X.shape, Y.shape)

    for step in range(num_steps):
        # Forward pass
        predictions = model.forward(X)

        # Compute the loss
        loss = criterion(predictions,Y)

        loss_steps[expr, step] = loss.item()

        # Backward pass
        loss.backward()

        # Estimate the Hessian of each parameters of the model
        # In the case of linear regression, the Hessian is fixed and can be analytcally computed
        
        # Hessian of quadratic regression model is: 

        hessian = 

        # Update parameters using the WoodTaylor optimizer, in this special case of Linear regression 
        for param in model.parameters():

            gradient = param.grad.data
            
            param.data =  topk( param.data - gradient @ hessian.inverse() , k) 



        # compute the distance to the optimal weight
        dist_steps[expr, step] = torch.norm(model.linear.weight.data.view(w_star.shape) - w_star, 2)

        # Zero the gradients for the next iteration
        model.linear.weight.grad.data.zero_()

file_name_loss = "./squadreg_loss_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

file_name_dist = "./squadreg_dist_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

with open(file_name_loss,'wb') as f:

    np.save(f, loss_steps.numpy())


with open(file_name_dist,'wb') as f:

    np.save(f, dist_steps.numpy())







