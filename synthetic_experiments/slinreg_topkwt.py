import torch
import torch.nn as nn
import torch.optim as optim


# For reproduce the results
torch.manual_seed(1234)

import numpy as np


from data_gen import sparse_linear_data

from algorithms import topk, OBC


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.linear(x)


num_expr = 20

n = 256

d = 128

k_star = 16

k = 64 # The sparsity of weights during training

    
num_steps = 750





loss_steps = torch.zeros([num_expr, num_steps  ])

dist_steps = torch.zeros([num_expr, num_steps  ])

kiht_loss_steps = torch.zeros([num_expr, num_steps  ])

kiht_dist_steps = torch.zeros([num_expr, num_steps  ])

obc_loss_steps = torch.zeros([num_expr, num_steps  ])

obc_dist_steps = torch.zeros([num_expr, num_steps  ])
   
obc_maskdist_steps = torch.zeros([num_expr, num_steps  ])

for expr in range(num_expr):

    X, Y, w_star = sparse_linear_data(n, d, k_star)

    
    # The Topk-WoodTaylor Methods --------------------------------------------------------------------------------
    
    model= LinearRegressionModel(d, 1)

    

    criterion = nn.MSELoss()

    # Hessian is analytically computed
    hessian = torch.matmul(X.t(), X)

    print(X.shape, Y.shape)

    # The topk_WoodTaylor method
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


        # Update parameters using the WoodTaylor optimizer, in this special case of Linear regression 
        for param in model.parameters():

            gradient = param.grad.data
            
            param.data, _ =  topk( param.data - gradient @ hessian.inverse() , k) 



        # compute the distance to the optimal weight
        dist_steps[expr, step] = torch.norm(model.linear.weight.data.view(w_star.shape) - w_star, 2)

        # Zero the gradients for the next iteration
        model.linear.weight.grad.data.zero_()


    # The K-IHT method--------------------------------------------------------------------------------------------------------------------------
        
    model_kiht =  LinearRegressionModel(d, 1)

    # symmetrize the matrix due to numerical error
    hessian_sym = 0.5*(hessian + hessian.t())
        
    eigvals =  torch.real(torch.linalg.eigvals(hessian_sym)) 

    max_lr = 1/torch.max( eigvals, dim = 0).values.item()



    for step in range(num_steps):
        # Forward pass
        predictions = model_kiht.forward(X)

        # Compute the loss
        loss = criterion(predictions,Y)

        kiht_loss_steps[expr, step] = loss.item()

        # Backward pass
        loss.backward()

        # Estimate the Hessian of each parameters of the model
        # In the case of linear regression, the Hessian is fixed and can be analytcally computed


        # Update parameters using the WoodTaylor optimizer, in this special case of Linear regression 
        for param in model_kiht.parameters():

            gradient = param.grad.data
            
            param.data, _ =  topk( param.data - max_lr*gradient , k) 



        # compute the distance to the optimal weight
        kiht_dist_steps[expr, step] = torch.norm(model_kiht.linear.weight.data.view(w_star.shape) - w_star, 2)

        # Zero the gradients for the next iteration
        model_kiht.linear.weight.grad.data.zero_()


    # The OBC_WoodTaylor methods ---------------------------------------------------------------------------------------------------------------

    model_obc= LinearRegressionModel(d, 1)

    criterion = nn.MSELoss()

    # Hessian is analytically computed
    hessian = torch.matmul(X.t(), X)

    h_inv = hessian.inverse()

    print(X.shape, Y.shape)

    # The topk_WoodTaylor method
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


        # Update parameters using the WoodTaylor optimizer, in this special case of Linear regression 
        for param in model.parameters():

            gradient = param.grad.data
            
            data_new = param.data - gradient @ hessian.inverse()

            param.data, mask_obc = OBC( data_new, h_inv, d, k )

            _, mask_topk = topk(data_new, k)

        







        # compute the distance to the optimal weight
        dist_steps[expr, step] = torch.norm(model.linear.weight.data.view(w_star.shape) - w_star, 2)

        # Zero the gradients for the next iteration
        model.linear.weight.grad.data.zero_()
    




    

    

file_name_loss = "./topkwt_slinearreg_loss_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

file_name_dist = "./topkwt_slinearreg_dist_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

file_name_loss_kiht = "./kiht_slinearreg_loss_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

file_name_dist_kiht = "./kiht_slinearreg_dist_{}steps_{}exprs_n{}_d{}_k{}_kstar{}.npy".format( num_steps, num_expr, n,d,k,k_star)

with open(file_name_loss,'wb') as f:

    np.save(f, loss_steps.numpy())


with open(file_name_dist,'wb') as f:

    np.save(f, dist_steps.numpy())

with open(file_name_loss_kiht,'wb') as f:

    np.save(f, kiht_loss_steps.numpy())


with open(file_name_dist_kiht,'wb') as f:

    np.save(f, kiht_dist_steps.numpy())





