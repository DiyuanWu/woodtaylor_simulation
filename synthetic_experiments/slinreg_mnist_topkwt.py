import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import random


# For reproduce the results
torch.manual_seed(1234)
random.seed(1234)

import numpy as np


from data_gen import sparse_linear_data, sparse_linear_data_dataset

from algorithms import topk


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.linear(x)
    

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])

# Download and load the training dataset
train_dataset = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)

# Create a DataLoader to iterate over the training dataset
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)


num_dataset = train_dataset.data.shape[0]

mnist_shape = train_dataset.data[0].shape

d = train_dataset.data[0].view(-1).shape[0]

n = 2*d


num_expr = 20

num_steps = 750



loss_steps = torch.zeros([num_expr, num_steps  ])

dist_steps = torch.zeros([num_expr, num_steps  ])

kiht_loss_steps = torch.zeros([num_expr, num_steps  ])

kiht_dist_steps = torch.zeros([num_expr, num_steps  ])

origin_signals = torch.zeros( [ num_expr, d ] )

recovered_signals = torch.zeros( [ num_expr, d ] )

kiht_recovered_signals = torch.zeros( [ num_expr, d ] )


for expr in range(num_expr):

    X, Y, w_star, k_star = sparse_linear_data_dataset(n, train_dataset)

    origin_signals[ expr,: ] = w_star.t()


    # The sparsity for each step 
    k = 2*k_star

    print((k_star,k))

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

    recovered_signals[expr, :] = model.linear.weight.data.view(w_star.shape).t()


    # The K-IHT method
        
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
            
            param.data =  topk( param.data - max_lr*gradient , k) 



        # compute the distance to the optimal weight
        kiht_dist_steps[expr, step] = torch.norm(model_kiht.linear.weight.data.view(w_star.shape) - w_star, 2)

        # Zero the gradients for the next iteration
        model_kiht.linear.weight.grad.data.zero_()


    kiht_recovered_signals[expr, :] = model_kiht.linear.weight.data.view(w_star.shape).t()

file_name_loss = "./topkwt_slinearreg_mnist_loss_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_dist = "./topkwt_slinearreg_mnist_dist_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_loss_kiht = "./kiht_slinearreg_mnist_loss_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_dist_kiht = "./kiht_slinearreg_mnist_dist_{}steps_{}exprs.npy".format( num_steps, num_expr)

file_name_origin_signals = "./slinearreg_mnist_orisigals.npy"

file_name_recov_signals = "./topkwt_slinearreg_mnist_recsigals.npy"

file_name_recov_signals_kiht = "./kiht_slinearreg_mnist_recsigals.npy"


with open(file_name_loss,'wb') as f:

    np.save(f, loss_steps.numpy())


with open(file_name_dist,'wb') as f:

    np.save(f, dist_steps.numpy())

with open(file_name_loss_kiht,'wb') as f:

    np.save(f, kiht_loss_steps.numpy())


with open(file_name_dist_kiht,'wb') as f:

    np.save(f, kiht_dist_steps.numpy())


with open( file_name_origin_signals, 'wb') as f:

    np.save(f, origin_signals.numpy() )

with open( file_name_recov_signals, 'wb') as f:

    np.save(f, recovered_signals.numpy() )

with open( file_name_recov_signals_kiht, 'wb') as f:

    np.save(f, kiht_recovered_signals.numpy() )

