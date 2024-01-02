import torch
import numpy as np
import math

import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def randomly_zero(vector, k):
    if k >= len(vector):
        raise ValueError("k should be less than the length of the vector.")

    # Generate a mask with k zeros
    mask_indices = torch.randperm(len(vector))[:k]
    mask = torch.zeros_like(vector)
    mask[mask_indices] = 1

    # Apply the mask to zero out selected elements
    result_vector = vector * (1 - mask)


    return result_vector


def sparse_linear_data(n, d, k_star):

    """ Generate data from y = w_star^T x, w_star is a k_star - sparse vector """

    w_star = torch.randn( [d,1] ) 

    w_star = randomly_zero(w_star, d - k_star) 

    X = 1/math.sqrt(n)*torch.randn( [n,d] )

    Y = torch.matmul( X, w_star)

    return X, Y, w_star


def sparse_quadratic_model(n,d,k_star): 

    """ Generate data from y = w_star^T x, w_star is a k_star - sparse vector """

    w_star = torch.randn( [d,1] ) 

    w_star = randomly_zero(w_star, d - k_star) 

    X = torch.randn( [n,d] )

    Y_hat = torch.matmul( X, w_star)

    Y = torch.square(Y_hat) 


    return X, Y, w_star



def sparse_linear_data_dataset(n, train_dataset):


    num_dataset = train_dataset.data.shape[0]

    d = train_dataset.data[0].view(-1).shape[0]

    random_index = random.randint(0, num_dataset - 1)

    w_star = train_dataset.data[random_index].view((-1,1)).float()
    #w_star =  math.sqrt(d)*torch.nn.functional.normalize(train_dataset.data[random_index].view((-1,1)).float())

    k_star = torch.nonzero( w_star).size(0)

    X = 1/math.sqrt(n)*torch.randn( [n,d] )

    Y = torch.matmul( X, w_star)

    return X, Y, w_star, k_star

            


