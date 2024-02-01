import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from algorithms import OBC


class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.linear(x)
    

class CustomDataset(Dataset):

    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index) -> Any:
        
        return self.data[index, : ], self.targets[index]




def get_custom_datasets(path, suffix=''):
    x_train = torch.load(os.path.join(path, f'features_train{suffix}.pt'))
    y_train = torch.load(os.path.join(path, f'targets_train{suffix}.pt'))
    x_val = torch.load(os.path.join(path, f'features_val{suffix}.pt'))
    y_val = torch.load(os.path.join(path, f'targets_val{suffix}.pt'))
    data_train = CustomDataset(data=x_train, targets=y_train)
    data_val = CustomDataset(data=x_val, targets=y_val)
    return data_train, data_val


def get_rn50x16openai_datasets(path):
    return get_custom_datasets(path)


def training_obc(model, criterion, num_epochs, training_loader ):

    loss_epochs = torch.zeros( num_epochs )

    for epoch in range(num_epochs):

        for i, data in enumerate(training_loader):

            # get input data

            inputs, labels = data

            # Forward pass
            predictions = model(inputs)

            # Compute the loss
            loss = criterion(predictions,labels)

            # Backward pass
            loss.backward()

            #print(loss.item())

            # Estimate the Hessian of each parameters of the model
            # In the case of linear regression, the Hessian is fixed and can be analytcally computed


            # Update parameters using the WoodTaylor optimizer, in this special case of Linear regression 

        
            for param in model_obc.parameters():

                gradient = param.grad.data
                
                data_new = param.data - gradient @ hessian.inverse()

                _ , mask_obc = OBC( param.data, h_inv, d, k )


                _, mask_topk = topk(data_new, k)

                param.data = mask_obc.view(param.shape) * data_new

                # param.data = torch.mul(mask_obc.view(param.shape), data_new)

                obc_maskdist_steps[expr, step] = torch.sum( torch.abs( mask_obc - mask_topk.view(mask_obc.shape)  ) )


            # compute the distance to the optimal weight
            obc_dist_steps[expr, step] = torch.norm(model_obc.linear.weight.data.view(w_star.shape) - w_star, 2)

            # Zero the gradients for the next iteration
            model_obc.linear.weight.grad.data.zero_()