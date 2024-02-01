import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader,Subset
from torchvision import transforms, utils


from algorithms import OBC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)



class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.linear(x)
    

class CustomDataset(Dataset):

    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data
        self.targets = targets
        self.dim = data.shape[1]

    def __len__(self):
        return len(self.targets)
    

    def __getitem__(self, index):
        
        return self.data[index, : ], self.targets[index]




def get_custom_datasets(path, subclasses ,suffix=''):

    
    x_train = torch.load(os.path.join(path, f'features_train{suffix}.pt'))
    y_train = torch.load(os.path.join(path, f'targets_train{suffix}.pt'))
    x_val = torch.load(os.path.join(path, f'features_val{suffix}.pt'))
    y_val = torch.load(os.path.join(path, f'targets_val{suffix}.pt'))

    sub_indices_train = (y_train[..., None] == subclasses).any(-1).nonzero(as_tuple=True)[0]

    sub_indices_val = (y_val[..., None] == subclasses).any(-1).nonzero(as_tuple=True)[0]

    data_train = CustomDataset(data=x_train[sub_indices_train,:], targets=y_train[sub_indices_train])
    data_val = CustomDataset(data=x_val[sub_indices_val,:], targets=y_val[sub_indices_val])

    return data_train, data_val


def get_rn50x16openai_datasets(path,subclasses):
    return get_custom_datasets(path,subclasses)


def training_obc(model, criterion, num_epochs, optimizer,training_loader, obc_sample_loader ,hessian_reg, k ,d  ):

    loss_epochs = torch.zeros( num_epochs )

    #train_data_full = training_loader.data

    #train_targets_full = training_loader.targets

    for epoch in range(num_epochs):

        # At the beginning of the step, randomly sample a data batch compute the OBC mask 

        X, Y = next(iter( obc_sample_loader ))

        X = X.to(device)
        Y = Y.to(device)


        hessian = X.t() @ X + hessian_reg

        h_inv = torch.linalg.inv(hessian)

        with torch.no_grad():

            W = model.linear.weight.data

            mask_obc =  torch.zeros(W.shape).to(device)

            # Applying OBC for each row
            for i in range(W.shape[0]):

                _ , mask_obc_i = OBC( W[i,:].view(1,-1), h_inv, d, k, device=device )

                mask_obc[i,:] = mask_obc_i.view(-1)


        # param.data = torch.mul(mask_obc.view(param.shape), data_new)

        for i, data in enumerate(training_loader):

            # get input data

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(inputs)

            # Compute the loss
            loss = criterion(predictions,labels)

            # Backward pass
            loss.backward()

            # Update in this epoches using the OBC mask: 
            optimizer.step()

            with torch.no_grad():
                model.linear.weight.data.mul_(mask_obc)
            

        with torch.no_grad():
            
            predictions = model(X)

            loss = criterion(predictions,Y)

            loss_epochs[epoch] = loss.item()

            print(loss.item())

    return loss_epochs


#data_path = '/home/dwu/Projects/Projects/Sparse SGD/woodtaylor_simulation'
data_path = '/nfs/scistore13/mondegrp/dwu/woodtaylor/woodtaylor_simulation'

num_class = 5

subclasses = torch.Tensor([i for i in range(num_class)])

dataset_train , dataset_val = get_rn50x16openai_datasets(data_path, subclasses)

train_loader = DataLoader( dataset_train, batch_size=256, shuffle=True )

# The number of samples used for 
obc_sample_loader = DataLoader( dataset_train, batch_size=1024, shuffle=True )


d = dataset_train.dim

sparsity = 0.5

k = int(sparsity*d )

model = LinearRegressionModel(d, num_class).to(device)

criterion = nn.CrossEntropyLoss()

num_epochs = 100

hessian_reg = 1e-3

print(model.linear.weight.shape)

optimizer = optim.SGD( model.parameters(), lr = 0.01, momentum=0.9)

training_loss = training_obc(model, criterion, num_epochs, optimizer,train_loader, obc_sample_loader ,hessian_reg, k ,d  )