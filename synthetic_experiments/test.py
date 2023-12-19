import torch
import torch.nn as nn
import torch.optim as optim


from data_gen import sparse_linear_data

from algorithms import topk

n = 256

d = 128

k_star = 16

k = 64 # The sparsity of weights during training

X, Y, w_star = sparse_linear_data(n, d, k_star)

print(X.shape, Y.shape)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        return self.linear(x)

    
model = LinearRegressionModel(d, 1)

print(model.linear.weight.data.shape)

num_epochs = 100

criterion = nn.MSELoss()

for epoch in range(num_epochs):
    # Forward pass
    predictions = model.forward(X)

    # Compute the loss
    loss = criterion(predictions,Y)

    # Backward pass
    loss.backward()

    # Estimate the Hessian of each parameters of the model
    # In the case of linear regression, the Hessian is fixed and can be analytcally computed

    hessian = torch.matmul(X.t(), X)

    # Update parameters using the WoodTaylor optimizer, in this special case of Linear regression 
    for param in model.parameters():

        gradient = param.grad.data
        
        param.data =  topk( param.data - gradient @ hessian.inverse() , k) 

    # Zero the gradients for the next iteration
    model.linear.weight.grad.data.zero_()


    # Print the loss every 100 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(model.linear.weight.data.shape)