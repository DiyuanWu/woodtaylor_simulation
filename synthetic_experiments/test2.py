import torch 


feature_train = torch.load("/home/dwu/Projects/Projects/Sparse SGD/woodtaylor_simulation/features_train.pt" )

targets_train = torch.load("/home/dwu/Projects/Projects/Sparse SGD/woodtaylor_simulation/targets_train.pt" )

targets_val = torch.load("/home/dwu/Projects/Projects/Sparse SGD/woodtaylor_simulation/targets_val.pt" )

print(targets_val.shape)