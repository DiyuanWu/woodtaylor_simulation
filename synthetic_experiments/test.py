import torch 

from algorithms import topk, OBC

from data_gen import sparse_linear_data


n = 20

d = 10

k_star = 5

k = 6

X, Y, w_star = sparse_linear_data(n, d, k_star)


hessian = torch.matmul(X.t(), X)

h_inv = hessian.inverse()

w_0 = torch.randn([1,d])


w_1, mask = OBC(w_0, h_inv, d,k)

print((w_1,mask))

