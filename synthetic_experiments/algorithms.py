import torch


def topk(vector_in, k):

    in_shape = vector_in.shape

    vector = vector_in.view(-1)

    # Find the indices of the top k elements with the largest absolute values
    top_k_indices = torch.topk(torch.abs(vector), k=k).indices

    # Create a mask with zeros and ones
    mask = torch.zeros_like(vector)
    mask[top_k_indices] = 1

    # Apply the mask to zero out elements other than the top k
    result_vector = vector * mask

    result_vector = result_vector.view(in_shape)

    return result_vector

def WT_topk(parameters, k):

    for param in parameters:
        gradient = param.grad.data
        print(gradient)
        hessian = torch.autograd.grad(gradient, param, retain_graph=True ,create_graph=True)[0]
        print(hessian.shape)
        param.data =  topk(param.data - (hessian.inverse() @ gradient))                  
        