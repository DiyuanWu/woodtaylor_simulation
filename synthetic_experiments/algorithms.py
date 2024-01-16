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

    return result_vector, mask





def OBC(w_in, H_inv_in,d,k):

    '''
    The computation of optimal brain compression algorithm 
    '''

    w = w_in

    H_inv = H_inv_in

    mask = torch.ones(w.shape)

    for i in range(d-k):

        diag_hinv = torch.diag(H_inv).view(w.shape)

        #print(w.shape, diag_hinv.shape)

        val =  torch.div(w**2, diag_hinv )

        idx = torch.argmin(val).item()

        w = w - (1/diag_hinv.view(-1)[idx].item()) * torch.mul( H_inv[:,idx], w )

        H_inv =  H_inv - (1/diag_hinv.view(-1)[idx].item()) * torch.matmul( H_inv[:,idx] , H_inv[idx,:] )

        mask[0,idx] = 0

        

    return w, mask


