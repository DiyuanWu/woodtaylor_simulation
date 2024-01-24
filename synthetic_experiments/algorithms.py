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

    mask = torch.ones(w.shape).view(-1)

    idx_pruned = []

    eps = 1e-3


    for i in range(d-k):

        diag_hinv = torch.diag(H_inv).view(w.shape)

        diag_hinv_temp = diag_hinv + eps 

        val =  torch.div(w**2, diag_hinv_temp ).view(-1)

        #print(i)

        #print(val)

        if len(idx_pruned) > 0:

            # This pervent the pruned indices to be selected again

            val[ idx_pruned ] = (torch.max(val)+100) * torch.ones(len( idx_pruned) ) 

            #print(val)

        idx = torch.argmin(val).item()

        idx_pruned.append(idx)

        mask[idx] = 0

        w = w - (1/diag_hinv.view(-1)[idx].item()) * torch.mul( H_inv[:,idx], w )

        H_inv =  H_inv - (1/diag_hinv.view(-1)[idx].item()) * torch.matmul( H_inv[:,idx].view(-1,1) , H_inv[idx,:].view(1,-1) )

        w = torch.mul(w, mask.view(w.shape)) # this is to eliminate numerical errors, in principle, w should already be sparse

        

    return w, mask


